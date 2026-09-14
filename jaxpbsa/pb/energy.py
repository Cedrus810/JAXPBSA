"""PB polar solvation energy via the two-solve method (DESIGN.md §3.8).

    G_PB = ½ Σ qᵢ (u_solvent(rᵢ) − u_ref(rᵢ)) · kT   [kcal/mol]

solvent solve: full ε/κ̄² maps + single-centre DH boundary (bcfl sdh).
reference solve: uniform ε_in, κ̄² = 0 + analytic per-atom Coulomb boundary.
Both on the same grid/charge assignment → discretised self-energy cancels in
the difference. (DST 直解参考方程是下一轮的 2× 白送加速, 先用 PCG 保证正确。)

用法: frame_fn = make_frame_solver(grid, radii, params)  # 一次编译
      g, info = frame_fn(coords, q)                      # 单帧
      batched = jax.vmap(frame_fn, in_axes=(0, None))    # 或按帧 vmap
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .. import ACCUM_DTYPE, dtype as _dtype
from ..constants import KT_TO_KCAL, PROBE_RADIUS, debye_kappa2
from .charges import assign_density, interpolate, source_term
from .grid import GridSpec
from .operator import (
    apply_operator,
    coulomb_boundary_values,
    dh_boundary_values,
    interior_mask,
    jacobi_diagonal,
    shell_indices,
)
from .dst import solve_reference
from .multigrid import build_levels, make_preconditioner
from .solver import pcg_solve
from .surface import ball_offsets, build_maps


#: "auto" 模式下启用 MG 的节点数下限。实测交叉点在 0.9–2.1 M 之间(两张卡一致),
#: 取 1.5 M 居中。低于它 MG 的粗层启动延迟压不住, 反而比 Jacobi 慢。
MG_NODE_THRESHOLD = 1_500_000


@dataclass(frozen=True)
class PBParams:
    eps_in: float = 1.0
    eps_out: float = 78.5
    ionic_strength_M: float = 0.15
    temperature_K: float = 298.15
    probe_radius: float = PROBE_RADIUS  # Å, 单源见 constants.py
    ion_radius: float = 2.0  # Å
    swin: float = 0.5  # ε/κ̄² 调和平滑半径, Å (≤0 关闭)
    ref_solver: str = "dst"  # "dst" (精确直解, 默认) | "pcg" (对照用)
    # 溶剂方程的预处理器。MG = V-cycle 预处理的 CG(不能独立求解: ε 跳变 1:80 会发散,
    # 见 multigrid.py)。"auto" 按网格规模选 —— **MG 在粗网格上是负收益**:
    #
    #   节点数    配置       fp32 MG / fp32 Jacobi    2080 Ti    5080
    #   0.91 M    h=1.0                               0.87x     0.45x  <- MG 更慢
    #   2.15 M    h=0.75                              1.27x     1.36x
    #   5.00 M    h=0.5                               1.74x     2.81x
    #
    # 交叉点落在 0.9–2.1 M 节点之间, **两张卡一致**(幅度不同: 粗网格上卡越快 MG 越吃亏,
    # 因为粗层那几个小网格是 kernel 启动延迟主导; 细网格上则相反)。
    precond: str = "auto"  # "auto" | "mg" | "jacobi"
    mg_nu: int = 2       # 每层 pre/post 光滑次数; 实测 V(3,3) 略优于 V(2,2)
    mg_coarse_sweeps: int = 50  # 最粗层 Jacobi 轮数
    # 参考边界的原子分块大小。实测 S4/h=0.5: ab=1 12.9ms -> ab=128 3.1ms -> ab=256 2.4ms;
    # 中间量 S×ab, ab=128 时 113 MB。默认取 128 兼顾速度与显存。
    boundary_atom_block: int = 128
    # fp64 下 tol 1e-7 与 1e-5 的 G_PB 只差 0.002 kcal/mol, 而离散误差是 40 kcal/mol
    # 量级 —— 1e-7 是纯粹的过度求解, 白花 1.8x 时间 (DESIGN.md §7.1C)。
    tol: float = 1e-5
    max_iter: int = 2000


def make_frame_solver(
    grid: GridSpec,
    radii,  # [N] Å —— 只用于确定静态 offset 尺寸(host 侧取 max)
    params: PBParams,
    return_potential: bool = False,
):
    """编译并返回 frame_fn(coords[N,3], q[N]) -> dict(g_pb, iters, relres, ...).

    grid/radii 的尺寸信息烘焙进闭包 → 同一 grid 只编译一次。
    """
    r_max = float(np.max(np.asarray(radii)))
    shell_xyz, shell_flat = shell_indices(grid)
    center = jnp.asarray(
        np.asarray(grid.origin) + 0.5 * (np.asarray(grid.shape) - 1) * grid.h
    )
    mask = interior_mask(grid)
    raster_offsets = ball_offsets(r_max + grid.h, grid.h)
    probe_offsets = ball_offsets(params.probe_radius, grid.h)
    ion_offsets = ball_offsets(params.ion_radius, grid.h)
    smooth_offsets = (
        ball_offsets(params.swin, grid.h) if params.swin > 0 else ball_offsets(0.0, grid.h)
    )
    kappa2_phys = debye_kappa2(
        params.ionic_strength_M, params.eps_out, params.temperature_K
    )
    kappa = float(np.sqrt(kappa2_phys)) if kappa2_phys > 0 else 0.0
    shape = grid.shape
    if params.precond not in ("auto", "mg", "jacobi"):
        raise ValueError(f'precond 必须是 "auto"/"mg"/"jacobi", 得到 {params.precond!r}')
    use_mg = params.precond == "mg" or (
        params.precond == "auto" and int(np.prod(shape)) >= MG_NODE_THRESHOLD)
    dt = _dtype()
    shell_xyz = shell_xyz.astype(dt)
    center = center.astype(dt)

    def _frame_impl(coords: jnp.ndarray, q: jnp.ndarray, radii: jnp.ndarray,
                    u_prev: jnp.ndarray | None = None,
                    u_ref_given: jnp.ndarray | None = None):
        """-> (能量/诊断 dict, u_solvent, u_ref)。

        u_prev: 上一帧**内部**势(warm start)。
        u_ref_given: 直接给定参考势, 跳过参考求解 —— 供 C/R/L 复用
        (u_ref_C = u_ref_R + u_ref_L, 见 make_triplet_solver)。
        """
        b = source_term(assign_density(coords, q, grid), grid)

        maps = build_maps(
            coords, radii, grid, params.eps_in, params.eps_out,
            params.probe_radius, params.ion_radius, kappa2_phys,
            smooth_offsets=smooth_offsets,
            raster_offsets=raster_offsets, probe_offsets=probe_offsets,
            ion_offsets=ion_offsets,
        )
        q_net = q.sum()
        # a 只在**有效原子**上取: 被屏蔽的原子会被挪到盒外(见 make_triplet_solver),
        # 直接取 max 会被它们主导, DH 边界就错了。q==0 是屏蔽原子的标记。
        dist = jnp.linalg.norm(coords - center, axis=1)
        active = q != 0
        a = jnp.max(jnp.where(active, dist, -jnp.inf)) + params.ion_radius
        ub = dh_boundary_values(shell_xyz, center, q_net, params.eps_out, kappa, a)
        # warm start: 内部取上一帧的解, 边界**永远用本帧新算的 Dirichlet 值** ——
        # 构象变了 a(离子排除半径)就变, 沿用上一帧边界是错的。
        u_start = jnp.zeros(shape, dt) if u_prev is None else (u_prev * mask)
        u0 = u_start.reshape(-1).at[shell_flat].set(ub).reshape(shape)

        def apply_solv(u):
            return apply_operator(u, maps["eps_x"], maps["eps_y"], maps["eps_z"],
                                  maps["kbar2"], grid.h)

        diag_s = jacobi_diagonal(maps["eps_x"], maps["eps_y"], maps["eps_z"],
                                 maps["kbar2"], grid.h)
        pre = None
        if use_mg:
            levels = build_levels(maps["eps_x"], maps["eps_y"], maps["eps_z"],
                                  maps["kbar2"], grid.h)
            pre = make_preconditioner(levels, nu1=params.mg_nu, nu2=params.mg_nu,
                                      coarse_sweeps=params.mg_coarse_sweeps)
        u_solv, it_s, rr_s, ok_s = pcg_solve(apply_solv, b, u0, diag_s, mask,
                                             precond=pre, tol=params.tol,
                                             max_iter=params.max_iter)

        # 参考方程: 均匀 ε_in, κ̄² = 0, 解析库仑 Dirichlet 边界
        eps_x = jnp.full((shape[0] - 1, shape[1], shape[2]), params.eps_in, dt)
        eps_y = jnp.full((shape[0], shape[1] - 1, shape[2]), params.eps_in, dt)
        eps_z = jnp.full((shape[0], shape[1], shape[2] - 1), params.eps_in, dt)
        zero = jnp.zeros(shape, dt)

        def apply_ref(u):
            return apply_operator(u, eps_x, eps_y, eps_z, zero, grid.h)

        if u_ref_given is not None:
            # 参考势由调用方给出(C/R/L 复用), 省掉一次库仑边界 + 一次 DST
            u_ref = u_ref_given
            it_r, rr_r = jnp.asarray(0, jnp.int32), jnp.zeros((), dt)
            ok_r = jnp.asarray(True)
            return _finish(u_solv, u_ref, coords, q, it_s, rr_s, ok_s, it_r, rr_r, ok_r)
        ubr = coulomb_boundary_values(shell_xyz, coords, q, params.eps_in,
                                      atom_block=params.boundary_atom_block)
        u0r = jnp.zeros(shape, dt).reshape(-1).at[shell_flat].set(ubr).reshape(shape)
        if params.ref_solver == "dst":
            # 常系数 Poisson -> DST 直解, 精确且零迭代 (dst.py 有实测依据)
            u_ref = solve_reference(b, u0r, apply_ref, params.eps_in, grid.h)
            it_r, rr_r = jnp.asarray(0, jnp.int32), jnp.zeros((), dt)
            ok_r = jnp.asarray(True)
        else:
            diag_r = jacobi_diagonal(eps_x, eps_y, eps_z, zero, grid.h)
            # 关键字传参: precond 是第 6 个位置参数, 位置传 tol 会静默错位
            u_ref, it_r, rr_r, ok_r = pcg_solve(apply_ref, b, u0r, diag_r, mask,
                                                tol=params.tol,
                                                max_iter=params.max_iter)

        return _finish(u_solv, u_ref, coords, q, it_s, rr_s, ok_s, it_r, rr_r, ok_r)

    def _finish(u_solv, u_ref, coords, q, it_s, rr_s, ok_s, it_r, rr_r, ok_r):
        u_reac = interpolate(u_solv, coords, grid) - interpolate(u_ref, coords, grid)
        return ({
            # 归约走 fp64: u_reac 是两个自能量级势的差, 逐原子求和会放大抵消误差
            "g_pb": 0.5 * jnp.sum(q * u_reac, dtype=ACCUM_DTYPE) * KT_TO_KCAL,
            "iters_solvent": it_s,
            "iters_ref": it_r,
            "relres_solvent": rr_s,
            "relres_ref": rr_r,
            # 收敛失败必须可被下游检测 —— benchmark 要拒绝这类结果, 不能计进加速比
            "converged": ok_s & ok_r,
        }, u_solv, u_ref)

    @jax.jit
    def frame_fn(coords, q, radii, u_prev=None):
        """单帧入口。整张势场只在 return_potential=True 时返回 —— 193³ 的 fp32
        势场是 27.4 MiB, 调用端保留一万帧单 species 就是 268 GiB。"""
        out, u, _ = _frame_impl(coords, q, radii, u_prev)
        return {**out, "u_solvent": u} if return_potential else out

    def _scan_warm(coords_traj, q, radii, state):
        def step(carry, x):
            out, u, _ = _frame_impl(x, q, radii, carry)
            return u, out
        return jax.lax.scan(step, state, coords_traj)

    def _scan_cold(coords_traj, q, radii, state):
        """冷启动: **每帧都从零势出发**, 既不用上一帧的解, 也不用 initial_state。

        原先这里把 carry 传给了 `_frame_impl` 当 u_prev —— 那是"每帧从同一个给定初值
        出发", 不是冷启动。作为 warm start 的对照组, 它必须不含任何来自其他帧或调用方
        的信息, 否则 cold/warm 的比较不公平(实测两者能量差 ~2e-6 相对)。
        """
        zero = jnp.zeros(shape, dt)

        def step(carry, x):
            out, _, _ = _frame_impl(x, q, radii, zero)
            return carry, out

        return jax.lax.scan(step, state, coords_traj)

    # **工厂里建一次、jit 一次**。原先每次调用 solve_trajectory 都重新定义 step,
    # 于是每次调用都重新 tracing/编译 —— 量出来的"scan 比单帧慢 7.6×"里混着编译时间,
    # 不能归因为设备执行损失。
    _traj_warm = jax.jit(_scan_warm)
    _traj_cold = jax.jit(_scan_cold)

    def solve_trajectory(coords_traj, q, radii_arrays=None, warm: bool = True,
                         initial_state=None):
        """[T,N,3] -> (每帧能量 dict, final_state)。

        `initial_state` / 返回的 `final_state` 是**上一帧的内部势, 留在设备上**。
        长轨迹分块调用时把它传下去, 否则每一块都从零势冷启动 —— warm start 的
        收益会在块边界上全部丢掉。

        `warm=False` 时**每帧都从零势出发**, initial_state 既不作初值也不参与递推 ——
        冷启动的定义是"不含任何来自其他帧或调用方的信息", 否则它作为 warm start 的
        对照组就不公平。返回的 final_state 等于传入值, 仅为接口对称。

        每帧的 `converged` 保留在输出里。**失败帧的势不会传给下一帧**, 否则一个坏帧
        会污染其后所有帧的初值。

        势场只作为 scan 的 carry, 不进输出: T=10000 时单 species 的势场是 268 GiB。
        """
        c = jnp.asarray(coords_traj, dtype=dt)
        qq = jnp.asarray(q, dtype=dt)
        r = jnp.asarray(radii_arrays if radii_arrays is not None else radii, dtype=dt)
        state = jnp.zeros(shape, dt) if initial_state is None else initial_state
        final_state, outs = (_traj_warm if warm else _traj_cold)(c, qq, r, state)
        return outs, final_state

    def solve(coords, q, radii_arrays=None):
        c = jnp.asarray(coords, dtype=dt)
        qq = jnp.asarray(q, dtype=dt)
        r = jnp.asarray(radii_arrays if radii_arrays is not None else radii, dtype=dt)
        return frame_fn(c, qq, r)

    def _species(coords, q, radii, keep):
        """把不属于该 species 的原子**挪到盒外并清零电荷/半径**。

        为什么不直接切片: 切出来的 receptor/ligand 原子数和 complex 不同 -> 形状不同
        -> **各要一次编译**(实测每次 ~8.5 s)。补齐到 N_complex 后三者共用同一份编译。

        挪到盒外而不是只把半径设 0: `rasterize_spheres` 判的是 `d2 <= radii²`,
        半径 0 时若原子恰好落在格点上 d2 == 0 仍会被标记为占据。挪出界后
        `inb` 恒为 False, 不会产生任何介电占据。
        """
        far = jnp.asarray(grid.origin, dt) - jnp.asarray(1e4, dt)
        return (jnp.where(keep[:, None], coords, far),
                jnp.where(keep, q, jnp.zeros((), dt)),
                jnp.where(keep, radii, jnp.zeros((), dt)))

    @jax.jit
    def _triplet(coords, q, radii, keep_r, keep_l):
        """ΔG_PB = G_C − G_R − G_L，参考解 3 次降 2 次。

        公共网格(§2)上参考方程处处 ε_in、κ̄²=0、源项对电荷线性、库仑边界也线性，
        因此 **u_ref_C = u_ref_R + u_ref_L**（组合的是**势**）。

        **不能把能量相加**: 参考能量含 R–L 交叉项 ½Σ_R q·u_ref_L + ½Σ_L q·u_ref_R。
        这里用组合后的势场在 complex 的全部原子上求和, 交叉项自动包含在内。

        **溶剂方程不能复用**: C/R/L 的介电图不同(配体在场时受体表面被遮挡),
        算子本身就不一样 —— 溶剂解仍是 3 次。
        """
        cr, qr, rr_ = _species(coords, q, radii, keep_r)
        cl, ql, rl = _species(coords, q, radii, keep_l)
        out_r, _, u_ref_r = _frame_impl(cr, qr, rr_)
        out_l, _, u_ref_l = _frame_impl(cl, ql, rl)
        out_c, _, _ = _frame_impl(coords, q, radii, u_ref_given=u_ref_r + u_ref_l)
        d = out_c["g_pb"] - out_r["g_pb"] - out_l["g_pb"]
        return {
            "g_pb_complex": out_c["g_pb"],
            "g_pb_receptor": out_r["g_pb"],
            "g_pb_ligand": out_l["g_pb"],
            "delta_g_pb": d,
            "iters": jnp.stack([out_c["iters_solvent"], out_r["iters_solvent"],
                                out_l["iters_solvent"]]),
            "relres": jnp.stack([out_c["relres_solvent"], out_r["relres_solvent"],
                                 out_l["relres_solvent"]]),
            "converged": out_c["converged"] & out_r["converged"] & out_l["converged"],
        }

    def solve_triplet(coords, q, receptor_idx, ligand_idx, radii_arrays=None):
        """ΔG_PB 单帧。receptor/ligand 用**索引**给出, 内部补齐到 complex 的原子数。"""
        n = int(np.asarray(coords).shape[0])
        kr = np.zeros(n, bool); kr[np.asarray(receptor_idx)] = True
        kl = np.zeros(n, bool); kl[np.asarray(ligand_idx)] = True
        if (kr & kl).any() or not (kr | kl).all():
            raise ValueError("receptor/ligand 必须互补且覆盖全部原子")
        r = radii_arrays if radii_arrays is not None else radii
        return _triplet(jnp.asarray(coords, dt), jnp.asarray(q, dt),
                        jnp.asarray(r, dt), jnp.asarray(kr), jnp.asarray(kl))

    solve.jitted = frame_fn  # 供 vmap/scan 直接复用同一份编译产物
    solve.trajectory = solve_trajectory
    solve.triplet = solve_triplet
    return solve


def _reference_field_is_additive_note() -> str:
    """ΔG_PB 汇总时参考解可以从 3 次降到 2 次 —— 设计约定, 待 trajectory/ 接入。

    公共网格(§2)上参考方程处处 ε_in、κ̄²=0、源项对电荷线性、库仑边界也线性:

        u_ref_complex = u_ref_receptor + u_ref_ligand

    **组合的是势, 不是能量。** 参考能量里含 R–L 交叉项
    ½Σ_R q·u_ref_L + ½Σ_L q·u_ref_R, 直接加能量会把它丢掉。

    **溶剂方程不能这样复用**: C/R/L 的介电图不同(配体在场时受体表面被遮挡),
    算子本身不一样, 溶剂解仍是 3 次。
    """
    return "u_ref_C = u_ref_R + u_ref_L (势相加, 非能量相加)"
