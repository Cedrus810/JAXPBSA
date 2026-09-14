"""M3–M4 correctness anchors: T1 Born ion, T2 Debye screening, plus
charge-conservation and batch-invariance checks (DESIGN.md §5)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxpbsa
import numpy as np
import pytest

from jaxpbsa.constants import COULOMB_K
from jaxpbsa.pb import PBParams, make_frame_solver, make_grid
from jaxpbsa.pb.charges import assign_density


def test_trilinear_charge_conservation():
    rng = np.random.default_rng(0)
    coords = rng.uniform(-5, 5, (20, 3))
    q = rng.uniform(-1, 1, 20)
    grid = make_grid(coords[None], h=0.5, padding=15)
    rho = assign_density(jnp.asarray(coords), jnp.asarray(q), grid)
    # 阈值跟着数组 dtype 走: fp32 的机器精度 ~1e-7, 写死 1e-10 只是在断言
    # 'x64 是开着的', 不是在断言电荷守恒。
    tol = 1e-10 if jnp.zeros(1, jaxpbsa.dtype()).dtype == jnp.float64 else 1e-5
    assert abs(float(rho.sum()) * grid.h**3 - q.sum()) < tol


def test_grid_shape_from_dime_set():
    coords = np.zeros((1, 3))
    grid = make_grid(coords[None], h=0.5, padding=22)
    assert grid.shape[0] in (65, 97, 129) and grid.shape[0] % 2 == 1
    assert grid.shape == (97, 97, 97)  # 2·22/0.5+1 = 89 → 下一档 97


@pytest.fixture(scope="module")
def born_setup():
    coords = np.zeros((1, 3))  # 单个离子, 恰在格心
    grid = make_grid(coords[None], 0.5, padding=22)
    # 解析 Born 对应锐利介电边界 → T1/T2 用 swin=0; 生产默认 swin=0.5 单独记录
    params = PBParams(eps_in=1.0, eps_out=80.0, ionic_strength_M=0.0, swin=0.0)
    solve = make_frame_solver(grid, [2.0], params)
    return solve, grid, params


def test_born_ion():
    """T1: q=+1, r=2 Å, ε 1/80, κ=0 → G_PB → −81.98 kcal/mol as h → 0.

    单点阈值卡不住离散化(锐利介电边界下 h=0.5 就有 ~4% 误差, 这是格式本身的
    一阶收敛, 不是 bug)。改成网格收敛检查: 误差必须随 h 单调下降并收敛到解析值。
    κ=0 时 DH 边界 u = BJERRUM_VAC·q/(ε_out·r) 对单个球对称离子是精确解, 所以
    小盒子不引入边界误差, 三档 h 可以直接比。
    """
    analytic = 0.5 * COULOMB_K * (1.0 / 80.0 - 1.0) / 2.0
    params = PBParams(eps_in=1.0, eps_out=80.0, ionic_strength_M=0.0, swin=0.0)
    errs = []
    for h in (1.0, 0.5, 0.25):
        grid = make_grid(np.zeros((1, 1, 3)), h, padding=8)
        out = make_frame_solver(grid, [2.0], params)(np.zeros((1, 3)), [1.0], [2.0])
        g = float(out["g_pb"])
        # 阈值跟着 params.tol 走, 不写死: 默认 tol 是按实测定的(§7.1C), 会变
        assert float(out["relres_solvent"]) <= params.tol
        assert float(out["relres_ref"]) <= params.tol  # DST 路径恒为 0
        errs.append(abs(g - analytic) / abs(analytic))
        print(f"\nBorn h={h}: G_PB = {g:.3f} (analytic {analytic:.3f}), "
              f"err {100 * errs[-1]:.2f}%, iters {int(out['iters_solvent'])}/"
              f"{int(out['iters_ref'])}")
    assert errs[0] > errs[1] > errs[2], errs      # 单调收敛
    assert errs[2] < 0.03, errs                    # h=0.25 落在 3% 内

    # 生产默认(swin=0.5 平滑)作为记录, 不设断言: 平滑壳层等效缩小介电边界,
    # 偏差方向可预期, 但它不该用解析 Born 来卡。
    grid = make_grid(np.zeros((1, 1, 3)), 0.25, padding=8)
    smooth = PBParams(eps_in=1.0, eps_out=80.0, ionic_strength_M=0.0, swin=0.5)
    g_smooth = float(
        make_frame_solver(grid, [2.0], smooth)(np.zeros((1, 3)), [1.0], [2.0])["g_pb"])
    print(f"\nBorn h=0.25 with swin=0.5 smoothing -> {g_smooth:.3f} kcal/mol")


def test_debye_screening(born_setup):
    """T2: 0.15 M 盐下远场 u(r)·r ∝ e^{−κr}; κ⁻¹ ≈ 7.86 Å."""
    solve, grid, _ = born_setup
    params = PBParams(eps_in=1.0, eps_out=78.5, ionic_strength_M=0.15, swin=0.0)
    solve_salt = make_frame_solver(grid, [2.0], params, return_potential=True)
    out = solve_salt(np.zeros((1, 3)), [1.0], [2.0])

    u = np.asarray(out["u_solvent"])
    c = grid.shape[0] // 2
    ix = np.arange(c + 16, c + 36)  # r ∈ [8, 18) Å
    r = (ix - c) * grid.h
    vals = u[ix, c, c]
    assert np.all(vals > 0)
    slope, _ = np.polyfit(r, np.log(vals * r), 1)[:2]
    kappa_inv = -1.0 / slope
    print(f"\nDebye: fitted κ⁻¹ = {kappa_inv:.2f} Å (expected 7.86)")
    assert abs(kappa_inv - 7.86) < 0.25


def test_batch_invariance(born_setup):
    solve, grid, params = born_setup
    vm = jax.vmap(solve.jitted, in_axes=(0, None, None))
    # 输入 dtype 必须和包的数组精度一致: x64 是开着的(归约需要 float64), 所以
    # 不带 dtype 的 jnp.ones 会是 float64, 和 fp32 的闭包常量对不上。
    dt = jaxpbsa.dtype()
    coords = jnp.zeros((3, 1, 3), dt)
    q = jnp.ones((1,), dt)
    radii = jnp.full((1,), 2.0, dt)
    out = vm(coords, q, radii)
    single = float(solve(np.zeros((1, 3)), [1.0], [2.0])["g_pb"])
    assert np.allclose(np.asarray(out["g_pb"]), single, rtol=1e-9)


def test_dtype_is_not_silently_promoted():
    """fp32 路径必须真的是 fp32。

    numpy 标量(grid.origin/h)在 JAX 里是强类型, 会把 fp32 数组提升回 fp64 而不报错;
    这个测试是那类静默提升的守卫 —— 它曾经真的发生过。
    """
    grid = make_grid(np.zeros((1, 1, 3)), 0.5, padding=8)
    assert all(type(o) is float for o in grid.origin), grid.origin
    assert type(grid.h) is float
    params = PBParams(eps_in=1.0, eps_out=80.0, ionic_strength_M=0.0, swin=0.0)
    solve = make_frame_solver(grid, [2.0], params, return_potential=True)
    out = solve(np.zeros((1, 3)), [1.0], [2.0])
    want = jnp.zeros(1, jaxpbsa.dtype()).dtype
    assert out["u_solvent"].dtype == want, (out["u_solvent"].dtype, want)
    # 能量归约固定走 fp64, 与数组 dtype 无关
    assert out["g_pb"].dtype == jnp.float64


def test_zero_source_does_not_divide_by_zero():
    """源项全零时不得出现 0/0。

    纯相对判据 ‖r‖/‖b‖ 在 b ≡ 0 时是 NaN, 而 `NaN > tol` 是 False —— 求解器会
    "收敛"并返回初值, 不报错。绝对项 atol 堵这个洞。
    (这说的是**源项全零**, 不是普通的净电荷为零体系 —— 后者 b 并不为零。)
    """
    from jaxpbsa.pb.operator import apply_operator, interior_mask, jacobi_diagonal
    from jaxpbsa.pb.solver import pcg_solve

    grid = make_grid(np.zeros((1, 1, 3)), 1.0, padding=8)
    dt = jaxpbsa.dtype()
    n = grid.shape
    ex = jnp.ones((n[0] - 1, n[1], n[2]), dt)
    ey = jnp.ones((n[0], n[1] - 1, n[2]), dt)
    ez = jnp.ones((n[0], n[1], n[2] - 1), dt)
    k2 = jnp.zeros(n, dt)
    A = lambda u: apply_operator(u, ex, ey, ez, k2, grid.h)
    d = jacobi_diagonal(ex, ey, ez, k2, grid.h)
    b = jnp.zeros(n, dt)  # 源项全零
    u, iters, relres, ok = pcg_solve(A, b, jnp.zeros(n, dt), d, interior_mask(grid),
                                     tol=1e-5, max_iter=100)
    assert bool(ok), "零源项应判为已收敛"
    assert np.isfinite(float(relres)), f"relres 不是有限值: {relres}"
    assert np.all(np.isfinite(np.asarray(u))), "解里有 NaN/Inf"
    assert float(jnp.abs(u).max()) == 0.0, "零源项 + 零边界的解必须是零"


def test_reports_non_convergence_instead_of_silently_returning():
    """迭代预算耗尽时必须返回 converged=False, 而不是悄悄返回一个未收敛的解。"""
    from jaxpbsa.pb.operator import apply_operator, interior_mask, jacobi_diagonal
    from jaxpbsa.pb.solver import pcg_solve

    grid = make_grid(np.zeros((1, 1, 3)), 0.5, padding=8)
    dt = jaxpbsa.dtype()
    n = grid.shape
    # ε 跳变 1:80, 不加预处理器, 只给 3 次迭代 —— 必然不收敛
    rng = np.random.default_rng(0)
    eps = jnp.asarray(rng.choice([1.0, 80.0], size=n), dt)
    hm = lambda a, b: 2 * a * b / (a + b)
    ex, ey, ez = (hm(eps[:-1], eps[1:]), hm(eps[:, :-1], eps[:, 1:]),
                  hm(eps[:, :, :-1], eps[:, :, 1:]))
    k2 = jnp.zeros(n, dt)
    A = lambda u: apply_operator(u, ex, ey, ez, k2, grid.h)
    d = jacobi_diagonal(ex, ey, ez, k2, grid.h)
    b = jnp.asarray(rng.normal(size=n), dt)
    _, iters, relres, ok = pcg_solve(A, b, jnp.zeros(n, dt), d, interior_mask(grid),
                                     tol=1e-12, max_iter=3)
    assert not bool(ok), "预算耗尽却报告收敛"
    assert float(relres) > 1e-12, "既然没收敛, 真残差应超过 tol"
    assert int(iters) <= 3 + 1, f"迭代数不该超过预算: {iters}"


def test_trajectory_state_carries_across_chunks():
    """分块调用时 final_state -> initial_state 必须接得上。

    不接的话每一块都在块边界冷启动, warm start 的收益全丢在块边界上。
    """
    grid = make_grid(np.zeros((1, 1, 3)), 1.0, padding=8)
    params = PBParams(eps_in=1.0, eps_out=80.0, ionic_strength_M=0.0, swin=0.0)
    solve = make_frame_solver(grid, [2.0], params)
    rng = np.random.default_rng(0)
    traj = np.cumsum(rng.normal(0, 0.05, (6, 1, 3)), axis=0)

    whole, st_end = solve.trajectory(traj, [1.0], [2.0], warm=True)
    a, st_a = solve.trajectory(traj[:3], [1.0], [2.0], warm=True)
    b, _ = solve.trajectory(traj[3:], [1.0], [2.0], warm=True, initial_state=st_a)

    assert st_end.shape == grid.shape and np.all(np.isfinite(np.asarray(st_end)))
    g_whole = np.asarray(whole["g_pb"])
    g_split = np.concatenate([np.asarray(a["g_pb"]), np.asarray(b["g_pb"])])
    # 同一 tol 下, 分块与整条的能量应当一致(warm start 只改初值, 不改解)
    assert np.allclose(g_whole, g_split, rtol=1e-4), (g_whole, g_split)
    # 收敛状态也要一致, 不能只比能量
    c_whole = np.asarray(whole["converged"])
    c_split = np.concatenate([np.asarray(a["converged"]), np.asarray(b["converged"])])
    assert np.array_equal(c_whole, c_split), (c_whole, c_split)
    assert c_whole.all(), "这个体系本该全部收敛"

    # warm=False 必须**逐位**忽略 initial_state: 冷启动是 warm start 的对照组,
    # 不能含任何来自其他帧或调用方的信息, 否则比较不公平。
    # (曾经不是这样: carry 被当成每帧的固定初值传了进去, 能量差 ~2e-6 相对。)
    cold_a, _ = solve.trajectory(traj, [1.0], [2.0], warm=False)
    cold_b, _ = solve.trajectory(traj, [1.0], [2.0], warm=False,
                                 initial_state=st_a)
    assert np.array_equal(np.asarray(cold_a["g_pb"]), np.asarray(cold_b["g_pb"])), (
        "warm=False 仍受 initial_state 影响", cold_a["g_pb"], cold_b["g_pb"])


def test_static_slice_morphology_matches_reference():
    """静态 slice 的膨胀/腐蚀必须和朴素参考实现逐位相同。

    这条和电荷无关 —— 表面形态只由坐标和半径决定。它验的是 §0.16 那个
    `dynamic_slice` → `lax.slice` 的改写没有改变表面定义。
    """
    from jaxpbsa.pb.surface import ball_offsets, dilate, erode

    rng = np.random.default_rng(0)
    m = jnp.asarray(rng.random((21, 23, 19)) < 0.12)
    offs = np.asarray(ball_offsets(1.4, 0.5))

    def ref_dilate(mask, offsets):
        """朴素参考: numpy 逐偏移移位取 max, 界外为 False。"""
        out = np.asarray(mask).copy()
        src = np.asarray(mask)
        for o in offsets:
            sh = np.zeros_like(src)
            sl_dst, sl_src = [], []
            for ax in range(3):
                d = int(o[ax])
                n = src.shape[ax]
                sl_dst.append(slice(max(d, 0), n + min(d, 0)))
                sl_src.append(slice(max(-d, 0), n + min(-d, 0)))
            sh[tuple(sl_dst)] = src[tuple(sl_src)]
            out |= sh
        return out

    assert np.array_equal(np.asarray(dilate(m, offs)), ref_dilate(m, offs))
    assert np.array_equal(np.asarray(erode(m, offs)), ~ref_dilate(~m, offs))
    assert offs.shape[0] > 50, "偏移太少, 没覆盖到融合路径"


def test_zero_charges_give_zero_energy_and_success():
    """全零电荷: 能量为零、诊断量有限、收敛状态为成功。

    这条验的是**求解器**, 与形态学无关 —— 表面照样按坐标和半径构建。
    """
    grid = make_grid(np.zeros((1, 1, 3)), 1.0, padding=8)
    params = PBParams(eps_in=1.0, eps_out=80.0, ionic_strength_M=0.0, swin=0.0)
    out = make_frame_solver(grid, [2.0], params)(np.zeros((1, 3)), [0.0], [2.0])
    assert bool(out["converged"]), "零电荷体系应判为收敛"
    assert np.isfinite(float(out["relres_solvent"]))
    assert np.isfinite(float(out["relres_ref"]))
    assert abs(float(out["g_pb"])) < 1e-9, f"零电荷的 G_PB 应为零, 得到 {out['g_pb']}"
