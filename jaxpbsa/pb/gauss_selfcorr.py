"""DelPhi 高斯 ε 的亚网格**自项**修正（RESULTS §18.11–18.12）。

问题：高斯 ε 在电荷中心按 r² 升高，内层 ℓ = σR·√(ε_in/Δε) ≈ 0.2 Å。h=0.5 时自项 G_R(i,i) 严重
欠解析（单离子 −63%），交叉项 G_R(i,j) 却已准（双原子 0.6–3.3%）。

修正：原子 i 中心附近 1 − ρ_mol ≈ C_i r²/a_i²（a_i = σR_i，C_i = Π_{j≠i}(1 − ρ_j(r_i))），其自项
网格误差等于**等效孤立原子** b_i = a_i/√C_i 在同 h、同相位下的误差（双原子、埋藏原子上修正后
自项与 h/相位无关到 ±0.13% / ±0.4%）。孤立原子有精确一维积分，于是

    ΔG_pb = ½ Σ_i q_i² · T(b_i/h, f_i) / h

T(β, f) = h·[G_exact(b) − G_grid(b, h, f)]（kcal·Å/mol，单位电荷）。离散问题尺度不变，所以只在
h=1 上打一次表；f 是电荷在格子里的分数坐标，按立方对称折叠到 0 ≤ f₁ ≤ f₂ ≤ f₃ ≤ ½。
表只依赖 ε_in/ε_out 与离散格式（节点 ε + 面上调和平均、三线性电荷、同一求解器）。
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import COULOMB_K, KT_TO_KCAL
from .charges import assign_density, interpolate, source_term
from .grid import GridSpec
from .operator import (apply_operator, coulomb_boundary_values, interior_mask, jacobi_diagonal,
                       shell_indices)
from .dst import solve_reference
from .multigrid import build_levels, make_preconditioner
from .solver import pcg_solve

_HERE = os.path.dirname(os.path.abspath(__file__))
# 不放 data/: .gitignore 的 `data/` 会把它一起忽略
TABLE_PATH = os.path.join(_HERE, "_tables", "gauss_selfcorr_ein1_eout78.5.npz")
F_NODES = np.array([0.0, 0.125, 0.25, 0.375, 0.5])


def exact_unit(b: float, eps_in: float, eps_out: float) -> float:
    """孤立原子 ε(r) = ε_out − Δε e^{−r²/b²} 的单位电荷 G_R(i,i)（kcal/mol）。= K/b。"""
    from scipy.integrate import quad

    d = eps_out - eps_in
    ep = lambda r: eps_out - d * np.exp(-(r / b) ** 2)
    top = 12.0 * b
    v = quad(lambda r: (1 / ep(r) - 1 / eps_in) / r ** 2, 0.0, top, limit=2000,
             points=[b, 2 * b, 3 * b])[0] + (1 / eps_out - 1 / eps_in) / top
    return COULOMB_K * v


def _iso_solver(b: float, h: float, eps_in: float, eps_out: float):
    """编译一次: x0 [3](格点 0 附近的电荷位置) -> 孤立原子网格 G_R(i,i)(单位电荷, kcal/mol)。"""
    pad = max(8.0 * h, 3.2 * b + 4.0 * h)
    n = int(np.ceil(2 * pad / h)) + 1
    n += (n + 1) % 2  # 奇数: 中心格点在 0
    grid = GridSpec(origin=(-(n - 1) / 2 * h,) * 3, shape=(n, n, n), h=float(h))
    ax = jnp.asarray(grid.origin[0] + np.arange(n) * h)
    P = jnp.stack(jnp.meshgrid(ax, ax, ax, indexing="ij"), -1)
    sh, fl = shell_indices(grid)
    mask = interior_mask(grid)
    k0 = jnp.zeros(grid.shape, P.dtype)

    @jax.jit
    def f(x0):
        e = eps_out - (eps_out - eps_in) * jnp.exp(-((P - x0) ** 2).sum(-1) / b ** 2)
        hm = lambda a, c: 2 * a * c / (a + c)
        ex, ey, ez = hm(e[:-1], e[1:]), hm(e[:, :-1], e[:, 1:]), hm(e[:, :, :-1], e[:, :, 1:])
        c, q = x0[None], jnp.ones(1, P.dtype)
        rhs = source_term(assign_density(c, q, grid), grid)
        u0 = k0.reshape(-1).at[fl].set(coulomb_boundary_values(sh, c, q, eps_out)).reshape(grid.shape)
        pre = make_preconditioner(build_levels(ex, ey, ez, k0, h))
        us, _, _, ok = pcg_solve(lambda u: apply_operator(u, ex, ey, ez, k0, h), rhs, u0,
                                 jacobi_diagonal(ex, ey, ez, k0, h), mask,
                                 precond=pre, tol=1e-7, max_iter=5000)
        one = [jnp.full_like(ex, eps_in), jnp.full_like(ey, eps_in), jnp.full_like(ez, eps_in)]
        u0r = k0.reshape(-1).at[fl].set(coulomb_boundary_values(sh, c, q, eps_in)).reshape(grid.shape)
        ur = solve_reference(rhs, u0r, lambda u: apply_operator(u, *one, k0, h), eps_in, h)
        return interpolate(us - ur, c, grid)[0] * KT_TO_KCAL, ok

    return f


def grid_unit(b: float, h: float, f, eps_in: float, eps_out: float) -> float:
    """同一离散格式下孤立原子的网格 G_R(i,i)：电荷在格点 0 + f·h。"""
    g, ok = _iso_solver(b, h, eps_in, eps_out)(jnp.asarray(np.asarray(f, float) * h))
    if not bool(ok):
        raise RuntimeError(f"孤立原子求解未收敛 b={b} h={h} f={f}")
    return float(g)


def build_table(betas, eps_in=1.0, eps_out=78.5, path=TABLE_PATH, verbose=True):
    """T[β, f1, f2, f3] 在 h=1 上。只算折叠后的 35 个相位, 其余按对称填。"""
    betas = np.asarray(betas, float)
    nf = len(F_NODES)
    T = np.full((len(betas), nf, nf, nf), np.nan)
    for ib, beta in enumerate(betas):
        ex = exact_unit(beta, eps_in, eps_out)
        solve = _iso_solver(beta, 1.0, eps_in, eps_out)
        for i in range(nf):
            for j in range(i, nf):
                for k in range(j, nf):
                    g, ok = solve(jnp.asarray([F_NODES[i], F_NODES[j], F_NODES[k]]))
                    if not bool(ok):
                        raise RuntimeError(f"孤立原子求解未收敛 β={beta}")
                    v = ex - float(g)
                    for p in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
                        T[(ib,) + p] = v
        if verbose:
            print(f"  β={beta:6.2f}: T ∈ [{np.nanmin(T[ib]):9.2f}, {np.nanmax(T[ib]):9.2f}]"
                  f"  (exact·β = {ex * beta:9.2f})", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, betas=betas, f_nodes=F_NODES, T=T, eps_in=eps_in, eps_out=eps_out)
    return T


def load_table(eps_in: float, eps_out: float, path=TABLE_PATH):
    z = np.load(path)
    if not (np.isclose(z["eps_in"], eps_in) and np.isclose(z["eps_out"], eps_out)):
        raise ValueError(f"自项修正表是 ε {float(z['eps_in'])}/{float(z['eps_out'])} 的, "
                         f"当前 {eps_in}/{eps_out}: 用 build_table 重建")
    return {k: z[k] for k in ("betas", "f_nodes", "T")}


def self_correction(coords, q, radii, grid: GridSpec, sigma: float, table) -> jnp.ndarray:
    """ΔG_pb(kcal/mol) = ½ Σ q_i² T(b_i/h, f_i)/h。radii == 0 的原子(屏蔽)既不贡献也不参与 C_i。"""
    dt = coords.dtype
    betas = jnp.asarray(np.log(table["betas"]), dt)
    T = jnp.asarray(table["T"], dt)
    fn = jnp.asarray(table["f_nodes"], dt)
    live = radii > 0
    a = sigma * radii
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    aj2 = jnp.where(live, a * a, 1.0)[None, :]
    rho = jnp.where(live[None, :] & (d2 <= 9.0 * aj2) & ~jnp.eye(coords.shape[0], dtype=bool),
                    jnp.exp(-d2 / aj2), 0.0)
    logC = jnp.log(jnp.maximum(1.0 - rho, 1e-30)).sum(1)
    b = jnp.where(live, a, 1.0) * jnp.exp(-0.5 * logC)
    lb = jnp.log(b / grid.h)
    # 分数坐标 → 折叠到 [0, ½]、排序 → 在 5 点节点上三线性插值
    f = (coords - jnp.asarray(grid.origin, dt)) / grid.h
    f = f - jnp.floor(f)
    f = jnp.sort(jnp.minimum(f, 1.0 - f), axis=1)
    step = fn[1] - fn[0]
    u = jnp.clip(f / step, 0.0, len(table["f_nodes"]) - 1 - 1e-6)
    i0 = jnp.floor(u).astype(jnp.int32)
    w = u - i0
    # β: log 空间线性插值, 超出表范围截断(1YCR: h=0.5 β∈[4.3,12], h=0.25 β∈[8.7,24])
    nb = betas.shape[0]
    ub = jnp.clip((lb - betas[0]) / (betas[1] - betas[0]), 0.0, nb - 1 - 1e-6)
    ib = jnp.floor(ub).astype(jnp.int32)
    wb = ub - ib

    def at(ibb):
        acc = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wt = ((w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1])
                          * (w[:, 2] if dz else 1 - w[:, 2]))
                    acc = acc + wt * T[ibb, i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return acc

    Ti = at(ib) * (1 - wb) + at(ib + 1) * wb
    return 0.5 * jnp.sum(jnp.where(live, q * q * Ti / grid.h, 0.0))
