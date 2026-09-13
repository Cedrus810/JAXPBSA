"""Matrix-free 7-point finite-difference linear PB operator (DESIGN.md §3.6).

    (Au)_i = Σ_faces ε_f (u_i − u_j)/h² + κ̄²_i u_i

ε_f are harmonic face means (surface.py). With κ̄² ≥ 0 and ε > 0 the interior
operator is symmetric positive definite → CG/PCG and multigrid both apply.
Boundary nodes are Dirichlet (values supplied by the caller, never updated).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..constants import BJERRUM_VAC
from .grid import GridSpec


def apply_operator(
    u: jnp.ndarray,
    eps_x: jnp.ndarray,  # [nx-1,ny,nz]
    eps_y: jnp.ndarray,  # [nx,ny-1,nz]
    eps_z: jnp.ndarray,  # [nx,ny,nz-1]
    kbar2: jnp.ndarray,
    h: float,
) -> jnp.ndarray:
    """(Au)ᵢ = Σ_faces ε_f (uᵢ − u_j)/h² + κ̄²ᵢuᵢ = −∇·(ε∇u) + κ̄²u."""
    h2 = h * h
    # face 通量 f = ε(u_{i+1} − u_i) = −ε·∂u/∂x·h 的负数... 直接用对称形式:
    flux_x = eps_x * (u[1:, :, :] - u[:-1, :, :])
    flux_y = eps_y * (u[:, 1:, :] - u[:, :-1, :])
    flux_z = eps_z * (u[:, :, 1:] - u[:, :, :-1])
    # −∇·: 节点 i 取 −(f[i] − f[i−1])/h², 即 Σ_faces ε(u_i − u_j)/h²
    au = jnp.zeros_like(u)
    au = au.at[1:-1, :, :].add(-(flux_x[1:, :, :] - flux_x[:-1, :, :]) / h2)
    au = au.at[:, 1:-1, :].add(-(flux_y[:, 1:, :] - flux_y[:, :-1, :]) / h2)
    au = au.at[:, :, 1:-1].add(-(flux_z[:, :, 1:] - flux_z[:, :, :-1]) / h2)
    return au + kbar2 * u


def jacobi_diagonal(
    eps_x: jnp.ndarray,
    eps_y: jnp.ndarray,
    eps_z: jnp.ndarray,
    kbar2: jnp.ndarray,
    h: float,
) -> jnp.ndarray:
    h2 = h * h
    d = kbar2 + 0.0
    d = d.at[1:-1, :, :].add((eps_x[:-1, :, :] + eps_x[1:, :, :]) / h2)
    d = d.at[:, 1:-1, :].add((eps_y[:, :-1, :] + eps_y[:, 1:, :]) / h2)
    d = d.at[:, :, 1:-1].add((eps_z[:, :, :-1] + eps_z[:, :, 1:]) / h2)
    return d


def node_coords(grid: GridSpec) -> jnp.ndarray:
    """[nx,ny,nz,3] node positions (Å). 测试规模专用; 大网格按需切片."""
    n = grid.shape
    axes = [grid.origin[i] + jnp.arange(n[i], dtype=jnp.float32) * grid.h for i in range(3)]
    x, y, z = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([x, y, z], axis=-1)


def shell_indices(grid: GridSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(shell node coords [S,3], flat indices [S]) for the 6 Dirichlet faces."""
    import numpy as np

    n = grid.shape
    axes = [grid.origin[i] + np.arange(n[i]) * grid.h for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    mask = np.zeros(n, dtype=bool)
    mask[0, :, :] = mask[-1, :, :] = True
    mask[:, 0, :] = mask[:, -1, :] = True
    mask[:, :, 0] = mask[:, :, -1] = True
    pts = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    flat = np.flatnonzero(mask.reshape(-1))
    return jnp.asarray(pts), jnp.asarray(flat)


def interior_mask(grid: GridSpec) -> jnp.ndarray:
    import numpy as np

    m = np.ones(grid.shape, dtype=bool)
    m[0, :, :] = m[-1, :, :] = False
    m[:, 0, :] = m[:, -1, :] = False
    m[:, :, 0] = m[:, :, -1] = False
    return jnp.asarray(m)


def dh_boundary_values(
    shell_xyz: jnp.ndarray,  # [S,3] Å
    center: jnp.ndarray,  # [3] Å
    q_net: float,
    eps_out: float,
    kappa: float,
    a: float,  # 离子排除球半径: max 原子心距 + ion_radius
) -> jnp.ndarray:
    """单中心 Debye–Hückel (APBS bcfl sdh), 含离子尺寸修正 e^{κa}/(1+κa)."""
    r = jnp.linalg.norm(shell_xyz - center, axis=-1)
    if kappa > 0:
        return BJERRUM_VAC * q_net / eps_out * jnp.exp(-kappa * (r - a)) / (r * (1.0 + kappa * a))
    return BJERRUM_VAC * q_net / (eps_out * r)


def coulomb_boundary_values(
    shell_xyz: jnp.ndarray,  # [S,3]
    coords: jnp.ndarray,  # [N,3]
    q: jnp.ndarray,  # [N]
    eps_in: float,
    atom_block: int = 128,
) -> jnp.ndarray:
    """逐原子解析库仑和 (参考方程的 Dirichlet 边界, DESIGN.md §3.8).

    这是精确求和, 不是多极展开等近似 —— 只是换了归约顺序。

    **为什么分块**: S4 / h=0.5 下 S = 221,186 个边界点 × N = 1,835 个原子
    = 4.06e8 次距离计算。边界点方向天然并行, 但原本按原子写成 fori_loop 时
    **原子方向是 1835 轮串行依赖**, 每轮只做一次 [S] 的向量运算 —— GPU 上
    是 1835 次小 kernel 的启动延迟, 不是算力问题。

    分块后串行轮数降到 ceil(N/atom_block), 每轮并行做 [S, atom_block] 再归约。
    **不一次展开 [S, N]**: fp32 下那是 1.6 GB。`atom_block` 是显存与并行度的
    折中, 默认 64 (S=221k 时约 57 MB 中间量)。
    """
    s, n = shell_xyz.shape[0], coords.shape[0]
    nb = -(-n // atom_block)  # ceil
    pad = nb * atom_block - n
    # 补零的原子电荷为 0 -> 对和无贡献; 坐标随便放(用第一个原子)避免 1/0
    c_pad = jnp.concatenate([coords, jnp.repeat(coords[:1], pad, axis=0)], axis=0)
    q_pad = jnp.concatenate([q, jnp.zeros((pad,), q.dtype)], axis=0)

    def body(i, acc):
        cb = jax.lax.dynamic_slice(c_pad, (i * atom_block, 0), (atom_block, 3))
        qb = jax.lax.dynamic_slice(q_pad, (i * atom_block,), (atom_block,))
        d = jnp.sqrt(((shell_xyz[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
        return acc + (qb[None, :] / jnp.maximum(d, 1e-8)).sum(-1)

    # dtype 跟着 shell_xyz: 不带 dtype 的 jnp.zeros 在 x64 下是 float64,
    # 之后 scatter 回 fp32 的 u0 会触发不安全转换警告并把路径拖回 fp64。
    acc = jax.lax.fori_loop(0, nb, body, jnp.zeros(s, shell_xyz.dtype))
    return BJERRUM_VAC * acc / eps_in
