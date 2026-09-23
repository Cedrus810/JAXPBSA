"""PB grid: fixed regular lattice, shapes from the APBS-compatible dime set.

C/R/L share one grid (DESIGN.md §2) — grid is built once per run from the
whole batch of complex coordinates, so batch dims stay static for one compile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# APBS mg-manual 接受的 dime 值(2^a·c+1 家族), 对拍时免插值
APBS_DIME = (33, 49, 65, 97, 129, 161, 193, 225, 257, 321, 385)


@dataclass(frozen=True)
class GridSpec:
    origin: tuple[float, float, float]  # Å, node (0,0,0) 位置
    shape: tuple[int, int, int]  # (nx, ny, nz)
    h: float  # Å

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.shape))

    def half_extent(self) -> tuple[float, float, float]:
        return tuple(0.5 * (n - 1) * self.h for n in self.shape)

    @property
    def center(self) -> np.ndarray:
        """网格中心(Å, float64 host 数组)。"""
        return np.asarray(self.origin) + np.asarray(self.half_extent())


def recenter_com(coords: np.ndarray, masses: np.ndarray,
                 center: np.ndarray) -> np.ndarray:
    """质量加权质心对齐到 `center`, 返回平移后的坐标。`coords` 可以是 [N,3] 或 [T,N,3]。

    **为什么不是包围盒中点**: 中点由 6 个极端原子决定, 远端侧链摆 1 Å → 中点移
    0.5 Å ≈ 一个 h → 溶质相对格点的亚格点相位逐帧扫动。RESULTS §15.4 实测同一构象
    只挪相位的 ΔG_PB 峰峰值 **8.39** —— 包围盒归位(或完全不归位、任由刚体漂移)
    等于往 ΔG(t) 注入一层 ~8 的帧间噪声。质心: 单原子动 1 Å 只挪 mᵢ/M(~1/2000 Å),
    相位基本冻结, 摆放噪声从「帧间噪声」退化成「常数系统偏移」。

    **为什么质量加权**: 氢的热运动幅度最大而质量最小, 不加权的算术平均会把氢的
    抖动放大约 12 倍进相位。
    """
    c = np.asarray(coords, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    com = (c * m[:, None]).sum(axis=-2, keepdims=True) / m.sum()
    return c - com + np.asarray(center, dtype=np.float64)


def _next_dime(needed: int) -> int:
    for n in APBS_DIME:
        if n >= needed:
            return n
    raise ValueError(f"required grid dimension {needed} exceeds largest APBS dime {APBS_DIME[-1]}")


def make_grid(
    coords: np.ndarray,  # [B,N,3] 或 [N,3], Å —— 整个 batch 的 complex 坐标
    h: float,
    padding: float = 20.0,
    cubic: bool = False,
    center=None,
) -> GridSpec:
    """Batch-fixed center (DESIGN.md §2): center = bounding-box midpoint over all
    frames; 每轴半长 = 该轴包围盒半长 + padding。

    **默认按各轴独立取尺寸（矩形网格）。** 原先取最大径向距离再建立方体，
    对非球形溶质浪费很多节点：S4 在 h=0.5/padding=20 下立方体是 193³，
    而按轴包围盒是 161×161×193 —— **少 30.4% 的节点**。

    C/R/L 仍共用同一 origin/shape/h（差分抵消的前提，§2），只是形状不再是立方体。
    `cubic=True` 保留立方体行为供对照。

    `center` 给定时网格以它为中心, 每轴半长 = max|x − center| + padding ——
    配合 `recenter_com` 用(质心不在包围盒中点上, 按中点算半长会少给一侧)。

    注意这是**静态结构**上的节点节省，不是实测时间收益；轨迹要按整条轨迹的
    共同包围盒重新计算，且边界位置变了之后 padding 收敛性和 ΔG_PB 都要复核。
    """
    c = np.asarray(coords, dtype=np.float64)
    flat = c.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    if center is None:
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
    else:
        center = np.asarray(center, dtype=np.float64)
        half = np.abs(flat - center).max(axis=0)
    if cubic:
        need = np.full(3, np.linalg.norm(flat - center, axis=1).max() + padding)
    else:
        need = half + padding
    n = tuple(_next_dime(int(np.ceil(2 * need[i] / h)) + 1) for i in range(3))
    # float() 不是装饰: center[i] 是 np.float64, 而 numpy 标量在 JAX 里是*强*类型,
    # 会把 fp32 数组一路提升回 fp64 (Python float 是弱类型, 不会)。origin 会流进
    # node_coords/assign_density/interpolate, 漏掉这一步整条 fp32 路径都会悄悄变 fp64。
    origin = tuple(float(center[i] - 0.5 * (n[i] - 1) * h) for i in range(3))
    return GridSpec(origin=origin, shape=n, h=float(h))
