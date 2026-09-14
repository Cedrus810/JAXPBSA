"""非极性（SA）项。

    G_SA = γ·SASA + β        ΔG_SA = γ·(A_C − A_R − A_L) + β·(1 − 1 − 1)
                                   = γ·ΔSASA − β

**注意 β 不抵消**：三个 species 各有一份常数项，差分后剩 −β。Amber `pbsa` 的
INP=1 默认 β=0 所以看不出来，但换成 INP=2 (γ=0.0378, β=−0.5692) 就会差 0.57。

实现是 `jax_sr.py` 的 JAX Shrake–Rupley —— 进图, 可 jit/vmap。

**与 PB 之间只共享物理定义**: rᵢ 来自 `openmm_io.assign_radii`, r_p 来自
`constants.PROBE_RADIUS`。PB 的 voxel mask / atom→grid 映射 / erosion 表示 /
网格间距 h 一律不外泄到这里 —— 那些是离散化细节, 不是物理 (DESIGN.md §3.10)。

接口收**数组**不收文件, 所以对拍参照物 (`tests/zsasa_ref.py` 的 zsasa) 能喂同一套
半径。参照物不是后端, 不进包。
"""
from __future__ import annotations

import numpy as np

#: Amber `pbsa` INP=1 的默认值。INP=2 是 (0.0378, −0.5692)，常见 MM-PBSA 还有
#: (0.00542, 0.92)。**论文里必须写明用的是哪一档** —— γ 直接平移 ΔG_SA。
GAMMA_INP1, BETA_INP1 = 0.005, 0.0


def sasa(coords, radii, **kw):
    """[B,N,3] 或 [N,3] (Å) + radii [N] (Å) -> SASA (Ų)。"""
    from .jax_sr import sasa as _f
    return _f(coords, radii, **kw)


def g_sa(area, gamma=GAMMA_INP1, beta=BETA_INP1):
    """G_SA = γ·SASA + β  (kcal/mol)。"""
    return gamma * np.asarray(area) + beta


def delta_g_sa(coords, radii, receptor_idx, ligand_idx,
               gamma=GAMMA_INP1, beta=BETA_INP1, **kw):
    """ΔG_SA = G_SA(C) − G_SA(R) − G_SA(L)，返回 (ΔG_SA, dict(各 species 面积))。

    三个 species 各算一次 —— **A_R 是受体在孤立状态下的 SASA**, 不是受体原子在
    复合物里的那部分面积, 两者是不同的量, 所以不能从一次计算里拆出来。

    这里直接切片, **不像 PB 那样把 R/L 补齐到 complex 的原子数**。代价是三种形状
    = 三次编译（PB 补齐正是为了共用一次编译）；但 SA 的编译只有秒级、且逐轨迹摊销一次，
    而补齐要为每个 species 都跑满 N 个原子的邻居搜索 —— 对 ligand(180 原子 vs 1835)
    就是十倍白算。**这里形状不统一是划算的，PB 那边不划算，两者的取舍不同。**
    """
    c = np.asarray(coords, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    rec, lig = np.asarray(receptor_idx), np.asarray(ligand_idx)
    a_c = sasa(c, r, **kw)
    a_r = sasa(c[..., rec, :], r[rec], **kw)
    a_l = sasa(c[..., lig, :], r[lig], **kw)
    d = gamma * (np.asarray(a_c) - np.asarray(a_r) - np.asarray(a_l)) - beta
    return d, {"complex": a_c, "receptor": a_r, "ligand": a_l,
               "delta_sasa": np.asarray(a_c) - np.asarray(a_r) - np.asarray(a_l)}


__all__ = ["sasa", "g_sa", "delta_g_sa", "GAMMA_INP1", "BETA_INP1"]
