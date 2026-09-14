"""非极性（SA）项。

    G_SA = γ·SASA + β        ΔG_SA = γ·(A_C − A_R − A_L) + β·(1 − 1 − 1)
                                   = γ·ΔSASA − β

**注意 β 不抵消**：三个 species 各有一份常数项，差分后剩 −β。Amber `pbsa` 的
INP=1 默认 β=0 所以看不出来，但换成 INP=2 (γ=0.0378, β=−0.5692) 就会差 0.57。

**后端是可换的**（DESIGN.md §3.10）：

    stage 1  zsasa —— 外部 CLI, host 侧, **不在 JIT 图里**。SA 不回馈 PB,
             所以它不需要进图。已验证与解析解一致。
    stage 2  JAX Shrake–Rupley —— 进图, 可 vmap/scan, 与 PB 共用 cell-list。
             届时只换 `backend`, 调用点不变。

接口收**数组**不收文件, 就是为了这个替换是无痛的。
"""
from __future__ import annotations

import numpy as np

#: Amber `pbsa` INP=1 的默认值。INP=2 是 (0.0378, −0.5692)，常见 MM-PBSA 还有
#: (0.00542, 0.92)。**论文里必须写明用的是哪一档** —— γ 直接平移 ΔG_SA。
GAMMA_INP1, BETA_INP1 = 0.005, 0.0


def sasa(coords, radii, backend="zsasa", **kw):
    """[B,N,3] 或 [N,3] (Å) + radii [N] (Å) -> SASA (Ų)。"""
    if backend == "zsasa":
        from .zsasa import sasa as _f
        return _f(coords, radii, **kw)
    if backend == "jax":
        raise NotImplementedError("stage 2: JAX Shrake–Rupley 待实现")
    raise ValueError(f'未知后端 {backend!r}; 可选 "zsasa" | "jax"')


def g_sa(area, gamma=GAMMA_INP1, beta=BETA_INP1):
    """G_SA = γ·SASA + β  (kcal/mol)。"""
    return gamma * np.asarray(area) + beta


def delta_g_sa(coords, radii, receptor_idx, ligand_idx,
               gamma=GAMMA_INP1, beta=BETA_INP1, backend="zsasa", **kw):
    """ΔG_SA = G_SA(C) − G_SA(R) − G_SA(L)，返回 (ΔG_SA, dict(各 species 面积))。

    三个 species 各算一次 —— **A_R 是受体在孤立状态下的 SASA**, 不是受体原子在
    复合物里的那部分面积, 两者是不同的量, 所以不能从一次计算里拆出来。

    这里直接切片即可（PB 那边要把 R/L 补齐到 complex 的原子数是为了共用编译，
    host 侧没有这个约束）。
    """
    c = np.asarray(coords, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    rec, lig = np.asarray(receptor_idx), np.asarray(ligand_idx)
    sl = (slice(None), ...) if c.ndim == 3 else (...,)
    a_c = sasa(c, r, backend=backend, **kw)
    a_r = sasa(c[..., rec, :], r[rec], backend=backend, **kw)
    a_l = sasa(c[..., lig, :], r[lig], backend=backend, **kw)
    d = gamma * (np.asarray(a_c) - np.asarray(a_r) - np.asarray(a_l)) - beta
    return d, {"complex": a_c, "receptor": a_r, "ligand": a_l,
               "delta_sasa": np.asarray(a_c) - np.asarray(a_r) - np.asarray(a_l)}


__all__ = ["sasa", "g_sa", "delta_g_sa", "GAMMA_INP1", "BETA_INP1"]
