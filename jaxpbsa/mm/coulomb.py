"""Receptor–ligand Coulomb cross energy, batched over frames (M2, DESIGN.md §3.2)."""
from __future__ import annotations

import jax.numpy as jnp

from .. import ACCUM_DTYPE
from ..constants import COULOMB_K


def coulomb_from_d2(d2, q_r, q_l):
    """[B,NR,NL] squared distances -> [B] kcal/mol.

    逐对项在输入 dtype 下算(便宜), **归约升 fp64**: NR×NL ~ 10^6 个带符号项且正负
    强烈抵消, fp32 累加会丢 ~11 bit。
    """
    qq = q_r[None, :, None] * q_l[None, None, :]
    # sum(dtype=) 在归约内部升精度; .astype().sum() 会先物化一整个 float64 副本
    # ([B,NR,NL] 百万元素), 凭空多一趟访存 —— 实测让 fp32 比 fp64 还慢。
    return COULOMB_K * jnp.sum(qq / jnp.sqrt(d2), axis=(1, 2), dtype=ACCUM_DTYPE)


def coulomb_cross(coords_r, coords_l, q_r, q_l):
    """[B,NR,3] × [B,NL,3] -> [B] kcal/mol. 单独调用时自己算距离。"""
    d2 = ((coords_r[:, :, None, :] - coords_l[:, None, :, :]) ** 2).sum(-1)
    return coulomb_from_d2(d2, q_r, q_l)
