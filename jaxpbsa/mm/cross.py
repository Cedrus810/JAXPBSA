"""Batched MM cross terms on a full-complex trajectory (M2)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import dtype as _dtype
from ..openmm_io.system import MMParams
from .coulomb import coulomb_from_d2
from .lj import lj_from_d2


@jax.jit
def _cross_arrays(coords, q, sigma, eps, ligand_idx, receptor_idx):
    """一个 JIT 入口, **距离只算一次**给 Coulomb 和 LJ 共用。

    之前 coulomb_cross 和 lj_cross 各自算一遍 [B,NR,NL] 的距离 —— NR×NL ~ 10^6,
    这是白算一倍。
    """
    x_l = coords[:, ligand_idx, :]
    x_r = coords[:, receptor_idx, :]
    d2 = ((x_r[:, :, None, :] - x_l[:, None, :, :]) ** 2).sum(-1)
    return {
        "e_coul_rl": coulomb_from_d2(d2, q[receptor_idx], q[ligand_idx]),
        "e_lj_rl": lj_from_d2(d2, sigma[receptor_idx], sigma[ligand_idx],
                              eps[receptor_idx], eps[ligand_idx]),
    }


def prepare_cross(params: MMParams, ligand_idx, receptor_idx):
    """把静态参数一次性转成设备数组, 供逐帧复用。

    `extract_nonbonded` 返回的是 float64 numpy。fp32 模式下每帧重新
    `jnp.asarray(..., float32)` 等于每帧做一次 host→device 转换 + 拷贝 ——
    实测这让 fp32 的 MM 比 fp64 还慢(3.06 ms vs 2.31 ms), 因为 fp64 路径
    不需要转换。轨迹分析里 topology 是固定的, 这些应该只做一次。
    """
    dt = _dtype()
    return (jnp.asarray(params.charge, dt), jnp.asarray(params.sigma, dt),
            jnp.asarray(params.epsilon, dt), jnp.asarray(ligand_idx),
            jnp.asarray(receptor_idx))


def mm_cross_prepared(coords, prepared) -> dict[str, jnp.ndarray]:
    """逐帧入口: 参数已在设备上。轨迹循环用这个, 不要用 mm_cross。"""
    return _cross_arrays(jnp.asarray(coords, _dtype()), *prepared)


def mm_cross(
    coords: jnp.ndarray,  # [B, N, 3] Å (complex, 与 System 原子序一致)
    params: MMParams,
    ligand_idx: jnp.ndarray,  # [NL]
    receptor_idx: jnp.ndarray,  # [NR]
) -> dict[str, jnp.ndarray]:
    """Receptor–ligand cross interaction energies per frame, kcal/mol.

    **dtype 在这里统一**。`extract_nonbonded` 用 `np.empty(n)` 返回 float64, 而
    x64 是开着的(归约需要), 所以不显式转换的话逐对乘积会全程走 float64 —— 坐标
    是不是 fp32 根本不影响, 精度策略等于没落实。
    """
    return mm_cross_prepared(coords, prepare_cross(params, ligand_idx, receptor_idx))
