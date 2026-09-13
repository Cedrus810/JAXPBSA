"""Receptor–ligand Lennard-Jones cross energy, Lorentz–Berthelot combining rules (M2)."""
from __future__ import annotations

import jax.numpy as jnp

from .. import ACCUM_DTYPE


def lj_from_d2(d2, sigma_r, sigma_l, eps_r, eps_l):
    """[B,NR,NL] squared distances -> [B] kcal/mol. 归约同样升 fp64 (见 coulomb.py)."""
    sig = 0.5 * (sigma_r[None, :, None] + sigma_l[None, None, :])  # Lorentz
    eps = jnp.sqrt(eps_r[None, :, None] * eps_l[None, None, :])  # Berthelot
    inv6 = (sig * sig / d2) ** 3
    # 同 coulomb.py: sum(dtype=) 而非 .astype().sum()
    return jnp.sum(4.0 * eps * (inv6 * inv6 - inv6), axis=(1, 2), dtype=ACCUM_DTYPE)


def lj_cross(coords_r, coords_l, sigma_r, sigma_l, eps_r, eps_l):
    d2 = ((coords_r[:, :, None, :] - coords_l[:, None, :, :]) ** 2).sum(-1)
    return lj_from_d2(d2, sigma_r, sigma_l, eps_r, eps_l)
