"""Trilinear point-charge assignment (APBS chgm spl0) and the adjoint interpolation.

Assignment and interpolation share the same weights, so the discretised
reaction energy ½Σq·u(rᵢ) is self-consistent with the discretised source term.
"""
from __future__ import annotations

import jax.numpy as jnp

from ..constants import BJERRUM_VAC
from .grid import GridSpec


def _weights(coords: jnp.ndarray, grid: GridSpec):
    """g = fractional node coordinate; i0 = floor(g); f = g - i0 ∈ [0,1)."""
    # dtype 跟着 coords 走。凡是"从零创建"的浮点数组都要显式带 dtype:
    # x64 开着时不带 dtype 默认就是 float64, 会把 fp32 路径静默提升回 fp64。
    origin = jnp.asarray(grid.origin, coords.dtype)
    g = (coords - origin) / grid.h
    i0 = jnp.floor(g).astype(jnp.int32)
    f = g - i0.astype(g.dtype)
    n = jnp.asarray(grid.shape)
    # box padding 保证原子在内部; jit 下无法布尔花式索引, 统一 clip + 权重置零
    ok = jnp.all((i0 >= 0) & (i0 + 1 < n), axis=1)
    i0 = jnp.clip(i0, 0, n - 2)
    return i0, f, ok


def assign_density(
    coords: jnp.ndarray,  # [N,3] Å
    q: jnp.ndarray,  # [N] e
    grid: GridSpec,
) -> jnp.ndarray:  # [nx,ny,nz], e/Å³
    i0, f, ok = _weights(coords, grid)
    rho = jnp.zeros(grid.shape, coords.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + jnp.array([dx, dy, dz])
                w = ((1 - f[:, 0]) if dx == 0 else f[:, 0]) * (
                    (1 - f[:, 1]) if dy == 0 else f[:, 1]
                ) * ((1 - f[:, 2]) if dz == 0 else f[:, 2]) * ok
                rho = rho.at[tuple(idx.T)].add(q * w / grid.h**3)
    return rho


def source_term(rho: jnp.ndarray, grid: GridSpec) -> jnp.ndarray:
    """b = 4π·l_B,vac·ρ —— 在 (Å, e, kT/e) 无量纲化下的源项 (DESIGN.md §3.4)."""
    return 4.0 * jnp.pi * BJERRUM_VAC * rho


def interpolate(
    phi: jnp.ndarray,  # [nx,ny,nz], kT/e
    coords: jnp.ndarray,  # [N,3] Å
    grid: GridSpec,
) -> jnp.ndarray:  # [N], kT/e
    i0, f, ok = _weights(coords, grid)
    out = jnp.zeros(coords.shape[0], coords.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + jnp.array([dx, dy, dz])
                w = ((1 - f[:, 0]) if dx == 0 else f[:, 0]) * (
                    (1 - f[:, 1]) if dy == 0 else f[:, 1]
                ) * ((1 - f[:, 2]) if dz == 0 else f[:, 2])
                out = out + w * phi[tuple(idx.T)] * ok
    return out
