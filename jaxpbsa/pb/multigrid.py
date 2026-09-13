"""Geometric multigrid V-cycle for the variable-coefficient PB operator (§3.7).

Jacobi-PCG needs ~(L/h)^0.64 iterations (measured: 142/174/221 at h=1.0/0.75/0.5
on S4). Multigrid removes that dependence: each V-cycle costs ~2-3 fine-grid
operator applies, and the iteration count should be roughly h-independent.

Vertex-centred coarsening, n_c = (n_f-1)/2 + 1, which is exactly what the APBS
dime family (2^a·c+1) gives: 193 -> 97 -> 49 -> 25 -> 13 -> 7. Level shapes are
computed on the host, so every array shape is static and the whole V-cycle is
one jit/vmap-able function.

Everything below the top level solves the *correction* equation with homogeneous
Dirichlet data, so boundary nodes stay zero at every level; the caller's
Dirichlet values live only in `u0`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .operator import apply_operator, jacobi_diagonal

OMEGA = 0.8  # damped-Jacobi weight; 3-D Laplacian optimum is ~6/7


def _to_last(x, axis):
    return jnp.moveaxis(x, axis, -1)


def _prolong_axis(c: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Linear interpolation along one axis: length m -> 2m−1."""
    c = _to_last(c, axis)
    m = c.shape[-1]
    out = jnp.zeros(c.shape[:-1] + (2 * m - 1,), c.dtype)
    out = out.at[..., ::2].set(c).at[..., 1::2].set(0.5 * (c[..., :-1] + c[..., 1:]))
    return jnp.moveaxis(out, -1, axis)


def _restrict_axis(f: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Full weighting [1,2,1]/4 along one axis: length 2m−1 -> m."""
    f = _to_last(f, axis)
    m = (f.shape[-1] - 1) // 2 + 1
    pad = [(0, 0)] * (f.ndim - 1) + [(1, 1)]
    fp = jnp.pad(f, pad)          # zero outside == homogeneous Dirichlet
    out = (0.25 * fp[..., 0::2][..., :m]
           + 0.5 * fp[..., 1::2][..., :m]
           + 0.25 * fp[..., 2::2][..., :m])
    return jnp.moveaxis(out, -1, axis)


def restrict(r: jnp.ndarray) -> jnp.ndarray:
    for ax in (-3, -2, -1):
        r = _restrict_axis(r, ax)
    return r


def prolong(e: jnp.ndarray) -> jnp.ndarray:
    for ax in (-3, -2, -1):
        e = _prolong_axis(e, ax)
    return e


def _coarsen_faces(eps_f: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Coarse face coefficient from the two fine faces it spans.

    Along the flux direction the two fine faces are resistors *in series*, so the
    series (harmonic) combination is the right one -- arithmetic averaging here
    smears sharp ε jumps and is the usual reason re-discretised MG stalls on
    dielectric interfaces. Transverse directions are injected.
    """
    e = _to_last(eps_f, axis)
    a, b = e[..., 0::2], e[..., 1::2]
    k = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :k], b[..., :k]
    ser = 2.0 * a * b / (a + b)
    ser = jnp.moveaxis(ser, -1, axis)
    for tax in (-3, -2, -1):
        if tax != axis and tax - 3 != axis and tax + 3 != axis:
            ser = jnp.moveaxis(jnp.moveaxis(ser, tax, -1)[..., ::2], -1, tax)
    return ser


def build_levels(eps_x, eps_y, eps_z, kbar2, h, min_n: int = 7):
    """Host-side level hierarchy; all shapes static. Returns list of dicts."""
    levels = [dict(eps_x=eps_x, eps_y=eps_y, eps_z=eps_z, kbar2=kbar2, h=h,
                   diag=jacobi_diagonal(eps_x, eps_y, eps_z, kbar2, h))]
    # 矩形网格必须检查**所有三个轴**: 只看末轴会在某个轴已经到底时继续粗化,
    # 那个轴的 (n-1)/2+1 就不再是合法的顶点中心粗化。
    shp = kbar2.shape[-3:]
    while min(shp) > min_n and all((n - 1) % 2 == 0 for n in shp):
        f = levels[-1]
        ex = _coarsen_faces(f["eps_x"], -3)
        ey = _coarsen_faces(f["eps_y"], -2)
        ez = _coarsen_faces(f["eps_z"], -1)
        k2 = f["kbar2"][..., ::2, ::2, ::2]
        hc = f["h"] * 2.0
        levels.append(dict(eps_x=ex, eps_y=ey, eps_z=ez, kbar2=k2, h=hc,
                           diag=jacobi_diagonal(ex, ey, ez, k2, hc)))
        shp = k2.shape[-3:]
    for lv in levels:
        m = np.ones((lv["kbar2"].shape[-3:]), dtype=bool)
        m[0, :, :] = m[-1, :, :] = False
        m[:, 0, :] = m[:, -1, :] = False
        m[:, :, 0] = m[:, :, -1] = False
        lv["mask"] = jnp.asarray(m)
        # diag vanishes at the 8 corners (no face flux reaches them); those are
        # boundary nodes, but 0/0 = NaN would poison every reduction (see solver.py)
        lv["inv_diag"] = jnp.where(lv["mask"],
                                   1.0 / jnp.where(lv["diag"] > 0, lv["diag"], 1.0), 0.0)
    return levels


def _apply(lv, u):
    return apply_operator(u, lv["eps_x"], lv["eps_y"], lv["eps_z"], lv["kbar2"], lv["h"])


def _smooth(lv, u, b, sweeps):
    def body(_, u):
        return u + OMEGA * (b - _apply(lv, u)) * lv["inv_diag"]
    return jax.lax.fori_loop(0, sweeps, body, u)


def v_cycle(levels, u, b, level=0, nu1=2, nu2=2, coarse_sweeps=50):
    lv = levels[level]
    if level == len(levels) - 1:
        return _smooth(lv, u, b, coarse_sweeps)
    u = _smooth(lv, u, b, nu1)
    r = (b - _apply(lv, u)) * lv["mask"]
    ec = v_cycle(levels, jnp.zeros_like(restrict(r)), restrict(r), level + 1,
                 nu1, nu2, coarse_sweeps)
    u = u + prolong(ec) * lv["mask"]
    return _smooth(lv, u, b, nu2)


def mg_solve(levels, b, u0, tol=1e-5, max_cycles=30, nu1=2, nu2=2):
    """Returns (u, cycles, rel_residual). u0 carries the Dirichlet shell values."""
    lv0 = levels[0]
    mask = lv0["mask"]
    rhs = (b - _apply(lv0, u0)) * mask          # correction equation, homogeneous BC
    b_norm = jnp.sqrt((rhs * rhs).sum())

    def cond(st):
        _, k, rr = st
        return (k < max_cycles) & (rr > tol)

    def body(st):
        e, k, _ = st
        e = v_cycle(levels, e, rhs, nu1=nu1, nu2=nu2) * mask
        r = (rhs - _apply(lv0, e)) * mask
        return e, k + 1, jnp.sqrt((r * r).sum()) / b_norm

    e, cycles, rr = jax.lax.while_loop(
        cond, body, (jnp.zeros_like(u0), jnp.asarray(0, jnp.int32),
                     jnp.asarray(1.0, b_norm.dtype)))
    return u0 + e, cycles, rr


def make_preconditioner(levels, cycles: int = 1, nu1: int = 2, nu2: int = 2,
                        coarse_sweeps: int = 50):
    """One (or `cycles`) V-cycle(s) as a CG preconditioner: r -> z ≈ A⁻¹r.

    **Why a preconditioner and not a solver.** Re-discretised MG diverges on this
    operator at production settings. Measured convergence factor per V-cycle
    (33³, random RHS, ε jump across a sphere):

        jump    1:2     1:10    1:80
        large   0.21    0.27    **1.8 (diverges)**
        small   0.21    0.21    **stalls at 3e-2, then grows**

    ε_in=1 / ε_out=78.5 is exactly the 1:80 column. The cause is the standard
    one: geometric (operator-independent) interpolation cannot represent a
    solution whose normal derivative jumps by 80x across the dielectric
    interface, so the coarse-grid correction is wrong where it matters most.
    CG cannot diverge -- it minimises -- so wrapping the V-cycle in CG turns a
    broken solver into a good preconditioner and keeps the h-independence that
    multigrid buys.

    The proper fix, if MG-as-solver is ever wanted, is operator-dependent
    interpolation (Alcouffe-Brandt / de Zeeuw "black box" MG) or a Galerkin
    coarse operator; both are much more code than this.

    Symmetry (required by CG) holds: restriction is P^T/2³ (the variational
    choice), the Jacobi smoother is symmetric, and nu1 == nu2.
    """
    if nu1 != nu2:
        raise ValueError("nu1 must equal nu2 or the preconditioner is not symmetric")

    def precond(r):
        z = jnp.zeros_like(r)
        for _ in range(cycles):
            z = v_cycle(levels, z, r, nu1=nu1, nu2=nu2,
                        coarse_sweeps=coarse_sweeps) * levels[0]["mask"]
        return z

    return precond
