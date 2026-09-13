"""Direct (non-iterative) solve of the reference equation via DST-I (DESIGN.md §3.8).

The reference problem is uniform ε_in with κ̄² = 0, i.e. a *constant-coefficient*
Poisson problem with Dirichlet data:

    (ε_in/h²)·(6u_i − Σ_neighbours u_j) = f_i   on the interior,  u = g on the shell

Its eigenvectors are exactly the sine basis, so one forward DST, a pointwise
divide and one inverse DST give the machine-precision solution with **zero
iterations**. Measured on S4 (1835 atoms, padding 20 Å) the reference solve is
the dominant cost and gets worse with refinement -- it is the harder of the two
equations precisely because it has no dielectric contrast and no κ̄² mass term:

    h      it_solvent   it_reference   ref/solv
    1.00      142           271          1.91
    0.75      174           348          2.00
    0.50      221           494          2.24     -> 69% of all iterations

JAX has no DST, so DST-I is built from an odd extension plus one complex FFT:
for x of length N, y = [0, x, 0, −reverse(x)] has length 2(N+1) and
FFT(y)_k = −2i·Σ_j x_j sin(πjk/(N+1)) for k = 1..N.
DST-I is its own inverse up to the factor 2/(N+1) per axis.
"""
from __future__ import annotations

import jax.numpy as jnp


def dst1(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Unnormalised DST-I along `axis`: X_k = Σ_j x_j sin(πjk/(N+1)), k = 1..N."""
    # work on the trailing axis so the slice below is written with Ellipsis,
    # which stays correct under vmap (axis numbers would not).
    x = jnp.moveaxis(x, axis, -1)
    n = x.shape[-1]
    pad = jnp.zeros(x.shape[:-1] + (1,), x.dtype)
    y = jnp.concatenate([pad, x, pad, -jnp.flip(x, axis=-1)], axis=-1)
    # y 是实数 -> rfft 只算需要的半频谱。取的索引 1..n 全落在 rfft 的输出范围内
    # (长度 2(n+1) 的实输入 -> rfft 给 n+2 个系数), 所以索引与归一化都不变。
    out = -jnp.imag(jnp.fft.rfft(y, axis=-1))[..., 1 : n + 1] / 2
    return jnp.moveaxis(out, -1, axis)


def _eigenvalues(n: int, dtype) -> jnp.ndarray:
    """Eigenvalues of the 1-D second difference (2u_i − u_{i−1} − u_{i+1}), N = n."""
    k = jnp.arange(1, n + 1, dtype=dtype)
    return 4.0 * jnp.sin(jnp.pi * k / (2.0 * (n + 1))) ** 2


def poisson_solve(f: jnp.ndarray, eps: float, h: float) -> jnp.ndarray:
    """Solve (eps/h²)·L u = f on an interior block with homogeneous Dirichlet data.

    `f` has the shape of the *interior* nodes; the returned u has the same shape.
    Exact to round-off -- no tolerance, no iteration count.
    """
    nx, ny, nz = f.shape[-3:]
    dt = f.dtype
    fh = dst1(dst1(dst1(f, -3), -2), -1)
    lam = (_eigenvalues(nx, dt)[:, None, None]
           + _eigenvalues(ny, dt)[None, :, None]
           + _eigenvalues(nz, dt)[None, None, :])
    uh = fh / (lam * (eps / (h * h)))
    u = dst1(dst1(dst1(uh, -3), -2), -1)
    # DST-I applied twice is (2/(N+1))^-1 · identity per axis
    return u * (8.0 / ((nx + 1) * (ny + 1) * (nz + 1)))


def solve_reference(
    b: jnp.ndarray,      # source term on the full grid
    u0: jnp.ndarray,     # zeros inside, Dirichlet shell values on the boundary
    apply_a,             # the reference operator (uniform eps, kbar2 = 0)
    eps_in: float,
    h: float,
) -> jnp.ndarray:
    """Full-grid reference potential: Dirichlet data folded into the RHS, then DST."""
    rhs = (b - apply_a(u0))[..., 1:-1, 1:-1, 1:-1]
    return u0.at[..., 1:-1, 1:-1, 1:-1].set(poisson_solve(rhs, eps_in, h))
