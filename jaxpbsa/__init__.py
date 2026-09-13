"""JAXPBSA: JAX-accelerated MM/PBSA analysis for OpenMM trajectories.

Units (see DESIGN.md §1): length Å, charge e, potential kT/e, energy kcal/mol.

Precision policy (measured, see DESIGN.md §1 and §3.7):

  * **storage and the PB solve run in fp32.** MD coordinates are float32 anyway
    (DCD stores float32), and on S4 at h=0.5 fp32 shifts G_PB by a constant
    0.035 kcal/mol out of -954 (3.7e-5 relative) while the *discretisation*
    error between h=0.75 and h=0.5 is 40 kcal/mol -- three orders of magnitude
    larger. fp32 buys 2.7x at matched tolerance. The two-solve cancellation
    (u_solvent - u_ref, both ~10^3 kT/e, difference ~10^1) survives fp32; that
    was checked, not assumed.

  * **reductions stay in fp64.** The R-L cross energy sums ~10^6 signed pairs
    and the final G_PB is a dot product over all atoms; fp32 accumulation there
    loses ~11 bits to cancellation. This is why x64 stays *enabled* -- it is the
    capability that makes float64 reachable. The array dtype is the policy, and
    that is what `set_precision` changes.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # capability: float64 must exist for reductions

__version__ = "0.1.0"

#: dtype for stored grids/coordinates and the PB solve. Reductions ignore this.
DTYPE = jnp.float32
ACCUM_DTYPE = jnp.float64


def set_precision(bits: int = 32) -> None:
    """Select the *array* dtype (32 or 64). Reduction dtype is unaffected."""
    global DTYPE
    if bits not in (32, 64):
        raise ValueError(f"bits must be 32 or 64, got {bits}")
    DTYPE = jnp.float32 if bits == 32 else jnp.float64


def dtype() -> "jnp.dtype":
    return DTYPE
