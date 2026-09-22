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


def enable_compilation_cache(path: str | None = None) -> str:
    """打开 XLA 持久编译缓存, 返回缓存目录。

    **实测(S4, h=0.5, MG)**: 首次编译 36.5 s -> 命中 5.9 s(6.2x),
    总冷启动 42.1 -> 10.7 s。缓存 3.5 MB。稳态速度不变 —— 这只省编译。

    **不在 import 时自动打开**: 修改 `jax.config` 是进程全局的, 库不该替调用方
    做这个决定。脚本入口显式调用, 或自己设 `jax_compilation_cache_dir`。

    剩下的 10.7 s 里约 3.7 s 是 Python import —— 那部分只有 AOT 导出(jax.export ->
    StableHLO -> PJRT C++)能拿走。**不划算**: 我们的 grid shape 随体系和 h 变
    (`make_grid` 从 APBS dime 集合里选), 而 `build_levels` 是 host 侧按具体 shape
    的 Python 循环, 没法符号化 —— 等于每个形状导一份产物。JaxForce 那套成立是因为
    它形状固定、调用上百万次; 我们两个前提都反过来。
    """
    import os
    import jax
    path = path or os.path.expanduser("~/.cache/jaxpbsa-xla")
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    return path
