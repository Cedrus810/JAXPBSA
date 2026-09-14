"""Numerical anchors from DESIGN.md §1."""
from __future__ import annotations

import numpy as np

from jaxpbsa.constants import BJERRUM_VAC, COULOMB_K, debye_kappa2


def test_bjerrum_vacuum():
    # e²/(4πε₀k_BT) at 298.15 K ≈ 560.7 Å
    assert abs(BJERRUM_VAC - 560.75) < 0.5


def test_debye_length_015M():
    # κ⁻¹ ≈ 7.86 Å @ I=0.15 M, ε=78.5, 298.15 K
    kappa2 = debye_kappa2(0.15, 78.5)
    assert abs(1.0 / np.sqrt(kappa2) - 7.86) < 0.02


def test_no_salt():
    assert debye_kappa2(0.0, 78.5) == 0.0


def test_coulomb_constant():
    # 1 e² at 1 Å in kcal/mol
    assert abs(COULOMB_K - 332.06371) < 1e-4


def test_pb_and_sa_share_one_probe_radius():
    """探针半径**只能有一处字面量**。

    这是 SA/PB 之间唯二该共享的物理定义之一（另一个是 rᵢ，单源
    `openmm_io.assign_radii`）。两边用不同的 r_p，ΔG_PB 的分子表面和 ΔG_SA 的
    可及表面就对应不同的溶剂，相加没有物理意义 —— 而且**不会报错**，只会悄悄
    给出一个偏掉的 ΔG_MM/PBSA。所以这里既比默认值，也扫源码里的裸字面量。
    """
    import inspect
    import pathlib

    from jaxpbsa.constants import PROBE_RADIUS
    from jaxpbsa.pb.energy import PBParams
    from jaxpbsa.sa import jax_sr

    assert PBParams().probe_radius == PROBE_RADIUS
    for f in (jax_sr.sasa, jax_sr.sasa_core):
        got = inspect.signature(f).parameters["probe_radius"].default
        assert got is PROBE_RADIUS, f"{f.__qualname__} 自带了一份探针半径: {got}"

    root = pathlib.Path(jax_sr.__file__).parents[1]
    src = [p for p in root.rglob("*.py")
           if p.name != "constants.py" and "test" not in p.parts]
    bad = [f"{p.relative_to(root)}:{i}" for p in src
           for i, ln in enumerate(p.read_text().splitlines(), 1)
           if "probe_radius=1.4" in ln.replace(" ", "")]
    assert not bad, f"探针半径的裸字面量应改成 constants.PROBE_RADIUS: {bad}"
