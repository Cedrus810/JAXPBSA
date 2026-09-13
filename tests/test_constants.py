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
