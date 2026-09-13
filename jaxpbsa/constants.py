"""Physical constants and unit conventions (DESIGN.md §1).

Internal units: Å, e, kT/e (potential), kcal/mol (energy).
OpenMM units at the io boundary: nm, e, kJ/mol — converted once on extraction.
"""
from __future__ import annotations

NM_TO_ANGSTROM = 10.0
KJ_TO_KCAL = 1.0 / 4.184

# Coulomb constant in internal units
COULOMB_K = 332.063713  # kcal·Å·mol⁻¹·e⁻²

# kT in kcal/mol at 298.15 K; PB 源项/DH 边界的公共因子见 BJERRUM_VAC
KT_TO_KCAL = 0.5921868

# 真空 Bjerrum 长度 l_B(ε=1) = e²/(4πε₀k_BT) = COULOMB_K / KT_TO_KCAL ≈ 560.75 Å.
# 线性 PB 在 (Å, e, kT/e) 无量纲化下: -∇·(ε∇u) + ε·κ²·u = 4π·BJERRUM_VAC·ρ[e/Å³]
BJERRUM_VAC = COULOMB_K / KT_TO_KCAL

# 介质中的 Bjerrum 长度 (Å)
def bjerrum_length(eps_out: float, T: float = 298.15) -> float:
    return BJERRUM_VAC * (KT_TO_KCAL / _kt_kcal(T)) / eps_out


def _kt_kcal(T: float) -> float:
    return KT_TO_KCAL * T / 298.15


AVOGADRO_MOL_PER_L_TO_ANG3 = 6.02214076e-4  # 1 M = 6.0221e-4 粒子/Å³


def debye_kappa2(ionic_strength_M: float, eps_out: float, T: float = 298.15) -> float:
    """物理 κ² (Å⁻²),体相屏蔽长度 = κ⁻¹. 0.15 M / ε=78.5 → κ⁻¹ = 7.86 Å.

    注意: 这是物理量,用于 padding 与远场拟合;
    算子 stencil 里用的是 κ̄² = ε(r)·κ² (APBS 约定, 见 DESIGN.md §1).
    """
    l_b = BJERRUM_VAC * (KT_TO_KCAL / _kt_kcal(T)) / eps_out
    I_n = ionic_strength_M * AVOGADRO_MOL_PER_L_TO_ANG3  # 1:1 盐, I_n = ½Σnᵢzᵢ²
    return 8.0 * 3.141592653589793 * l_b * I_n
