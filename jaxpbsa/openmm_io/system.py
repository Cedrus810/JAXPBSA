"""Extract per-particle nonbonded parameters from an OpenMM System (M1).

Unit conversion happens here and only here: OpenMM (nm, kJ/mol) → internal (Å, kcal/mol).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm as mm

from ..constants import KJ_TO_KCAL, NM_TO_ANGSTROM


@dataclass
class MMParams:
    """Per-atom nonbonded parameters in internal units."""

    charge: np.ndarray  # [N] e
    sigma: np.ndarray  # [N] Å
    epsilon: np.ndarray  # [N] kcal/mol (LJ well depth)
    # exceptions[i] = (p1, p2, chargeProd[e²], sigma[Å], epsilon[kcal/mol]); v1 cross-energy 用不到,保存备用
    exceptions: list


def extract_nonbonded(system: mm.System) -> MMParams:
    if system.getNumParticles() == 0:
        raise ValueError("empty System")
    forces = system.getForces()
    nbs = [f for f in forces if isinstance(f, mm.NonbondedForce)]
    if len(nbs) != 1:
        raise ValueError("System must contain exactly one NonbondedForce")
    nb = nbs[0]
    unsupported = [type(f).__name__ for f in forces if isinstance(
        f, (mm.CustomNonbondedForce, mm.CustomGBForce, mm.GBSAOBCForce))
        or type(f).__name__.startswith("Amoeba")]
    if unsupported:
        raise ValueError(f"unsupported nonbonded force representation: {unsupported}")
    if nb.getNumParticleParameterOffsets() or nb.getNumExceptionParameterOffsets():
        raise ValueError("NonbondedForce parameter offsets are not supported")
    if nb.getNumParticles() != system.getNumParticles():
        raise ValueError("NonbondedForce particle count differs from System")
    if nb.getNonbondedMethod() not in (
        mm.NonbondedForce.NoCutoff,
        mm.NonbondedForce.CutoffNonPeriodic,
        mm.NonbondedForce.CutoffPeriodic,
        mm.NonbondedForce.Ewald,
        mm.NonbondedForce.PME,
    ):
        raise ValueError(f"unsupported nonbonded method: {nb.getNonbondedMethod()}")

    n = system.getNumParticles()
    charge = np.empty(n)
    sigma = np.empty(n)
    epsilon = np.empty(n)
    for i in range(n):
        q, s, e = nb.getParticleParameters(i)
        charge[i] = q.value_in_unit(mm.unit.elementary_charge)
        sigma[i] = s.value_in_unit(mm.unit.nanometer) * NM_TO_ANGSTROM
        epsilon[i] = e.value_in_unit(mm.unit.kilojoule_per_mole) * KJ_TO_KCAL
    if not all(np.isfinite(x).all() for x in (charge, sigma, epsilon)):
        raise ValueError("nonbonded parameters contain NaN/Inf")
    if (sigma < 0).any() or (epsilon < 0).any():
        raise ValueError("nonbonded sigma/epsilon must be nonnegative")

    exceptions = []
    for i in range(nb.getNumExceptions()):
        p1, p2, qprod, s, e = nb.getExceptionParameters(i)
        exceptions.append(
            (
                p1,
                p2,
                qprod.value_in_unit(mm.unit.elementary_charge**2),
                s.value_in_unit(mm.unit.nanometer) * NM_TO_ANGSTROM,
                e.value_in_unit(mm.unit.kilojoule_per_mole) * KJ_TO_KCAL,
            )
        )
    return MMParams(charge=charge, sigma=sigma, epsilon=epsilon, exceptions=exceptions)


def validate_cross_exceptions(params: MMParams, ligand_idx, receptor_idx) -> None:
    """The cross kernel supports independent, noncovalently bound species only."""
    lig, rec = set(map(int, ligand_idx)), set(map(int, receptor_idx))
    for a, b, *_ in params.exceptions:
        if (a in lig and b in rec) or (b in lig and a in rec):
            raise ValueError(f"unsupported receptor-ligand nonbonded exception: {a}, {b}")
