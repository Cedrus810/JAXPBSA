"""T3: MM cross energies vs OpenMM CustomNonbondedForce (reference platform, fp64).

Covers parameter extraction + unit conversion + Lorentz–Berthelot rules + kernels.
"""
from __future__ import annotations

import numpy as np

import jaxpbsa
import jax.numpy as _jnp

_FP64 = _jnp.zeros(1, jaxpbsa.dtype()).dtype == _jnp.float64
import openmm as mm
import openmm.unit as unit
import jax.numpy as jnp

from jaxpbsa.constants import KJ_TO_KCAL
from jaxpbsa.mm import mm_cross
from jaxpbsa.openmm_io import chain_indices, extract_nonbonded, species_indices

ONE_4PI_EPS0_NM = 138.935456  # kJ·nm/mol/e²


def _reference_cross(system, rec, lig, coords_nm, expression):
    cnf = mm.CustomNonbondedForce(expression)
    cnf.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0_NM)
    for p in ("q", "s", "e"):
        cnf.addPerParticleParameter(p)
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    ref_system = mm.System()
    for i in range(system.getNumParticles()):
        ref_system.addParticle(1.0)
        q, s, e = nb.getParticleParameters(i)  # raw OpenMM values, independent path
        cnf.addParticle([q / unit.elementary_charge, s / unit.nanometer, e / unit.kilojoule_per_mole])
    cnf.addInteractionGroup([int(i) for i in rec], [int(i) for i in lig])
    cnf.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    ref_system.addForce(cnf)
    platform = mm.Platform.getPlatformByName("Reference")  # double precision
    ctx = mm.Context(ref_system, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(unit.Quantity(coords_nm, unit.nanometer))
    kj = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx
    return kj * KJ_TO_KCAL


def test_mm_cross_matches_openmm(peptide_system):
    system, topology, positions = peptide_system
    params = extract_nonbonded(system)
    lig, rec = species_indices(topology, chain_indices(topology, "B"))

    rng = np.random.default_rng(0)
    B = 3
    coords = np.stack([positions + rng.normal(0.0, 0.15, positions.shape) for _ in range(B)])

    ours = mm_cross(jnp.asarray(coords), params, jnp.asarray(lig), jnp.asarray(rec))
    coul = np.asarray(ours["e_coul_rl"])
    lj = np.asarray(ours["e_lj_rl"])

    expr_coul = "ONE_4PI_EPS0*q1*q2/r"
    expr_lj = "4*sqrt(e1*e2)*((s12/r)^12-(s12/r)^6); s12=0.5*(s1+s2)"
    for b in range(B):
        ref_c = _reference_cross(system, rec, lig, coords[b] * 0.1, expr_coul)
        ref_l = _reference_cross(system, rec, lig, coords[b] * 0.1, expr_lj)
        # 容差跟着数组 dtype 走。fp32 逐对计算在**这个小体系**上相对偏差 ~2e-5:
        # 对数少(NR×NL ~ 1e3), 抵消占比高。真实体系 S4(297,900 对)上反而好得多 ——
        # E_coul 4e-7、E_LJ 2e-5(见 RESULTS.md)。所以不能拿小体系的容差外推生产精度,
        # 反过来也不行。
        rtol = 1e-8 if _FP64 else 5e-5
        assert np.isclose(coul[b], ref_c, rtol=rtol), (b, coul[b], ref_c)
        assert np.isclose(lj[b], ref_l, rtol=rtol), (b, lj[b], ref_l)


def test_species_indices_validation(peptide_system):
    _, topology, _ = peptide_system
    n = topology.getNumAtoms()
    lig0 = chain_indices(topology, "A")
    lig, rec = species_indices(topology, lig0)
    assert lig.size + rec.size == n
    assert not np.intersect1d(lig, rec).size
    assert lig.tolist() == sorted(set(lig.tolist()))
    import pytest

    with pytest.raises(ValueError):
        species_indices(topology, [n])  # out of range
    with pytest.raises(ValueError):
        species_indices(topology, list(range(n)))  # empty receptor
