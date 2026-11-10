"""Failure/admission contracts using static systems and synthetic arrays; no MD/PB solves."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import openmm as mm
from openmm import app
import pytest

from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
from jaxpbsa.openmm_io.system import extract_nonbonded, validate_cross_exceptions
from jaxpbsa.pb.energy import _scan_warm_frames


def frame():
    return dict(e_coul_rl=1., e_lj_rl=2., g_pb_complex=3., g_pb_receptor=1.,
                g_pb_ligand=1., delta_g_pb=1., delta_g_sa=-1., delta_g_mmpbsa=3.,
                margin_A=2., margin_lig_A=1., converged=True, sa_ok=True, solver_iters=[1, 1, 1])


@pytest.mark.parametrize("key", ["e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor",
                                   "g_pb_ligand", "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa", "margin_A"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_result_is_flagged(key, value):
    r = frame(); r[key] = value
    assert key in PBSAReporter._violation(SimpleNamespace(margin_min=.8), r)


def test_shared_grid_nan_margin_remains_allowed():
    r = frame(); r["margin_lig_A"] = float("nan")
    assert PBSAReporter._violation(SimpleNamespace(margin_min=.8), r) is None
    r["margin_lig_A"] = float("inf")
    assert "margin_lig_A" in PBSAReporter._violation(SimpleNamespace(margin_min=.8), r)


@pytest.mark.parametrize("threshold", [-1., float("nan"), float("inf")])
def test_reporter_rejects_invalid_margin_threshold(threshold):
    with pytest.raises(ValueError, match="margin_min"):
        PBSAReporter(SimpleNamespace(margin_flag=threshold), 1, [0])


def static_system():
    s, nb, top = mm.System(), mm.NonbondedForce(), app.Topology()
    for q in [1., -1.]:
        s.addParticle(16.)
        nb.addParticle(q, .3, .1)
        top.addAtom("O", app.element.oxygen, top.addResidue("LIG", top.addChain()))
    s.addForce(nb)
    return s, nb, top


@pytest.mark.parametrize("kind", ["custom", "multiple", "offset", "exception_offset"])
def test_unsupported_systems_are_rejected_before_scoring(kind):
    s, nb, _ = static_system()
    if kind == "custom":
        f = mm.CustomNonbondedForce("0"); f.addParticle([]); f.addParticle([]); s.addForce(f)
    elif kind == "multiple":
        s.addForce(mm.NonbondedForce())
    else:
        nb.addGlobalParameter("lambda_q", 1.)
        if kind == "offset":
            nb.addParticleParameterOffset("lambda_q", 0, 1., 0., 0.)
        else:
            nb.addException(0, 1, 0., .3, 0.)
            nb.addExceptionParameterOffset("lambda_q", 0, 1., 0., 0.)
    with pytest.raises(ValueError):
        extract_nonbonded(s)


def test_supported_extraction_and_cross_exception_boundary():
    s, nb, _ = static_system()
    p = extract_nonbonded(s)
    np.testing.assert_allclose(p.charge, [1., -1.])
    validate_cross_exceptions(p, [1], [0])
    nb.addException(0, 1, 0., .3, 0.)
    with pytest.raises(ValueError, match="receptor-ligand"):
        validate_cross_exceptions(extract_nonbonded(s), [1], [0])


@pytest.mark.parametrize("solute,lig", [([0, 0], [1]), ([0, 1], [1, 1]), ([0., 1.], [1]),
                                        ([0, 1], [.5]), ([-1, 1], [1])])
def test_bad_atom_indices_fail_before_analyzer_construction(solute, lig):
    s, _, top = static_system()
    with pytest.raises(ValueError, match="indices"):
        OnlineMMPBSA(s, top, solute, lig, np.zeros((2, 3)))


def test_failed_warm_state_is_cleared_and_next_frame_recovers():
    def synthetic_frame(x, q, radii, previous):
        # Two kinds of failed output: a finite unconverged iterate, then NaN.
        u = jnp.where(x == 0, 999., jnp.where(x == 1, jnp.nan, previous + 1.))
        return dict(converged=x != 0, input_state=previous), u, None

    final, out = _scan_warm_frames(synthetic_frame, jnp.arange(4), None, None, jnp.asarray(0.))
    np.testing.assert_array_equal(out["input_state"], [0., 0., 0., 1.])
    assert float(final) == 2.
