"""Radii assignment rules (mbondi2 / bondi) and element coverage errors."""
from __future__ import annotations

import numpy as np
import pytest
from openmm.app import Topology, element

from jaxpbsa.openmm_io import assign_radii


def _mini_topology():
    t = Topology()
    c = t.addChain()
    r = t.addResidue("LIG", c)
    h1 = t.addAtom("H1", element.hydrogen, r)
    n = t.addAtom("N", element.nitrogen, r)
    h2 = t.addAtom("H2", element.hydrogen, r)
    ca = t.addAtom("CA", element.carbon, r)
    s = t.addAtom("SG", element.sulfur, r)
    t.addBond(h1, n)
    t.addBond(h2, ca)
    t.addBond(ca, n)
    return t, h1, h2, n, ca, s


def test_mbondi2_rules():
    t, h1, h2, n, ca, s = _mini_topology()
    r = assign_radii(t, "mbondi2")
    assert r[h1.index] == 1.30  # H bonded to N
    assert r[h2.index] == 1.20  # H bonded to C
    assert r[n.index] == 1.55
    assert r[ca.index] == 1.70
    assert r[s.index] == 1.80


def test_bondi_rules():
    t, h1, h2, *_ = _mini_topology()
    r = assign_radii(t, "bondi")
    assert r[h1.index] == 1.20 and r[h2.index] == 1.20


def test_unknown_element_raises():
    t = Topology()
    c = t.addChain()
    r = t.addResidue("XBE", c)
    t.addAtom("BE", element.beryllium, r)
    with pytest.raises(ValueError, match="Bondi"):
        assign_radii(t)


def test_radii_on_real_system(peptide_system):
    _, topology, _ = peptide_system
    r = assign_radii(topology, "mbondi2")
    assert r.shape == (topology.getNumAtoms(),)
    assert np.all((r >= 1.2) & (r <= 1.8))
