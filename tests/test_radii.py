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


def test_atom_indices_skips_solvent_ions():
    """溶剂化体系的回归: Na/Cl 不在 Bondi 表, 但它们本来就不进 PB。

    先算全体再切片会在**切片之前**就 raise —— online.py 实测挂在 S4 的
    Na 23807。守卫本身必须保住: 不给 atom_indices 仍然 raise, 而不是放宽表。
    """
    t, h1, h2, n, ca, s = _mini_topology()
    ion = t.addResidue("NA", t.addChain())
    na = t.addAtom("NA", element.sodium, ion)

    with pytest.raises(ValueError, match="Bondi"):
        assign_radii(t)

    # 注意 _mini_topology 的**创建序**(h1,n,h2,ca,s)与**返回序**(h1,h2,n,...)不同;
    # atom_indices 当集合用, 返回仍是按原子序的全长数组 —— 所以这里排序后再比
    solute = sorted(a.index for a in (h1, h2, n, ca, s))
    r = assign_radii(t, "mbondi2", atom_indices=solute)
    assert r.shape == (t.getNumAtoms(),)          # 长度不变, 调用方自己切
    assert np.isnan(r[na.index])                  # 未选中的留 NaN, 不是 0
    t0, *_ = _mini_topology()                     # 同一套溶质, 没有离子
    assert np.array_equal(r[solute], assign_radii(t0, "mbondi2")[solute])
