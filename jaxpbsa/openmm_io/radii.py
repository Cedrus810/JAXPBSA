"""Atomic radii for PB: Bondi heavy atoms + mbondi2 hydrogen rules (M1, DESIGN.md §1).

mbondi2 (Amber, igb=2/5 惯例, 也是 APBS 对拍用的半径档):
    H bonded to N → 1.30 Å, 其他 H → 1.20 Å; 重原子用 Bondi.
bondi: 所有 H 1.20 Å, 重原子 Bondi.
"""
from __future__ import annotations

import numpy as np
from openmm.app import Topology

BONDI = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.50, "F": 1.47,
    "S": 1.80, "P": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}

_MODELS = ("bondi", "mbondi2")


def assign_radii(topology: Topology, model: str = "mbondi2") -> np.ndarray:
    if model not in _MODELS:
        raise ValueError(f"unknown radii model {model!r}; available: {_MODELS}")

    h_on_nitrogen = set()
    for a1, a2 in topology.bonds():
        for x, y in ((a1, a2), (a2, a1)):
            if x.element is not None and y.element is not None:
                if x.element.symbol == "H" and y.element.symbol == "N":
                    h_on_nitrogen.add(x.index)

    radii = np.empty(topology.getNumAtoms())
    for atom in topology.atoms():
        if atom.element is None:
            raise ValueError(
                f"atom {atom.index} ({atom.name}) has no element (virtual site?); "
                "PB radii are undefined for it"
            )
        sym = atom.element.symbol
        if sym == "H":
            radii[atom.index] = 1.30 if (model == "mbondi2" and atom.index in h_on_nitrogen) else 1.20
        elif sym in BONDI:
            radii[atom.index] = BONDI[sym]
        else:
            raise ValueError(
                f"element {sym} (atom {atom.index}, {atom.name}) not in Bondi table; "
                "extend BONDI explicitly rather than guessing"
            )
    return radii
