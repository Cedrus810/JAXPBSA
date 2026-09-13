"""Receptor/ligand index bookkeeping (M1)."""
from __future__ import annotations

import numpy as np
from openmm.app import Topology


def chain_indices(topology: Topology, chain_id: str) -> np.ndarray:
    """Atom indices of every residue in the chain with the given id."""
    idx = [a.index for chain in topology.chains() if chain.id == chain_id for a in chain.atoms()]
    if not idx:
        raise ValueError(f"chain {chain_id!r} not found or empty")
    return np.asarray(idx, dtype=np.int64)


def species_indices(topology: Topology, ligand_atoms) -> tuple[np.ndarray, np.ndarray]:
    """Validate a ligand atom selection; receptor is the complement (DESIGN.md §4).

    Returns (ligand_idx, receptor_idx), both sorted int64.
    """
    n = topology.getNumAtoms()
    lig = np.unique(np.asarray(list(ligand_atoms), dtype=np.int64))
    if lig.size == 0:
        raise ValueError("ligand_atoms is empty")
    if lig[0] < 0 or lig[-1] >= n:
        raise ValueError(f"ligand atom indices out of range [0, {n})")
    mask = np.zeros(n, dtype=bool)
    mask[lig] = True
    rec = np.flatnonzero(~mask).astype(np.int64)
    if rec.size == 0:
        raise ValueError("ligand selection covers all atoms; receptor is empty")
    return lig, rec
