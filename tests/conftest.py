"""Offline test-system builder + shared fixtures.

Network is unavailable, so test systems are built from sequence:
fake backbone PDB → PDBFixer rebuilds side chains → amber14 → System.
Coordinates are not physically meaningful — T3 only compares our kernels against
OpenMM at the *same* coordinates, so geometry quality is irrelevant.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from pdbfixer import PDBFixer

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

# PDB-format 4-char atom names + element, placed on a rough extended chain
_BACKBONE = [(" N  ", "N", 0.0, 0.0, 0.0), (" CA ", "C", 0.47, 0.72, 0.0),
             (" C  ", "C", 1.72, 0.42, 0.2), (" O  ", "O", 2.10, -0.50, 0.5)]


def _write_fake_pdb(seqs, path):
    lines = []
    serial = 1
    for ci, seq in enumerate(seqs):
        chain = chr(ord("A") + ci)
        res = None
        for ri, one in enumerate(seq):
            res = ONE_TO_THREE[one.upper()]
            x0 = 5.5 * ri + 40.0 * ci  # keep the two chains well separated
            for name, elem, dx, dy, dz in _BACKBONE:
                line = (
                    f"ATOM  {serial:5d} {name} {res} {chain}{ri + 1:4d}    "
                    f"{x0 + dx:8.3f}{dy:8.3f}{dz:8.3f}  1.00  0.00          {elem:>2s}"
                )
                assert line[76:78].strip() == elem and len(line) == 78, line
                lines.append(line)
                serial += 1
        lines.append(f"TER   {serial:5d}      {res} {chain}{len(seq):4d}")
        serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def build_peptide_pair_system(seqs=("EAKLWKG", "DYQKA"), ph=7.0):
    """Returns (system, topology, positions_Å). Chain A = receptor, chain B = ligand."""
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "fake.pdb")
        _write_fake_pdb(seqs, pdb)
        fixer = PDBFixer(filename=pdb)
        fixer.findMissingResidues()
        fixer.missingResidues = {}  # nothing to model; only rebuild missing atoms
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
        ff = app.ForceField("amber14-all.xml")
        system = ff.createSystem(
            fixer.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False,
        )
        positions = np.array([[v.x, v.y, v.z] for v in fixer.positions]) * 10.0
        topology = fixer.topology
    return system, topology, positions


@pytest.fixture(scope="session")
def peptide_system():
    return build_peptide_pair_system()
