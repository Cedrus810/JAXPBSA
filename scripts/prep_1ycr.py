#!/usr/bin/env python
"""1YCR (MDM2 + p53 transactivation peptide) -> canonical artifact, same format as S4.

    python scripts/prep_1ycr.py

Chain A = MDM2 25-109 (receptor), chain B = p53 17-29 ETFSDLWKLLPEN (ligand).
F19 / W23 / L26 insert into MDM2's hydrophobic pocket -- binding is hydrophobic, not
electrostatic, which is the point of having this system next to S4 (pY, -4 into a
basic pocket): does "the whole ΔG_PB error sits in the ligand" (RESULTS §15/§17)
survive when ΔG_PB is small?

Unlike 1SPS there is nothing to repair: no nonstandard residues, no missing atoms
(no REMARK 470), no HETATM. Only the disordered tails are absent (A 17-24 / 110-125,
B 15-16) and they are **truncated, not modeled**, as for S4.

**Both chains are fragments, so both get ACE/NME caps.** Free NH3+/COO- on a
fragment would add four charges that do not exist in the full proteins; in a
system whose electrostatics are already weak that is not a small perturbation of
ΔE_coul / ΔG_PB.

Reproducibility is the same recipe as prep_s4.py (fixed Python `random` seed,
Reference platform for every internal minimisation, single-thread CPU for the final
vacuum minimisation) -- see PREP_SEED there. The artifact, not the recipe, is what
downstream trusts: `load_canonical("1YCR")` checks both sha256.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import openmm as mm
import openmm.app as app
from pdbfixer import PDBFixer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_s4 import PREP_SEED, _rename_his_by_protons  # noqa: E402

from jaxpbsa.openmm_io import assign_radii, chain_indices, extract_nonbonded  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "1YCR.pdb")
OUT_DIR = os.path.join(ROOT, "data", "prepared")
NAME = "1YCR"
FF_XML = ("amber/ff14SB.xml",)
REC_CHAIN, LIG_CHAIN = "A", "B"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = mm.Platform.getPlatformByName("Reference")
    random.seed(PREP_SEED)

    fixer = PDBFixer(filename=SRC, platform=ref)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    tails = sorted(fixer.missingResidues.items())
    # 只加帽, 不补尾: 覆盖掉 SEQRES 推出来的缺失残基
    chains = list(fixer.topology.chains())
    fixer.missingResidues = {}
    for ci, ch in enumerate(chains):
        n = len(list(ch.residues()))
        fixer.missingResidues[(ci, 0)] = ["ACE"]
        fixer.missingResidues[(ci, n)] = ["NME"]
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        raise RuntimeError(f"unexpected nonstandard residues: {fixer.nonstandardResidues}")
    fixer.findMissingAtoms()
    heavy_missing = {f"{r.name}{r.id}{r.chain.id}": [a.name for a in atoms]
                     for r, atoms in fixer.missingAtoms.items()}
    fixer.addMissingAtoms(seed=PREP_SEED)
    print(f"[1/4] {SRC}: truncated tails {[(k, len(v)) for k, v in tails]}; "
          f"caps ACE/NME on {[c.id for c in chains]}; "
          f"missing heavy atoms filled: {heavy_missing or 'none'}")

    ff = app.ForceField(*FF_XML)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    random.seed(PREP_SEED)  # addHydrogens 用全局 random 放氢(见 prep_s4.PREP_SEED)
    modeller.addHydrogens(ff, pH=7.0, platform=ref)
    his_fixed = _rename_his_by_protons(modeller.topology)
    print(f"[2/4] hydrogens added: {modeller.topology.getNumAtoms()} atoms; "
          f"histidines {his_fixed}")

    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                             constraints=app.HBonds, rigidWater=False,
                             removeCMMotion=False)
    # 单线程: 多线程 CPU 的力求和顺序不确定, 最小化轨迹会分叉(prep_s4 实测 RMSD 0.78 Å)
    context = mm.Context(system, mm.VerletIntegrator(0.001 * mm.unit.picoseconds),
                         mm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    context.setPositions(modeller.positions)
    kj = mm.unit.kilojoule_per_mole
    e0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 2000)
    e1 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
    positions = context.getState(getPositions=True).getPositions()
    del context
    print(f"[3/4] vacuum minimisation {e0:.2e} -> {e1:.1f} kJ/mol")

    params = extract_nonbonded(system)
    radii = assign_radii(modeller.topology, "mbondi2")
    rec_idx = chain_indices(modeller.topology, REC_CHAIN)
    lig_idx = chain_indices(modeller.topology, LIG_CHAIN)
    q = params.charge
    net_q, rec_q, lig_q = q.sum(), q[rec_idx].sum(), q[lig_idx].sum()
    if rec_idx.size + lig_idx.size != system.getNumParticles():
        raise RuntimeError("receptor + ligand do not cover all atoms")
    for tag, v in (("complex", net_q), ("receptor", rec_q), ("ligand", lig_q)):
        if abs(v - round(v)) > 1e-3:
            raise RuntimeError(f"{tag} net charge {v:+.4f} is not an integer")
    res_counts = Counter(r.name for r in modeller.topology.residues())
    for cap in ("ACE", "NME"):
        if res_counts.get(cap) != 2:
            raise RuntimeError(f"expected 2 {cap} caps, got {res_counts.get(cap)}")
    print(f"[4/4] atoms={system.getNumParticles()} (receptor {rec_idx.size} + ligand "
          f"{lig_idx.size}); net charge {net_q:+.3f} = {rec_q:+.3f} + {lig_q:+.3f}; "
          f"radii {radii.min():.2f}-{radii.max():.2f} Å")

    out_pdb = os.path.join(OUT_DIR, f"{NAME}_complex.pdb")
    out_cif = os.path.join(OUT_DIR, f"{NAME}_complex.cif")
    out_sys = os.path.join(OUT_DIR, f"{NAME}_system.xml")
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, positions, fh, keepIds=True)
    with open(out_cif, "w") as fh:
        app.PDBxFile.writeFile(modeller.topology, positions, fh, keepIds=True)
    with open(out_sys, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()  # noqa: E731
    meta = {
        "source": "1YCR chains A (MDM2 25-109) + B (p53 17-29 ETFSDLWKLLPEN), "
                  "ACE/NME caps on both",
        "forcefield": list(FF_XML),
        "radii_model": "mbondi2",
        "prep_seed": PREP_SEED,
        "canonical_structure": os.path.basename(out_cif),
        "canonical_system": os.path.basename(out_sys),
        "canonical_sha256": sha(out_cif),
        "system_sha256": sha(out_sys),
        "receptor_idx": [int(i) for i in rec_idx],
        "ligand_idx": [int(i) for i in lig_idx],
        "n_atoms": system.getNumParticles(),
        "n_receptor": int(rec_idx.size),
        "n_ligand": int(lig_idx.size),
        "net_charge_complex": float(net_q),
        "net_charge_receptor": float(rec_q),
        "net_charge_ligand": float(lig_q),
        "receptor_chain": REC_CHAIN,
        "ligand_chain": LIG_CHAIN,
        "residue_counts": dict(res_counts),
        "histidine_renamed": his_fixed,
        "truncated_tails": {f"chain{k[0]}@{k[1]}": v for k, v in tails},
    }
    with open(os.path.join(OUT_DIR, f"{NAME}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"       wrote {out_cif} sha256 {meta['canonical_sha256'][:16]}…, "
          f"{out_sys} sha256 {meta['system_sha256'][:16]}…")


if __name__ == "__main__":
    main()
