#!/usr/bin/env python3
"""Prepare the eight primary-table MDM2 pilots and score a small smoke subset.

This is a minimized bound-pose engineering smoke, not MD or affinity validation.
Each experimental sequence gets its own repaired atoms and force-field system.
The missing MDM2 Glu25 and peptide residue12 are restored; no ACE/NME caps are
added to these synthesized molecules. Free termini remain a documented assumption.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import openmm as mm
from openmm import app, unit
from pdbfixer import PDBFixer

ROOT = Path(__file__).resolve().parents[1] / "experiments/2026-10-10_peptide_samples"
AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))
AA1 = {v: k for k, v in AA3.items()}
TEMPLATE = "TSFAEYWNLLSP"
SEED = 20261010


def prepare(row):
    started = time.monotonic()
    seq = row["peptide_sequence"]
    out = ROOT / "prepared" / seq
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    reference = mm.Platform.getPlatformByName("Reference")
    fixer = PDBFixer(filename=str(ROOT / "raw/mdm2_3EQS.pdb"), platform=reference)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    missing = {str(k): v for k, v in fixer.missingResidues.items()}
    mutations = [f"{AA3[a]}-{i + 1}-{AA3[b]}" for i, (a, b) in enumerate(zip(TEMPLATE[:11], seq[:11])) if a != b]
    if mutations:
        fixer.applyMutations(mutations, "B")
    fixer.missingResidues[(1, 11)] = [AA3[seq[-1]]]
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=SEED)
    chains = list(fixer.topology.chains())
    observed = "".join(AA1[r.name] for r in chains[1].residues())
    if observed != seq:
        raise ValueError(f"sequence reconstruction mismatch: {observed} vs {seq}")
    ff = app.ForceField("amber/ff14SB.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    random.seed(SEED)
    modeller.addHydrogens(ff, pH=7.4, platform=reference)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                             constraints=app.HBonds, rigidWater=False, removeCMMotion=False)
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    context.setPositions(modeller.positions)
    mm.LocalEnergyMinimizer.minimize(context, tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=200)
    positions = context.getState(getPositions=True).getPositions()
    with (out / "complex.cif").open("w") as handle:
        app.PDBxFile.writeFile(modeller.topology, positions, handle, keepIds=True)
    (out / "system.xml").write_text(mm.XmlSerializer.serialize(system))
    from jaxpbsa.openmm_io import assign_radii, chain_indices, extract_nonbonded
    params = extract_nonbonded(system)
    ligand = chain_indices(modeller.topology, "B")
    receptor = chain_indices(modeller.topology, "A")
    xyz = np.asarray(positions.value_in_unit(unit.angstrom))
    np.savez(out / "parameters.npz", coords_A=xyz, charge=params.charge,
             sigma_A=params.sigma, epsilon_kcal=params.epsilon,
             radii_A=assign_radii(modeller.topology), receptor_idx=receptor, ligand_idx=ligand)
    meta = {**row, "template_pdb": "3EQS", "receptor_chain": "A", "peptide_chain": "B",
            "missing_template_residues": missing, "mutations": mutations,
            "receptor_residues": len(list(chains[0].residues())), "peptide_residues": len(seq),
            "n_atoms": system.getNumParticles(), "forcefield": "amber/ff14SB.xml",
            "prep_seed": SEED, "prep_platform": "Reference hydrogens; single-thread CPU minimization",
            "caps": "none; free termini assumption, not yet confirmed by experimental mass data",
            "net_charge_complex": float(params.charge.sum()),
            "net_charge_peptide": float(params.charge[ligand].sum()),
            "preparation_s": time.monotonic() - started,
            "evidence_scope": "sequence-specific prepared bound pose, no MD ensemble"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    del context, integrator
    print(json.dumps({"prepared": seq, "atoms": system.getNumParticles(), "seconds": meta["preparation_s"]}), flush=True)


def score(row, repeats):
    import jax
    import jaxpbsa
    from jaxpbsa.online import OnlineMMPBSA
    from jaxpbsa.pb import PBParams
    out = ROOT / "prepared" / row["peptide_sequence"]
    structure = app.PDBxFile(str(out / "complex.cif"))
    system = mm.XmlSerializer.deserialize((out / "system.xml").read_text())
    data = np.load(out / "parameters.npz")
    jaxpbsa.enable_compilation_cache(str(ROOT / "smoke/xla_cache"))
    start = time.monotonic()
    analyzer = OnlineMMPBSA(system, structure.topology, np.arange(system.getNumParticles()),
                           data["ligand_idx"], data["coords_A"],
                           fluctuation_allowance=0.0, pb_params=PBParams(), h=0.5, h_lig=0.25)
    initialization = time.monotonic() - start
    results = []
    for _ in range(repeats):
        start = time.monotonic()
        result = analyzer(data["coords_A"])
        results.append({**result, "call_s": time.monotonic() - start})
    report = {"sample": row, "jax_devices": [str(d) for d in jax.devices()],
              "protocol": "same_minimized_bound_pose_repeated; no MD; repeats are timing measurements",
              "h_A": 0.5, "h_lig_A": 0.25, "pb_params": vars(PBParams()),
              "initialization_s": initialization, "sizing": analyzer.sizing, "results": results,
              "evidence_scope": "engineering smoke only; not an experimental ranking or binding free energy"}
    (ROOT / "smoke" / f"{row['peptide_sequence']}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scored": row["peptide_sequence"], "init_s": initialization,
                      "devices": report["jax_devices"], "results": results}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--score-count", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    with (ROOT / "manifests/pilot_mdm2_8.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    (ROOT / "smoke").mkdir(parents=True, exist_ok=True)
    if args.prepare:
        for row in rows:
            prepare(row)
    for row in rows[:args.score_count]:
        score(row, args.repeats)


if __name__ == "__main__":
    main()
