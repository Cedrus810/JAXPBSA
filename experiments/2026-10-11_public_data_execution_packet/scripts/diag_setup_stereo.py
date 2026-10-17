#!/usr/bin/env python3
"""Why did setup stage 2 invert A:THR26 in P12Q_L10A/model02 (p2_diagnostic_model02.log, 2026-10-12)?

    cd /home/ruigengji/JAXPBSA
    $PY experiments/2026-10-11_public_data_execution_packet/scripts/diag_setup_stereo.py \
        --sample P12Q_L10A --model-id model02 --watch A:THR26

Rebuilds the solvated system exactly as `run_peptide_md.py setup` does (same
prepared model, solvation seed, force field, restraint, tolerance, platform) and
records, without writing anything under md/:

  start      Thr26 environment in the solvated start: closest atoms (solute, water, ion)
             to each atom of the watched residue
  stage1     after the restrained minimisation: largest |theta - theta0| of the stereo
             restraint terms, the watched chirality, and the largest unrestrained forces
             (k_stereo = 0) on atoms within 6 A of the watched CA
  stage2     (a) one unrestrained minimisation exactly as setup does -> does the flip
             reproduce?  (b) from the same stage-1 state, the same minimisation cut into
             --chunk-iter iteration pieces, recording the watched chirality, energy and
             closest contacts after each piece (where along the path it flips; L-BFGS
             restarts at each piece, so this path is close to, not identical with, (a))

Output: <packet>/diagnostics/setup_stereo_<date>/<sample>_<model>/trace.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import peptide_geometry as geo                         # noqa: E402
import run_peptide_md as md                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--watch", default="A:THR26", help="residue label chain:RESnum")
    ap.add_argument("--seed", type=int, default=20261011, help="solvation seed (setup default)")
    ap.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU", "Reference"])
    ap.add_argument("--device-index", default="0")
    ap.add_argument("--chunk-iter", type=int, default=25)
    ap.add_argument("--out", type=Path, default=PACKET / "diagnostics" / f"setup_stereo_{time.strftime('%Y%m%d')}")
    a = ap.parse_args()

    import numpy as np
    import openmm as mm
    from openmm import app, unit

    row = md.samples(PACKET)[a.sample]
    prep, _ = md.setup_paths(PACKET, row["peptide_name"], a.model_id)
    prep_rec = json.loads((prep / "prep_record.json").read_text())
    cif = app.PDBxFile(str(prep / "complex_dry.cif"))
    ff = app.ForceField(*md.FF_XML)
    modeller = app.Modeller(cif.topology, cif.positions)
    random.seed(a.seed); np.random.seed(a.seed % 2**32)
    modeller.addSolvent(ff, model="tip3p", padding=md.PADDING_NM * unit.nanometer,
                        ionicStrength=md.IONIC_M * unit.molar, neutralize=True)
    top = modeller.topology
    n_sol = prep_rec["n_atoms"]
    atoms = list(top.atoms())
    system = ff.createSystem(top, nonbondedMethod=app.PME, nonbondedCutoff=md.CUTOFF_NM * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True, removeCMMotion=True)
    terms = geo.stereo_terms(geo.residues_from_topology(top, n_sol))
    before = geo.measure(terms, modeller.positions.value_in_unit(unit.angstrom))
    restraint = mm.CustomTorsionForce("0.5*k_stereo*d^2; d=min(dt, 2*pi-dt); dt=abs(theta-theta0); pi=3.141592653589793")
    restraint.addGlobalParameter("k_stereo", md.STEREO_K)
    restraint.addPerTorsionParameter("theta0")
    for t, m in zip(terms, before):
        restraint.addTorsion(*t["atoms"], [np.radians(m["deg"])])
    system.addForce(restraint)
    integ = mm.LangevinMiddleIntegrator(md.T_K * unit.kelvin, md.FRICTION_PS / unit.picosecond,
                                        md.DT_PS * unit.picoseconds)
    plat, props = md.make_platform(a.platform, a.device_index if a.platform in ("CUDA", "OpenCL") else None)
    sim = app.Simulation(top, system, integ, plat, props)
    sim.context.setPositions(modeller.positions)
    tol = 10.0 * unit.kilojoule_per_mole / unit.nanometer

    chain, resnum = a.watch.split(":")[0], int("".join(c for c in a.watch.split(":")[1] if c.isdigit()))
    watch_atoms = [x.index for x in atoms if x.residue.chain.id == chain and int(x.residue.id) == resnum]
    ca = next(i for i in watch_atoms if atoms[i].name == "CA")
    wterm = next(i for i, t in enumerate(terms) if t["kind"] == "chirality" and t["atoms"][1] == ca)
    nbr = {x.index: set() for x in atoms}
    for b1, b2 in top.bonds():
        nbr[b1.index].add(b2.index); nbr[b2.index].add(b1.index)

    def within3(i):
        s = {i} | nbr[i]
        for _ in range(2):
            s |= {k for j in s for k in nbr[j]}
        return s
    excl = {i: within3(i) for i in watch_atoms}
    lab = lambda i: f"{atoms[i].residue.chain.id}:{atoms[i].residue.name}{atoms[i].residue.id}.{atoms[i].name}"

    def snapshot(state, n_contacts=6):
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        box = np.array([state.getPeriodicBoxVectors()[k][k].value_in_unit(unit.angstrom) for k in range(3)])
        m = geo.measure(terms, xyz[:n_sol])
        cont = []
        for i in watch_atoms:
            d = xyz - xyz[i]
            d -= box * np.round(d / box)
            r = np.linalg.norm(d, axis=1)
            for j in np.argsort(r)[:20]:
                if int(j) not in excl[i]:
                    cont.append((round(float(r[j]), 3), lab(i), lab(int(j))))
        cont.sort()
        return xyz, dict(watch_chirality_deg=round(m[wterm]["deg"], 1),
                         E_kJ_mol=round(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), 1),
                         closest_contacts_A=cont[:n_contacts],
                         class_changes_vs_start=geo.class_changes(before, m))

    trace = dict(sample=a.sample, model_id=a.model_id, watch=a.watch, platform=a.platform,
                 n_atoms=top.getNumAtoms(), solvation_seed=a.seed)
    s0 = sim.context.getState(getPositions=True, getEnergy=True)
    _, trace["start"] = snapshot(s0, 10)

    sim.minimizeEnergy(tolerance=tol, maxIterations=0)                         # stage 1 as in setup
    s1 = sim.context.getState(getPositions=True, getEnergy=True)
    xyz1, trace["stage1"] = snapshot(s1, 10)
    m1 = geo.measure(terms, xyz1[:n_sol])
    dev = sorted(((abs((x["deg"] - b["deg"] + 180) % 360 - 180), x["label"], x["kind"]) for b, x in zip(before, m1)),
                 reverse=True)[:6]
    trace["stage1"]["largest_restraint_deviation_deg"] = [(round(d, 1), l, k) for d, l, k in dev]
    sim.context.setParameter("k_stereo", 0.0)
    f = sim.context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    near = np.where(np.linalg.norm(xyz1 - xyz1[ca], axis=1) < 6.0)[0]
    fn = np.linalg.norm(f[near], axis=1)
    trace["stage1"]["largest_unrestrained_force_near_watch_kJ_mol_nm"] = [
        (round(float(fn[k]), 1), lab(int(near[k]))) for k in np.argsort(fn)[::-1][:8]]
    trace["stage1"]["largest_unrestrained_force_anywhere_kJ_mol_nm"] = [
        (round(float(np.linalg.norm(f[k])), 1), lab(int(k))) for k in np.argsort(np.linalg.norm(f, axis=1))[::-1][:5]]

    sim.minimizeEnergy(tolerance=tol, maxIterations=0)                         # (a) stage 2 as in setup
    _, trace["stage2_single_call"] = snapshot(sim.context.getState(getPositions=True, getEnergy=True), 6)

    sim.context.setState(s1)                                                   # (b) same, in pieces
    sim.context.setParameter("k_stereo", 0.0)
    path, last = [], None
    for k in range(400):
        sim.minimizeEnergy(tolerance=tol, maxIterations=a.chunk_iter)
        st = sim.context.getState(getPositions=True, getEnergy=True)
        _, snap = snapshot(st, 4)
        snap["iterations"] = (k + 1) * a.chunk_iter
        path.append(snap)
        e = snap["E_kJ_mol"]
        if e == last:          # minimiser returned at once: converged to the setup tolerance
            break
        last = e
    trace["stage2_pieces"] = path

    out = a.out / f"{a.sample}_{a.model_id}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps({k: trace[k] for k in ("start", "stage1", "stage2_single_call")}, indent=1))
    flips = [p for p in path if p["class_changes_vs_start"]]
    print("pieces:", len(path), "first piece with a class change:",
          json.dumps(flips[0] if flips else None))
    for p in path[:: max(1, len(path) // 15)]:
        print(p["iterations"], p["watch_chirality_deg"], p["E_kJ_mol"], p["closest_contacts_A"][:2])
    print("trace:", out / "trace.json")


if __name__ == "__main__":
    main()
