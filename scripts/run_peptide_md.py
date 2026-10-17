#!/usr/bin/env python3
"""Solvated multi-replica MD for the public-data peptide packet (P1 entry).

Two subcommands; both print a static plan with --dry-run (no OpenMM import):

    # once per (sample, model): solvate + minimise, shared by all replicas
    python scripts/run_peptide_md.py setup --sample PMI --model-id model01 --platform CUDA
    # one replica = one run_id row of <packet>/proposed_runs.csv
    python scripts/run_peptide_md.py run --run-id PMI_model01_rep01 --platform CUDA --prod-ns 10

Physics follows packet protocol.md §2 and the existing scripts/run_s4_md.py
pipeline: ff14SB + TIP3P, 1.2 nm padding, 0.15 M NaCl + neutralise, PME 1.0 nm,
HBonds, 2 fs, LangevinMiddle 298.15 K 1/ps, MC barostat 1 atm in NPT/production.
No production restraints.  --eq-restraint-k (default 0, i.e. same as
run_s4_md.py) optionally restrains solute heavy atoms during NVT/NPT only.

Seeds come from the run row: velocity seed -> setVelocitiesToTemperature at NVT
start.  Integrator/barostat seeds are base + SEED_STRIDE*k with context slot k
(0 NVT, 1 NPT, 2 production, 3 + c production resumed at chunk c), so stages and
resumes never replay one stream and never reach another row's base seed (the
dry-run checks this over all rows).  Every seed used is appended to status.json.

Resume: re-run the same command.  Completed stages are skipped; production is
written in chunks (`prod/chunk_NNNN.dcd` + `prod/chunk_NNNN_state.xml`), a chunk
counts only after its state file and status.json are written, and an unfinished
chunk is rewritten from the previous chunk's state.  Raising --prod-ns later
extends the run; it must stay a multiple of --chunk-ps.  Velocities/positions
resume from XML state (portable across GPUs); the Langevin RNG stream does not,
so a resumed replica is statistically, not bitwise, the same as an unbroken one.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform as host_platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
FF_XML = ("amber/ff14SB.xml", "amber/tip3p_standard.xml")
T_K, DT_PS, FRICTION_PS, P_ATM = 298.15, 0.002, 1.0, 1.0
PADDING_NM, IONIC_M, CUTOFF_NM = 1.2, 0.15, 1.0
# Stage-1 setup minimisation restrains every omega and CA-chirality dihedral to its
# prepared value.  PDBFixer's soft-core rebuild can leave rebuilt terminal residues
# overlapping (3EQS Glu25 CG-O 0.93 A); an unrestrained first minimisation flipped
# Glu25-Thr26 trans->cis (smoke 2026-10-11).  1e4 kJ/mol/rad^2 makes reaching 90 deg
# cost 1.2e4 kJ/mol, ~600x the ff14SB X-C-N-X barrier (2.5 kcal/mol, n=2 -> 21 kJ/mol);
# it is zero at the prepared geometry and is removed for stage 2 and all MD.  Whether
# it suffices is checked by the cis/trans + L/D class comparison afterwards.
STEREO_K = 1.0e4
SEED_STRIDE = 7919  # prime, larger than any base-seed gap between rows


def context_seed(base: int, stage: str, chunk: int | None) -> int:
    k = {"nvt": 0, "npt": 1}.get(stage, 2 if chunk is None else 3 + chunk)
    return base + SEED_STRIDE * k


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def samples(packet: Path) -> dict:
    rows = list(csv.DictReader((packet / "sample_structure_audit.csv").open()))
    return {**{r["peptide_name"]: r for r in rows}, **{r["sample_id"]: r for r in rows}}


def setup_paths(packet: Path, peptide_name: str, model_id: str) -> tuple[Path, Path]:
    return packet / "prepared" / peptide_name / model_id, packet / "md" / peptide_name / model_id / "setup"


def make_platform(name: str, device_index: str | None):
    import openmm as mm
    plat = mm.Platform.getPlatformByName(name)
    props = {}
    if name == "CUDA":
        props["CudaPrecision"] = "mixed"
        if device_index is not None:
            props["CudaDeviceIndex"] = device_index
    elif name == "OpenCL":
        props["OpenCLPrecision"] = "mixed"
        if device_index is not None:
            props["OpenCLDeviceIndex"] = device_index
    return plat, props


def device_info(context) -> dict:
    plat = context.getPlatform()
    info = {"platform": plat.getName()}
    for key in plat.getPropertyNames():
        if "Device" in key or "Precision" in key:
            info[key] = plat.getPropertyValue(context, key)
    return info


# ------------------------------------------------------------------ setup
JUNCTIONS = ((("A", 25), ("A", 26)), (("B", 11), ("B", 12)))


def check_c3(prep_rec, top, n_solute, xyz_A, system, geo) -> dict:
    """Prep v2 design §5 C3: junction geometry after the unrestrained minimisation.

    Heavy-atom ff14SB bond/angle terms touching the junction residues and mutated
    residues within 4 sigma at 298 K, plus the junction / Pro-ring table.  Raises
    (nothing written) on failure.  Applied to every setup; the v1 setups that already
    exist were measured normal after minimisation (handoff §9.2).
    """
    import numpy as np
    import openmm as mm
    pc = prep_rec["peptide_chain"]
    focus_res = {r for j in JUNCTIONS for r in j} | {(pc, int(m.split("-")[1])) for m in prep_rec["mutations"]}
    atoms = list(top.atoms())[:n_solute]
    is_h = [a.element is not None and a.element.symbol == "H" for a in atoms]
    focus = [a.index for a in atoms if (a.residue.chain.id, int(a.residue.id)) in focus_res]
    bonds, angles = [], []
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for n in range(f.getNumBonds()):
                i, j, r0, k = f.getBondParameters(n)
                if i < n_solute and j < n_solute and not (is_h[i] or is_h[j]):
                    bonds.append((i, j, r0._value, k._value))
        elif isinstance(f, mm.HarmonicAngleForce):
            for n in range(f.getNumAngles()):
                i, j, l, t0, k = f.getAngleParameters(n)
                if max(i, j, l) < n_solute and not (is_h[i] or is_h[j] or is_h[l]):
                    angles.append((i, j, l, t0._value, k._value))
    xyz_A = np.asarray(xyz_A)[:n_solute]
    lab = lambda i: f"{atoms[i].residue.chain.id}:{atoms[i].residue.name}{atoms[i].residue.id}.{atoms[i].name}"
    fails = [f"{kind} {'-'.join(lab(i) for i in idx)} {val} (ff14SB {x0} +- {lim})"
             for kind, idx, val, x0, lim in geo.harmonic_outliers(xyz_A / 10.0, bonds, angles, focus)]
    vals, jf = geo.junction_report(geo.residues_from_topology(top, n_solute), xyz_A, JUNCTIONS)
    fails += jf
    if fails:
        raise RuntimeError(f"C3 junction geometry after setup minimisation failed: {fails[:12]}; nothing written")
    return dict(passed=True, focus_residues=sorted(f"{c}:{r}" for c, r in focus_res), junction=vals)


def cmd_setup(a) -> None:
    row = samples(a.packet)[a.sample]
    prep, setup = setup_paths(a.packet, row["peptide_name"], a.model_id)
    plan = dict(stage="setup", sample=row["peptide_name"], model_id=a.model_id, prepared=str(prep),
                setup=str(setup), prepared_exists=(prep / "prep_record.json").exists(),
                setup_done=(setup / "setup_record.json").exists(), solvation_seed=a.seed)
    if a.dry_run:
        print(json.dumps(plan)); return
    if plan["setup_done"] and not a.force:
        print(json.dumps({"skip_setup_done": str(setup)})); return
    if not plan["prepared_exists"]:
        raise SystemExit(f"no prep_record.json in {prep}; run prep_peptide_sample.py first")

    import numpy as np
    import openmm as mm
    from openmm import app, unit
    t0 = time.monotonic()
    setup.mkdir(parents=True, exist_ok=True)
    prep_rec = json.loads((prep / "prep_record.json").read_text())
    cif = app.PDBxFile(str(prep / "complex_dry.cif"))
    ff = app.ForceField(*FF_XML)
    modeller = app.Modeller(cif.topology, cif.positions)
    random.seed(a.seed); np.random.seed(a.seed % 2**32)
    modeller.addSolvent(ff, model="tip3p", padding=PADDING_NM * unit.nanometer,
                        ionicStrength=IONIC_M * unit.molar, neutralize=True)
    top = modeller.topology
    n_solute = prep_rec["n_atoms"]
    solute = list(top.atoms())[:n_solute]
    ref = np.load(prep / "parameters.npz")
    if [x.name for x in solute] != list(ref["atom_name"]) or [x.residue.name for x in solute] != list(ref["res_name"]):
        raise RuntimeError("solvated topology does not start with the prepared solute in order")
    counts = {}
    for r in list(top.residues())[len(list(cif.topology.residues())):]:
        counts[r.name] = counts.get(r.name, 0) + 1

    system = ff.createSystem(top, nonbondedMethod=app.PME, nonbondedCutoff=CUTOFF_NM * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True, removeCMMotion=True)
    system_xml = mm.XmlSerializer.serialize(system)  # saved without the stage-1 restraint

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import peptide_geometry as geo
    terms = geo.stereo_terms(geo.residues_from_topology(top, n_solute))
    xyz0 = modeller.positions.value_in_unit(unit.angstrom)
    before = geo.measure(terms, xyz0)
    restraint = mm.CustomTorsionForce("0.5*k_stereo*d^2; d=min(dt, 2*pi-dt); dt=abs(theta-theta0); pi=3.141592653589793")
    restraint.addGlobalParameter("k_stereo", STEREO_K)
    restraint.addPerTorsionParameter("theta0")
    for t, m in zip(terms, before):
        restraint.addTorsion(*t["atoms"], [np.radians(m["deg"])])
    system.addForce(restraint)

    integ = mm.LangevinMiddleIntegrator(T_K * unit.kelvin, FRICTION_PS / unit.picosecond, DT_PS * unit.picoseconds)
    plat, props = make_platform(a.platform, a.device_index)
    sim = app.Simulation(top, system, integ, plat, props)
    sim.context.setPositions(modeller.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    t_min = time.monotonic()
    tol = 10.0 * unit.kilojoule_per_mole / unit.nanometer
    sim.minimizeEnergy(tolerance=tol, maxIterations=0)                 # stage 1: stereo-restrained
    st1 = sim.context.getState(getPositions=True, getEnergy=True)
    mid = geo.measure(terms, st1.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    sim.context.setParameter("k_stereo", 0.0)
    # force the system feels the moment the restraint is released (P12Q_L10A/model02: ~5500
    # kJ/mol/nm on Thr26/Ser11 CA preceded their inversion) -- recorded, not a gate
    f_rel = sim.context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    f_norm = np.linalg.norm(f_rel[:n_solute], axis=1)
    atoms_all = list(top.atoms())
    lab = lambda i: f"{atoms_all[i].residue.chain.id}:{atoms_all[i].residue.name}{atoms_all[i].residue.id}.{atoms_all[i].name}"
    release_force = [(round(float(f_norm[i]), 1), lab(int(i))) for i in np.argsort(f_norm)[::-1][:5]]
    sim.minimizeEnergy(tolerance=tol, maxIterations=0)                 # stage 2: unrestrained
    state = sim.context.getState(getPositions=True, getEnergy=True, getForces=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    fmax = float(np.abs(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).max())
    after = geo.measure(terms, state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    changes = {"stage1": geo.class_changes(before, mid), "stage2": geo.class_changes(before, after)}
    if changes["stage1"] or changes["stage2"]:
        raise RuntimeError(f"cis/trans or L/D class changed during setup minimisation: {changes}; nothing written")
    c3 = check_c3(prep_rec, top, n_solute, state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                  system, geo)
    drift = max(abs((b["deg"] - x["deg"] + 180) % 360 - 180) for b, x in zip(before, after))

    with (setup / "solvated.cif").open("w") as fh:
        app.PDBxFile.writeFile(top, state.getPositions(), fh, keepIds=True)
    with (setup / "solvated.pdb").open("w") as fh:
        app.PDBFile.writeFile(top, state.getPositions(), fh)
    (setup / "system.xml").write_text(system_xml)
    (setup / "minimized_state.xml").write_text(mm.XmlSerializer.serialize(state))
    box = [v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors()]
    write_json(setup / "setup_record.json", dict(
        sample_id=row["sample_id"], peptide_name=row["peptide_name"], model_id=a.model_id,
        prepared=str(prep), solvation_seed=a.seed, forcefield=list(FF_XML), padding_nm=PADDING_NM,
        ionic_strength_M=IONIC_M, neutralize=True, n_solute_atoms=n_solute, n_atoms=top.getNumAtoms(),
        added_residues=counts, box_vectors_nm=[[round(c, 4) for c in v] for v in box],
        pme_cutoff_nm=CUTOFF_NM, constraints="HBonds", rigid_water=True,
        minimisation=dict(tolerance_kJ_mol_nm=10.0, max_iterations="until converged",
                          stage1=f"omega + CA-chirality torsion restraints, k={STEREO_K} kJ/mol/rad^2, theta0=prepared",
                          stage2="unrestrained", E_before_kJ_mol=e0,
                          E_stage1_incl_restraint_kJ_mol=st1.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                          E_after_kJ_mol=e1, max_force_after_kJ_mol_nm=fmax,
                          stereo_terms=len(terms), stereo_class_changes=0,
                          max_stereo_dihedral_shift_deg=round(drift, 2),
                          largest_solute_force_at_restraint_release_kJ_mol_nm=release_force,
                          c3_checks=c3,
                          wall_s=round(time.monotonic() - t_min, 2)),
        device=device_info(sim.context), openmm=mm.__version__, host=host_platform.node(),
        wall_s=round(time.monotonic() - t0, 2)))
    print(json.dumps({"setup_done": str(setup), "n_atoms": top.getNumAtoms(), "added": counts,
                      "E_min_kJ_mol": e1, "wall_s": round(time.monotonic() - t0, 2)}), flush=True)


# ------------------------------------------------------------------ replica
def run_row(packet: Path, run_id: str) -> dict:
    for r in csv.DictReader((packet / "proposed_runs.csv").open()):
        if r["run_id"] == run_id:
            return r
    raise SystemExit(f"run_id {run_id} not in proposed_runs.csv")


def seed_collisions(packet: Path, n_chunks: int) -> list:
    """Integrator/barostat/velocity seeds every row would use, checked for overlaps."""
    seen, clash = {}, []
    for r in csv.DictReader((packet / "proposed_runs.csv").open()):
        used = [("velocity", int(r["proposed_velocity_seed"]))]
        for kind in ("integrator", "barostat"):
            base = int(r[f"proposed_{kind}_seed"])
            used += [(kind, context_seed(base, st, None)) for st in ("nvt", "npt", "prod")]
            used += [(kind, context_seed(base, "prod", c)) for c in range(1, n_chunks)]
        for kind, s in used:
            if s >= 2**31:
                clash.append((r["run_id"], kind, s, "exceeds int32"))
            if s in seen and seen[s][0] != r["run_id"]:
                clash.append((r["run_id"], kind, s, seen[s]))
            seen.setdefault(s, (r["run_id"], kind))
    return clash


def cmd_run(a) -> None:
    row = run_row(a.packet, a.run_id)
    srow = samples(a.packet)[row["sample_id"]]
    out = Path(row["output_directory"])
    _, setup = setup_paths(a.packet, srow["peptide_name"], row["model_id"])
    if out.parent.parent.name != srow["peptide_name"] or out.parent.name != row["model_id"]:
        raise SystemExit(f"output_directory {out} does not match {srow['peptide_name']}/{row['model_id']}")
    frame_steps = round(a.dcd_ps / DT_PS)
    chunk_steps = round(a.chunk_ps / DT_PS)
    prod_steps = round(a.prod_ns * 1000 / DT_PS)
    if chunk_steps % frame_steps or prod_steps % chunk_steps:
        raise SystemExit("--chunk-ps must be a multiple of --dcd-ps and --prod-ns a multiple of --chunk-ps")
    seeds = dict(velocity=int(row["proposed_velocity_seed"]), integrator=int(row["proposed_integrator_seed"]),
                 barostat=int(row["proposed_barostat_seed"]))
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    plan = dict(stage="run", run_id=a.run_id, sample=srow["peptide_name"], model_id=row["model_id"],
                replica=row["replica"], out=str(out), setup=str(setup),
                setup_done=(setup / "setup_record.json").exists(), seeds=seeds,
                nvt_steps=round(a.nvt_ps / DT_PS), npt_steps=round(a.npt_ps / DT_PS), prod_steps=prod_steps,
                chunk_steps=chunk_steps, frame_steps=frame_steps, n_chunks=prod_steps // chunk_steps,
                eq_restraint_k=a.eq_restraint_k,
                resume_from=None if status is None else {k: status[k] for k in ("nvt_done", "npt_done", "chunks_done")})
    if a.dry_run:
        print(json.dumps({**plan, "seed_collisions": seed_collisions(a.packet, plan["n_chunks"])})); return
    if not plan["setup_done"]:
        raise SystemExit(f"no setup_record.json in {setup}; run `setup` for this sample/model first")

    import openmm as mm
    from openmm import app, unit
    (out / "prod").mkdir(parents=True, exist_ok=True)
    if status is None:
        status = dict(run_id=a.run_id, sample_id=row["sample_id"], peptide_name=srow["peptide_name"],
                      model_id=row["model_id"], replica=int(row["replica"]), setup=str(setup), seeds=seeds,
                      dt_ps=DT_PS, frame_steps=frame_steps, chunk_steps=chunk_steps,
                      nvt_steps=plan["nvt_steps"], npt_steps=plan["npt_steps"], eq_restraint_k=a.eq_restraint_k,
                      nvt_done=False, npt_done=False, chunks_done=0, prod_target_history=[],
                      contexts=[], stage_wall_s={}, complete=False)
    for key in ("frame_steps", "chunk_steps", "nvt_steps", "npt_steps", "eq_restraint_k"):
        if status[key] != plan[key]:
            raise SystemExit(f"{key}={plan[key]} differs from the existing run ({status[key]}); "
                             "use the original value or a new output directory")
    if not status["prod_target_history"] or status["prod_target_history"][-1] != prod_steps:
        status["prod_target_history"].append(prod_steps)
    write_json(status_path, status)

    pdbx = app.PDBxFile(str(setup / "solvated.cif"))
    top = pdbx.topology
    sys_xml = (setup / "system.xml").read_text()
    plat, props = make_platform(a.platform, a.device_index)
    T = T_K * unit.kelvin

    def build(stage: str, chunk: int | None):
        system = mm.XmlSerializer.deserialize(sys_xml)
        if stage in ("nvt", "npt") and a.eq_restraint_k > 0:
            ref = mm.XmlSerializer.deserialize((setup / "minimized_state.xml").read_text()).getPositions()
            f = mm.CustomExternalForce("0.5*k_eq*periodicdistance(x,y,z,x0,y0,z0)^2")
            f.addGlobalParameter("k_eq", a.eq_restraint_k)
            for p in ("x0", "y0", "z0"):
                f.addPerParticleParameter(p)
            for atom in list(top.atoms())[:json.loads((setup / "setup_record.json").read_text())["n_solute_atoms"]]:
                if atom.element.symbol != "H":
                    f.addParticle(atom.index, ref[atom.index].value_in_unit(unit.nanometer))
            system.addForce(f)
        used = {"stage": stage, "chunk": chunk, "integrator_seed": context_seed(seeds["integrator"], stage, chunk)}
        if stage in ("npt", "prod"):
            baro = mm.MonteCarloBarostat(P_ATM * unit.atmosphere, T)
            used["barostat_seed"] = context_seed(seeds["barostat"], stage, chunk)
            baro.setRandomNumberSeed(used["barostat_seed"])
            system.addForce(baro)
        integ = mm.LangevinMiddleIntegrator(T, FRICTION_PS / unit.picosecond, DT_PS * unit.picoseconds)
        integ.setRandomNumberSeed(used["integrator_seed"])
        sim = app.Simulation(top, system, integ, plat, props)
        used["device"] = device_info(sim.context)
        status["contexts"].append(used)
        return sim

    def load(sim, path: Path):
        st = mm.XmlSerializer.deserialize(path.read_text())
        sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
        sim.context.setPositions(st.getPositions())
        try:
            sim.context.setVelocities(st.getVelocities())
        except Exception:  # minimized_state.xml has no velocities; NVT draws them
            pass

    def save(sim, path: Path):
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        tmp = path.with_name(f".{path.name}.partial")
        tmp.write_text(mm.XmlSerializer.serialize(st))
        os.replace(tmp, path)

    def run_steps(sim, n, label):
        t = time.monotonic()
        sim.reporters = [app.StateDataReporter(str(out / f"{label}.log"), max(1, min(n, 5000)), step=True,
                                               time=True, potentialEnergy=True, temperature=True,
                                               volume=True, density=True, speed=True)]
        sim.step(n)
        sim.reporters = []
        status["stage_wall_s"][label] = round(time.monotonic() - t, 2)

    if not status["nvt_done"]:
        sim = build("nvt", None)
        load(sim, setup / "minimized_state.xml")
        sim.context.setVelocitiesToTemperature(T, seeds["velocity"])
        run_steps(sim, plan["nvt_steps"], "nvt")
        save(sim, out / "nvt_end.xml")
        status["nvt_done"] = True; write_json(status_path, status)
    if not status["npt_done"]:
        sim = build("npt", None)
        load(sim, out / "nvt_end.xml")
        run_steps(sim, plan["npt_steps"], "npt")
        save(sim, out / "npt_end.xml")
        status["npt_done"] = True; write_json(status_path, status)

    n_chunks = prod_steps // chunk_steps
    sim = None
    while status["chunks_done"] < n_chunks:
        c = status["chunks_done"]
        if sim is None:
            sim = build("prod", None if c == 0 else c)
            load(sim, out / "npt_end.xml" if c == 0 else out / "prod" / f"chunk_{c - 1:04d}_state.xml")
        t = time.monotonic()
        dcd_path = out / "prod" / f"chunk_{c:04d}.dcd"
        with dcd_path.open("wb") as fh:
            dcd = app.DCDFile(fh, top, DT_PS * unit.picoseconds, firstStep=c * chunk_steps, interval=frame_steps)
            sim.reporters = [app.StateDataReporter(str(out / "prod" / f"chunk_{c:04d}.log"), 5000, step=True,
                                                   time=True, potentialEnergy=True, temperature=True,
                                                   volume=True, density=True, speed=True)]
            for _ in range(chunk_steps // frame_steps):
                sim.step(frame_steps)
                st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
                dcd.writeModel(st.getPositions(), periodicBoxVectors=st.getPeriodicBoxVectors())
            sim.reporters = []
        save(sim, out / "prod" / f"chunk_{c:04d}_state.xml")
        status["chunks_done"] = c + 1
        status["stage_wall_s"][f"prod_chunk_{c:04d}"] = round(time.monotonic() - t, 2)
        write_json(status_path, status)
        print(json.dumps({"run_id": a.run_id, "chunk": c, "of": n_chunks,
                          "wall_s": status["stage_wall_s"][f"prod_chunk_{c:04d}"]}), flush=True)
    status["complete"] = status["chunks_done"] * chunk_steps >= prod_steps
    status["openmm"] = mm.__version__
    status["host"] = host_platform.node()
    write_json(status_path, status)
    print(json.dumps({"run_id": a.run_id, "complete": status["complete"], "chunks_done": status["chunks_done"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "run"):
        p = sub.add_parser(name)
        p.add_argument("--packet", type=Path, default=PACKET)
        p.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA",
                       help="no automatic fallback (run_s4_md.py 'auto' may silently drop to CPU)")
        p.add_argument("--device-index", default=None)
        p.add_argument("--dry-run", action="store_true", help="print the plan; no OpenMM import")
    s = sub.choices["setup"]
    s.add_argument("--sample", required=True, help="peptide_name or sample_id")
    s.add_argument("--model-id", default="model01")
    s.add_argument("--seed", type=int, default=20261011, help="solvent/ion placement seed")
    s.add_argument("--force", action="store_true", help="redo setup (replicas built on the old one become inconsistent)")
    r = sub.choices["run"]
    r.add_argument("--run-id", required=True)
    r.add_argument("--prod-ns", type=float, required=True, help="production length; not yet fixed by protocol")
    r.add_argument("--nvt-ps", type=float, default=100.0)
    r.add_argument("--npt-ps", type=float, default=1000.0)
    r.add_argument("--dcd-ps", type=float, default=1.0)
    r.add_argument("--chunk-ps", type=float, default=1000.0, help="checkpoint/DCD chunk length")
    r.add_argument("--eq-restraint-k", type=float, default=0.0,
                   help="kJ/mol/nm^2 on solute heavy atoms during NVT/NPT only; 0 = none")
    a = ap.parse_args()
    {"setup": cmd_setup, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
