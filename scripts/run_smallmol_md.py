#!/usr/bin/env python3
"""Solvated multi-replica MD for the small-molecule benchmark (BRD4 / Hou), from Amber inputs.

Plan: docs/BRD4_HOU_EXECUTION_HANDOFF.md §4 (physics, budget) and
docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md §7 (MD) / §11 (layout).  Three subcommands; each
prints a static plan with --dry-run (no OpenMM / mdtraj import):

    # once per (sample, model): load the solvated prmtop/crds as given, minimise
    python scripts/run_smallmol_md.py setup --sample BRD4-3 --platform CUDA
    # one replica = one run_id row of <package>/manifests/runs.csv
    python scripts/run_smallmol_md.py run --run-id BRD4-3_mobley_amber_rep01 --prod-ns 5
    # dry (solute-only, imaged) trajectory + exact time axis + structural diagnostics
    python scripts/run_smallmol_md.py strip --run-id BRD4-3_mobley_amber_rep01

Differences from scripts/run_peptide_md.py (which this follows for seeds/resume/chunks):
  * no solvation or force-field assignment: the system comes from the input prmtop
    (BRD4: MobleyLab/benchmarksets 5233e0a, ff14SB/GAFF/AM1-BCC/TIP3P/JC ions, box as given);
  * receptor/ligand split comes from the inputs manifest (residue + atom range), checked
    against the prmtop, never guessed from residue names;
  * setup is a plain unrestrained minimisation (no peptide omega/chirality restraint);
  * setup refuses a System whose LJ lives outside NonbondedForce (NBFIX/12-6-4 ->
    CustomNonbondedForce), because extract_nonbonded would then read zero LJ for the
    score (design §5: unsupported force representation is an input error);
  * strip writes frame_times.csv: frame k (0-based over all chunks) is the state after
    production step (k+1)*frame_steps, time_ps = step*dt.  Nothing is labelled 10*k.

Physics (handoff §4.4): PME 1.0 nm, HBonds, rigid water, 2 fs, LangevinMiddle 298.15 K
1/ps, MC barostat 1 atm in NPT/production; NVT 100 ps + NPT 1 ns; no production restraints.

Resume: re-run the same command.  Completed stages are skipped; production is written in
chunks (prod/chunk_NNNN.dcd + _state.xml) and a chunk counts only after its state file and
status.json are written.  Raising --prod-ns later extends the run (multiple of --chunk-ps).

Online MM/PBSA (`run --online`, ONLINE_PLAN.md / docs/ONLINE_SCHEDULING.md): the frozen
OnlineMMPBSA scores the solute while MD runs, through AsyncPBSAReporter (same-GPU worker
process, MPS thread limit --mps-percentage, priority 1; 0 = plain async without MPS, the MPS
daemon must already run).  One report per saved frame (--dcd-ps), so the online rows are the
same states as the DCD frames.  The NPT stage stores the solute every --dcd-ps in
npt_solute_A.npy; those 1 ns of frames are the pilot for the grid sizing (pilot_coords_A),
nothing else is passed to the analyzer.  Production contexts get currentStep/time set to the
production step, so CSV step/time_ps follow the frame_times.csv convention.  Each process writes
online/pbsa_from_chunk_NNNN.csv (NNNN = first chunk it ran); after a crash, rows with
step > chunks_done*chunk_steps in the older file belong to a discarded segment, and the newer
file covers the rerun from its own first chunk. A repeated first chunk gets a unique
CSV suffix; committed segment ranges are saved before draining the worker. Completion
requires exact online step coverage in addition to completed MD chunks. Missing online
rows leave MD intact and require offline replay to a separate score directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform as host_platform
import time
import uuid

from smallmol_records import (bind_online_inputs, file_identity, online_rows, row_violation,
                              setup_input_contract, steps_from_ps, validate_md_resume, validate_setup)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/2026-10-17_small_molecule_benchmark"
T_K, DT_PS, FRICTION_PS, P_ATM, CUTOFF_NM = 298.15, 0.002, 1.0, 1.0, 1.0
SEED_STRIDE = 7919  # as run_peptide_md.py; row bases are 1e6 apart (manifests/runs.csv)


def context_seed(base: int, stage: str, chunk: int | None) -> int:
    k = {"nvt": 0, "npt": 1}.get(stage, 2 if chunk is None else 3 + chunk)
    return base + SEED_STRIDE * k


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def input_row(inputs: Path, sample: str) -> dict:
    for r in csv.DictReader(inputs.open()):
        if r["sample"] == sample:
            return r
    raise SystemExit(f"sample {sample} not in {inputs}")


def run_row(package: Path, run_id: str) -> dict:
    for r in csv.DictReader((package / "manifests/runs.csv").open()):
        if r["run_id"] == run_id:
            return r
    raise SystemExit(f"run_id {run_id} not in manifests/runs.csv")


def setup_dir(package: Path, target: str, sample: str, model_id: str) -> Path:
    return package / "md" / target / sample / model_id / "setup"


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
def cmd_setup(a) -> None:
    row = input_row(a.inputs, a.sample)
    setup = setup_dir(a.package, a.target, a.sample, a.model_id)
    lig0, lig1 = int(row["ligand_atom_start_0based"]), int(row["ligand_atom_end_exclusive"])
    n_sol = int(row["n_solute_atoms"])
    plan = dict(stage="setup", sample=a.sample, target=a.target, model_id=a.model_id,
                prmtop=row["prmtop"], coordinates=row["coordinates"], setup=str(setup),
                ligand_atoms=[lig0, lig1], n_solute_atoms=n_sol,
                setup_done=(setup / "setup_record.json").exists())
    if a.dry_run:
        print(json.dumps(plan)); return
    input_contract = setup_input_contract(row)
    if plan["setup_done"] and not a.force:
        saved = validate_setup(setup, row, contract=input_contract)
        print(json.dumps({"skip_setup_done": str(setup),
                          "legacy_setup_input_history_unverified": "input_contract" not in saved})); return

    import numpy as np
    import openmm as mm
    from openmm import app, unit
    t0 = time.monotonic()
    prm = app.AmberPrmtopFile(row["prmtop"])
    crd = app.AmberInpcrdFile(row["coordinates"])
    top = prm.topology
    atoms = list(top.atoms())
    residues = list(top.residues())

    # manifest <-> prmtop consistency (static audit re-checked on the real objects)
    lig_res = residues[int(row["ligand_residue_1based"]) - 1]
    lig_idx = [x.index for x in lig_res.atoms()]
    problems = []
    if top.getNumAtoms() != int(row["n_system_atoms"]):
        problems.append(f"n_atoms {top.getNumAtoms()} != manifest {row['n_system_atoms']}")
    if lig_res.name != row["ligand_residue"]:
        problems.append(f"residue {row['ligand_residue_1based']} is {lig_res.name}, manifest {row['ligand_residue']}")
    if lig_idx != list(range(lig0, lig1)):
        problems.append(f"ligand residue atoms {lig_idx[0]}..{lig_idx[-1]} != manifest [{lig0},{lig1})")
    if lig1 != n_sol:
        problems.append(f"ligand does not end the solute ({lig1} != {n_sol})")
    after = {r.name for r in residues[int(row["ligand_residue_1based"]):]}
    if not after <= {"WAT", "HOH", "Na+", "Cl-", "NA", "CL"}:
        problems.append(f"non-solvent residues after the ligand: {sorted(after)}")
    if any(x.element is None for x in atoms[:n_sol]):
        problems.append("solute atom without element")
    if problems:
        raise RuntimeError(f"{a.sample}: inputs manifest and prmtop disagree: {problems}; nothing written")

    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=CUTOFF_NM * unit.nanometer,
                              constraints=app.HBonds, rigidWater=True, removeCMMotion=True)
    forces = [f.__class__.__name__ for f in system.getForces()]
    if "CustomNonbondedForce" in forces:
        raise RuntimeError(f"{a.sample}: prmtop LJ is not representable by NonbondedForce alone "
                           f"(forces {forces}); extract_nonbonded would score zero LJ; nothing written")
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    q = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                  for i in range(system.getNumParticles())])
    system_xml = mm.XmlSerializer.serialize(system)

    integ = mm.LangevinMiddleIntegrator(T_K * unit.kelvin, FRICTION_PS / unit.picosecond, DT_PS * unit.picoseconds)
    plat, props = make_platform(a.platform, a.device_index)
    sim = app.Simulation(top, system, integ, plat, props)
    if crd.boxVectors is None:
        raise RuntimeError(f"{a.sample}: coordinates have no box")
    sim.context.setPeriodicBoxVectors(*crd.boxVectors)
    sim.context.setPositions(crd.positions)
    x0 = np.asarray(crd.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    t_min = time.monotonic()
    sim.minimizeEnergy(tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=0)
    state = sim.context.getState(getPositions=True, getEnergy=True, getForces=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    fmax = float(np.abs(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).max())
    x1 = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    heavy_lig = [i for i in lig_idx if atoms[i].element.symbol != "H"]
    lig_shift = np.linalg.norm(x1[heavy_lig] - x0[heavy_lig], axis=1)

    setup.mkdir(parents=True, exist_ok=True)
    (setup / "system.xml").write_text(system_xml)
    (setup / "minimized_state.xml").write_text(mm.XmlSerializer.serialize(state))
    with (setup / "minimized.pdb").open("w") as fh:
        app.PDBFile.writeFile(top, state.getPositions(), fh)
    rec_ca = [x.index for x in atoms[:lig0] if x.name == "CA"]
    np.savez(setup / "solute_reference.npz", coords_A=x1[:n_sol], atom_name=np.array([x.name for x in atoms[:n_sol]]),
             element=np.array([x.element.symbol for x in atoms[:n_sol]]),
             ligand_local_idx=np.arange(lig0, lig1), receptor_ca_idx=np.array(rec_ca))
    counts = {}
    for r in residues[int(row["ligand_residue_1based"]):]:
        counts[r.name] = counts.get(r.name, 0) + 1
    elem_lig = {}
    for i in lig_idx:
        elem_lig[atoms[i].element.symbol] = elem_lig.get(atoms[i].element.symbol, 0) + 1
    box = [v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors()]
    write_json(setup / "setup_record.json", dict(
        sample=a.sample, target=a.target, model_id=a.model_id, prmtop=row["prmtop"], coordinates=row["coordinates"],
        input_contract=input_contract,
        n_atoms=top.getNumAtoms(), n_solute_atoms=n_sol, ligand_residue=lig_res.name,
        ligand_local_idx=[lig0, lig1], receptor_local_idx=[0, lig0], ligand_elements=elem_lig,
        solvent_residues=counts, forces=forces,
        charge_e=dict(total=float(q.sum()), solute=float(q[:n_sol].sum()), receptor=float(q[:lig0].sum()),
                      ligand=float(q[lig0:lig1].sum()),
                      solute_off_integer=float(q[:n_sol].sum() - round(q[:n_sol].sum()))),
        box_vectors_nm=[[round(c, 4) for c in v] for v in box], pme_cutoff_nm=CUTOFF_NM,
        constraints="HBonds", rigid_water=True,
        minimisation=dict(tolerance_kJ_mol_nm=10.0, max_iterations="until converged", restraints="none",
                          E_before_kJ_mol=e0, E_after_kJ_mol=e1, max_force_after_kJ_mol_nm=fmax,
                          ligand_heavy_shift_A=dict(max=float(lig_shift.max()), mean=float(lig_shift.mean())),
                          wall_s=round(time.monotonic() - t_min, 2)),
        device=device_info(sim.context), openmm=mm.__version__, host=host_platform.node(),
        wall_s=round(time.monotonic() - t0, 2)))
    print(json.dumps({"setup_done": str(setup), "n_atoms": top.getNumAtoms(), "forces": forces,
                      "solute_charge": float(q[:n_sol].sum()), "E_min_kJ_mol": e1,
                      "wall_s": round(time.monotonic() - t0, 2)}), flush=True)


# ------------------------------------------------------------------ replica
def seed_collisions(package: Path, n_chunks: int) -> list:
    seen, clash = {}, []
    for r in csv.DictReader((package / "manifests/runs.csv").open()):
        used = [("velocity", int(r["velocity_seed"]))]
        for kind in ("integrator", "barostat"):
            base = int(r[f"{kind}_seed"])
            used += [(kind, context_seed(base, st, None)) for st in ("nvt", "npt", "prod")]
            used += [(kind, context_seed(base, "prod", c)) for c in range(1, n_chunks)]
        for kind, s in used:
            if s >= 2**31:
                clash.append((r["run_id"], kind, s, "exceeds int32"))
            if s in seen and seen[s][0] != r["run_id"]:
                clash.append((r["run_id"], kind, s, seen[s]))
            seen.setdefault(s, (r["run_id"], kind))
    return clash


def replica_paths(a, row) -> tuple[Path, Path, dict]:
    out = a.package / row["output_directory"]
    setup = setup_dir(a.package, row["target"], row["sample_id"], row["model_id"])
    if out.parent != setup.parent:
        raise SystemExit(f"output_directory {out} is not next to {setup}")
    return out, setup, input_row(a.inputs, row["sample_id"])


def cmd_run(a) -> None:
    row = run_row(a.package, a.run_id)
    out, setup, irow = replica_paths(a, row)
    if not all(math.isfinite(x) and x > 0 for x in
               (a.dcd_ps, a.chunk_ps, a.prod_ns, a.nvt_ps, a.npt_ps)):
        raise SystemExit("MD durations and save/chunk intervals must be finite and positive")
    frame_steps = steps_from_ps(a.dcd_ps, DT_PS, "--dcd-ps")
    chunk_steps = steps_from_ps(a.chunk_ps, DT_PS, "--chunk-ps")
    prod_steps = steps_from_ps(a.prod_ns * 1000, DT_PS, "--prod-ns")
    nvt_steps = steps_from_ps(a.nvt_ps, DT_PS, "--nvt-ps")
    npt_steps = steps_from_ps(a.npt_ps, DT_PS, "--npt-ps")
    if a.online and npt_steps // frame_steps < 2:
        raise ValueError("online grid sizing requires at least two NPT pilot frames; increase --npt-ps or reduce --dcd-ps")
    if (min(frame_steps, chunk_steps, prod_steps) <= 0 or a.nvt_ps <= 0 or a.npt_ps <= 0
            or chunk_steps % frame_steps or prod_steps % chunk_steps):
        raise SystemExit("--chunk-ps must be a multiple of --dcd-ps and --prod-ns a multiple of --chunk-ps")
    seeds = dict(velocity=int(row["velocity_seed"]), integrator=int(row["integrator_seed"]),
                 barostat=int(row["barostat_seed"]))
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    plan = dict(stage="run", run_id=a.run_id, sample=row["sample_id"], model_id=row["model_id"],
                replica=row["replica"], out=str(out), setup=str(setup),
                setup_done=(setup / "setup_record.json").exists(), seeds=seeds,
                nvt_steps=nvt_steps, npt_steps=npt_steps, prod_steps=prod_steps,
                chunk_steps=chunk_steps, frame_steps=frame_steps, n_chunks=prod_steps // chunk_steps,
                resume_from=None if status is None else {k: status[k] for k in ("nvt_done", "npt_done", "chunks_done")})
    if a.dry_run:
        print(json.dumps({**plan, "seed_collisions": seed_collisions(a.package, max(plan["n_chunks"], 50))})); return
    if not plan["setup_done"]:
        raise SystemExit(f"no setup_record.json in {setup}; run `setup --sample {row['sample_id']}` first")
    srec = validate_setup(setup, irow)

    source_inputs = {k: file_identity(v) for k, v in dict(
        prmtop=irow["prmtop"], coordinates=irow["coordinates"],
        system=setup / "system.xml", minimized=setup / "minimized_state.xml",
        setup_record=setup / "setup_record.json").items()}
    if status is None:
        status = dict(run_id=a.run_id, target=row["target"], sample_id=row["sample_id"], model_id=row["model_id"],
                      replica=int(row["replica"]), setup=str(setup), seeds=seeds, dt_ps=DT_PS,
                      frame_steps=frame_steps, chunk_steps=chunk_steps,
                      nvt_steps=plan["nvt_steps"], npt_steps=plan["npt_steps"],
                      nvt_done=False, npt_done=False, chunks_done=0, prod_target_history=[],
                      contexts=[], stage_wall_s={}, complete=False, source_inputs=source_inputs)
    validate_md_resume(status, row, setup, seeds, source_inputs)
    status["legacy_setup_input_history_unverified"] = "input_contract" not in srec
    dynamics = dict(temperature_K=T_K, dt_ps=DT_PS, friction_per_ps=FRICTION_PS,
                    pressure_atm=P_ATM, cutoff_nm=CUTOFF_NM)
    if status.get("dynamics", dynamics) != dynamics:
        raise RuntimeError("MD dynamics parameters changed; use a new run")
    status["dynamics"] = dynamics
    for key in ("frame_steps", "chunk_steps", "nvt_steps", "npt_steps"):
        if status[key] != plan[key]:
            raise SystemExit(f"{key}={plan[key]} differs from the existing run ({status[key]}); "
                             "use the original value or a new output directory")
    if not status["prod_target_history"] or status["prod_target_history"][-1] != prod_steps:
        status["prod_target_history"].append(prod_steps)
    online_required = status.get("online_required", bool(status.get("online")))
    if a.online and not online_required and status["chunks_done"]:
        raise RuntimeError("cannot add online scoring after offline production has begun; use a new run")
    online_required = online_required or a.online
    status["online_required"] = online_required
    if online_required and npt_steps // frame_steps < 2:
        raise ValueError("online grid sizing requires at least two NPT pilot frames; increase --npt-ps or reduce --dcd-ps")
    status["complete"] = False
    status["md_complete"] = status["chunks_done"] >= plan["n_chunks"]
    status["online_complete"] = False
    (out / "prod").mkdir(parents=True, exist_ok=True)
    write_json(status_path, status)
    if online_required:
        # Check already committed chunks before running any more MD. In
        # particular, a retry after the final worker error must not skip scoring.
        prior_rows = online_rows(status)
        status["online_valid"] = not any(row_violation(x, x["_margin_flag"]) for x in prior_rows)
        if status["chunks_done"] < plan["n_chunks"] and not a.online:
            raise RuntimeError("this run requires --online on continuation")
    if status["chunks_done"] >= plan["n_chunks"]:
        status["md_complete"] = True
        status["online_complete"] = online_required
        status["complete"] = True
        write_json(status_path, status)
        print(json.dumps({"run_id": a.run_id, "complete": True, "skip_md_done": True,
                          "online_complete": status["online_complete"]}))
        return

    pilot_path = out / "npt_solute_A.npy"
    if online_required and status["npt_done"]:
        # Inspect/bind before constructing any production CUDA Context. A
        # changed pilot changes the fixed grids even with identical MD inputs.
        bind_online_inputs(status, online_input_contract(pilot_path, srec, frame_steps))
        write_json(status_path, status)

    import openmm as mm
    from openmm import app, unit

    top = app.AmberPrmtopFile(irow["prmtop"]).topology
    sys_xml = (setup / "system.xml").read_text()
    plat, props = make_platform(a.platform, a.device_index)
    T = T_K * unit.kelvin

    def build(stage: str, chunk: int | None):
        system = mm.XmlSerializer.deserialize(sys_xml)
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

    def run_steps(sim, n, label, collect=None):
        t = time.monotonic()
        sim.reporters = [app.StateDataReporter(str(out / f"{label}.log"), max(1, min(n, 5000)), step=True,
                                               time=True, potentialEnergy=True, temperature=True,
                                               volume=True, density=True, speed=True)]
        if collect is None:
            sim.step(n)
        else:   # solute snapshots every saved-frame interval (online pilot)
            for _ in range(n // frame_steps):
                sim.step(frame_steps)
                st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
                collect.append(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)[:n_sol])
            sim.step(n % frame_steps)
        sim.reporters = []
        status["stage_wall_s"][label] = round(time.monotonic() - t, 2)

    n_sol = int(srec["n_solute_atoms"])

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
        frames = [] if a.online else None
        run_steps(sim, plan["npt_steps"], "npt", collect=frames)
        if a.online:
            import numpy as np
            np.save(pilot_path, np.asarray(frames, dtype=np.float32))
        save(sim, out / "npt_end.xml")
        status["npt_done"] = True; write_json(status_path, status)

    if online_required:
        bind_online_inputs(status, online_input_contract(pilot_path, srec, frame_steps))
        write_json(status_path, status)

    n_chunks = prod_steps // chunk_steps
    sim = None
    reporter = None
    try:
        while status["chunks_done"] < n_chunks:
            c = status["chunks_done"]
            if sim is None:
                sim = build("prod", None if c == 0 else c)
                load(sim, out / "npt_end.xml" if c == 0 else out / "prod" / f"chunk_{c - 1:04d}_state.xml")
                # Production logs and reporters use the committed production
                # clock in both modes, including offline extensions/restarts.
                sim.currentStep = c * chunk_steps
                sim.context.setTime(c * chunk_steps * DT_PS)
                if a.online:
                    reporter = start_online(a, out, c, top, sys_xml, srec, pilot_path, frame_steps)
                    segment = dict(csv=reporter.out_csv_path, from_chunk=c, to_chunk=c,
                                   state="running", n_submitted=0, n_reports=0, n_flagged=0,
                                   n_skipped=0, margin_flag=reporter.margin_min)
                    status.setdefault("online", []).append(segment)
                    write_json(status_path, status)
            t = time.monotonic()
            with (out / "prod" / f"chunk_{c:04d}.dcd").open("wb") as fh:
                dcd = app.DCDFile(fh, top, DT_PS * unit.picoseconds, firstStep=c * chunk_steps, interval=frame_steps)
                sim.reporters = [app.StateDataReporter(str(out / "prod" / f"chunk_{c:04d}.log"), 5000, step=True,
                                                       time=True, potentialEnergy=True, temperature=True,
                                                       volume=True, density=True, speed=True)]
                if reporter is not None:
                    sim.reporters.append(reporter)
                for _ in range(chunk_steps // frame_steps):
                    sim.step(frame_steps)
                    st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
                    dcd.writeModel(st.getPositions(), periodicBoxVectors=st.getPeriodicBoxVectors())
                sim.reporters = []
            save(sim, out / "prod" / f"chunk_{c:04d}_state.xml")
            status["chunks_done"] = c + 1
            status["md_complete"] = c + 1 >= n_chunks
            if reporter is not None:
                segment["to_chunk"] = c + 1
            status["stage_wall_s"][f"prod_chunk_{c:04d}"] = round(time.monotonic() - t, 2)
            write_json(status_path, status)
            print(json.dumps({"run_id": a.run_id, "chunk": c, "of": n_chunks,
                              "wall_s": status["stage_wall_s"][f"prod_chunk_{c:04d}"],
                              **({} if reporter is None else {"pbsa_reports": reporter.n_reports,
                                                              "pbsa_flagged": reporter.n_flagged})}), flush=True)
    finally:
        if reporter is not None:
            finish_online(reporter, segment, status, status_path, a.mps_percentage)
    status["md_complete"] = status["chunks_done"] * chunk_steps >= prod_steps
    if online_required:
        rows = online_rows(status)
        status["online_complete"] = len(rows) == status["chunks_done"] * chunk_steps // frame_steps
        status["online_valid"] = not any(row_violation(x, x["_margin_flag"]) for x in rows)
    status["complete"] = status["md_complete"] and (not online_required or status["online_complete"])
    status["openmm"] = mm.__version__
    status["host"] = host_platform.node()
    write_json(status_path, status)
    print(json.dumps({"run_id": a.run_id, "complete": status["complete"], "chunks_done": status["chunks_done"]}))


def finish_online(reporter, segment, status, status_path, mps_percentage):
    """Persist the accepted range and counters even when worker drain fails."""
    failure = None
    try:
        reporter.close()
    except Exception as exc:
        failure = exc
    segment.update(
        to_chunk=status["chunks_done"], state="failed" if failure else "closed",
        n_submitted=reporter.n_submitted, n_reports=reporter.n_reports,
        n_flagged=reporter.n_flagged, n_skipped=reporter.n_skipped,
        total_pbsa_s=round(reporter.total_pbsa_s, 2), max_latency_s=round(reporter.max_latency_s, 3),
        warmup_time_s=round(reporter.warmup_time_s, 2),
        sizing=json.loads(json.dumps(reporter.sizing, default=str)),
        worker_config=json.loads(json.dumps(reporter.worker_config, default=str)),
        mps_thread_percentage=mps_percentage or None)
    if failure is not None:
        segment["error"] = f"{type(failure).__name__}: {failure}"
    write_json(status_path, status)
    if failure is not None:
        raise failure


def online_input_contract(pilot_path, srec, frame_steps):
    """Actual analyzer defaults and pilot bytes, without creating a device Context."""
    import inspect
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    import jaxpbsa
    from jaxpbsa.online import OnlineMMPBSA
    from jaxpbsa.pb import PBParams

    defaults = {k: p.default for k, p in inspect.signature(OnlineMMPBSA).parameters.items()
                if p.default is not inspect.Parameter.empty and k != "pilot_coords_A"}
    defaults["pb_params"] = vars(PBParams(precond="auto"))
    return dict(schema=1, pilot=file_identity(pilot_path), analyzer=defaults,
                precision_bits=np.dtype(jaxpbsa.dtype()).itemsize * 8,
                n_solute_atoms=srec["n_solute_atoms"], ligand_local_idx=srec["ligand_local_idx"],
                frame_steps=frame_steps)


def start_online(a, out: Path, chunk: int, top, sys_xml: str, srec: dict, pilot_path: Path, frame_steps: int):
    """AsyncPBSAReporter for one production process (analyzer built/compiled in the worker)."""
    import sys
    import numpy as np
    import openmm as mm
    if not pilot_path.exists():
        raise SystemExit(f"{pilot_path} missing: --online needs the NPT stage run with --online (pilot frames)")
    sys.path.insert(0, str(ROOT))
    from jaxpbsa.online_async import AsyncPBSAReporter
    pilot = np.load(pilot_path).astype(np.float64)
    n_sol = int(srec["n_solute_atoms"])
    csv_path = out / "online" / f"pbsa_from_chunk_{chunk:04d}.csv"
    csv_path.parent.mkdir(exist_ok=True)
    if csv_path.exists():
        # A retry at the same chunk must retain the failed session's evidence.
        csv_path = csv_path.with_name(f"{csv_path.stem}_{uuid.uuid4().hex[:12]}.csv")
    mps = dict(mps_thread_percentage=a.mps_percentage, mps_priority=1) if a.mps_percentage else {}
    rep = AsyncPBSAReporter.from_system(
        mm.XmlSerializer.deserialize(sys_xml), top, np.arange(n_sol), np.arange(*srec["ligand_local_idx"]),
        pilot[-1], interval_steps=frame_steps, analyzer_kwargs={"pilot_coords_A": pilot},
        out_csv=str(csv_path), max_pending=2, cpu_threads=a.online_threads, close_timeout=None, **mps)
    rep.out_csv_path, rep.from_chunk = str(csv_path), chunk
    print(json.dumps({"run_id": a.run_id, "online_ready": str(csv_path), "warmup_s": round(rep.warmup_time_s, 1),
                      "pilot_frames": int(pilot.shape[0]), "mps_thread_percentage": a.mps_percentage or None}),
          flush=True)
    return rep


# ------------------------------------------------------------------ strip
def kabsch(P, Q):
    """R, t with R @ p + t ≈ q (rows of P onto rows of Q)."""
    import numpy as np
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def cmd_strip(a) -> None:
    """Write <replica>/analysis/: solute_dry.dcd/.pdb (Å, ligand imaged next to the receptor),
    frame_times.csv (frame, chunk, step, time_ps), diagnostics.csv, diagnostics_summary.json.
    Rebuilt from scratch from all complete chunks each call (cheap next to MD)."""
    row = run_row(a.package, a.run_id)
    out, setup, irow = replica_paths(a, row)
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
    chunks = [] if status is None else [out / "prod" / f"chunk_{c:04d}.dcd" for c in range(status["chunks_done"])]
    plan = dict(stage="strip", run_id=a.run_id, replica=str(out), chunks_complete=len(chunks),
                outputs=str(out / "analysis"), diag_stride=a.diag_stride)
    if a.dry_run:
        print(json.dumps(plan)); return
    if not chunks:
        raise SystemExit(f"no completed production chunks in {out}")

    import numpy as np
    import mdtraj as md
    rec = json.loads((setup / "setup_record.json").read_text())
    ref = np.load(setup / "solute_reference.npz")
    n_sol = rec["n_solute_atoms"]
    lig = np.arange(*rec["ligand_local_idx"])
    top = md.load_topology(irow["prmtop"])
    if [x.name for x in list(top.atoms)[:n_sol]] != list(ref["atom_name"]):
        raise RuntimeError("prmtop atom order differs from setup solute_reference.npz")
    elem = ref["element"]
    rec_ca = ref["receptor_ca_idx"]
    lig_h = np.array([i for i in lig if elem[i] != "H"])
    rec_h = np.array([i for i in range(rec["ligand_local_idx"][0]) if elem[i] != "H"])
    ref_nm = ref["coords_A"] / 10.0
    lig_com_ref = ref_nm[lig_h].mean(0)
    fs, cs = status["frame_steps"], status["chunk_steps"]
    per_chunk = cs // fs

    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    sol_top = top.subset(range(n_sol))
    anchors = [m for m in sol_top.find_molecules() if any(x.index < rec["ligand_local_idx"][0] for x in m)]
    rows, times, frame = [], [], 0
    tmp_dcd = adir / ".solute_dry.dcd.partial"
    dcd_out = md.formats.DCDTrajectoryFile(str(tmp_dcd), "w")
    first = None
    for c, path in enumerate(chunks):
        k_chunk = 0
        for tr in md.iterload(str(path), top=top, chunk=500):
            sl = tr.atom_slice(range(n_sol))
            sl.image_molecules(inplace=True, anchor_molecules=anchors)
            for k in range(sl.n_frames):
                f = frame + k
                step = c * cs + (k_chunk + k + 1) * fs
                times.append(dict(frame=f, chunk=c, step=step, time_ps=round(step * status["dt_ps"], 6)))
                if f % a.diag_stride == 0:
                    X = sl.xyz[k]
                    R, t = kabsch(X[rec_ca], ref_nm[rec_ca])
                    Y = X @ R.T + t
                    rmsd = lambda idx: float(np.sqrt(((Y[idx] - ref_nm[idx]) ** 2).sum(1).mean()) * 10)
                    d = np.linalg.norm(Y[lig_h][:, None] - Y[rec_h][None], axis=2).min() * 10
                    rows.append(dict(frame=f, time_ps=times[-1]["time_ps"], receptor_CA_rmsd_A=rmsd(rec_ca),
                                     ligand_heavy_rmsd_receptor_frame_A=rmsd(lig_h),
                                     ligand_com_shift_A=float(np.linalg.norm(Y[lig_h].mean(0) - lig_com_ref) * 10),
                                     min_lig_rec_heavy_A=float(d)))
            dcd_out.write(sl.xyz * 10, cell_lengths=sl.unitcell_lengths * 10, cell_angles=sl.unitcell_angles)
            if first is None:
                first = sl[0]
            frame += sl.n_frames
            k_chunk += sl.n_frames
        if k_chunk != per_chunk:
            raise RuntimeError(f"{path.name}: {k_chunk} frames, expected {per_chunk}")
    dcd_out.close()
    os.replace(tmp_dcd, adir / "solute_dry.dcd")
    first.save_pdb(str(adir / "solute_dry.pdb"))
    for name, data in (("frame_times.csv", times), ("diagnostics.csv", rows)):
        with (adir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    q = lambda key: {p: round(float(np.percentile([r[key] for r in rows], p)), 3) for p in (5, 50, 95)}
    logs = {}
    for c in range(len(chunks)):
        for ln in (out / "prod" / f"chunk_{c:04d}.log").read_text().splitlines()[1:]:
            parts = ln.split(",")
            if len(parts) >= 6:
                logs.setdefault("T_K", []).append(float(parts[3])); logs.setdefault("density", []).append(float(parts[5]))
    summary = dict(run_id=a.run_id, frames_total=frame, chunks_used=len(chunks), diag_stride=a.diag_stride,
                   n_solute_atoms=n_sol, first_frame=times[0], last_frame=times[-1],
                   quantiles_p5_p50_p95={k: q(k) for k in rows[0] if k not in ("frame", "time_ps")},
                   log_means={k: round(float(np.mean(v)), 4) for k, v in logs.items()},
                   reference="setup minimized structure; ligand RMSD without symmetry correction",
                   note="diagnostic report only; no acceptance thresholds applied")
    (adir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "run", "strip"):
        p = sub.add_parser(name)
        p.add_argument("--package", type=Path, default=PACKAGE)
        p.add_argument("--inputs", type=Path, default=PACKAGE / "manifests/brd4_inputs.csv",
                       help="per-sample prmtop/coordinates/ligand range (brd4_inputs.csv layout)")
        p.add_argument("--dry-run", action="store_true", help="print the plan; no OpenMM/mdtraj import")
        if name != "strip":
            p.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA",
                           help="no automatic fallback")
            p.add_argument("--device-index", default=None)
    s = sub.choices["setup"]
    s.add_argument("--sample", required=True)
    s.add_argument("--target", default="brd4")
    s.add_argument("--model-id", default="mobley_amber")
    s.add_argument("--force", action="store_true", help="redo setup (replicas built on the old one become inconsistent)")
    r = sub.choices["run"]
    r.add_argument("--run-id", required=True)
    r.add_argument("--prod-ns", type=float, required=True)
    r.add_argument("--nvt-ps", type=float, default=100.0)
    r.add_argument("--npt-ps", type=float, default=1000.0)
    r.add_argument("--dcd-ps", type=float, default=10.0, help="design §7: save every 10 ps")
    r.add_argument("--chunk-ps", type=float, default=100.0, help="checkpoint/DCD chunk length")
    r.add_argument("--online", action="store_true", help="score online with AsyncPBSAReporter (see docstring)")
    r.add_argument("--mps-percentage", type=int, default=30,
                   help="MPS thread limit of the PBSA worker (docs/ONLINE_SCHEDULING.md: 30); 0 = no MPS")
    r.add_argument("--online-threads", type=int, default=1, help="CPU threads of the PBSA worker")
    t = sub.choices["strip"]
    t.add_argument("--run-id", required=True)
    t.add_argument("--diag-stride", type=int, default=1, help="frames between diagnostic samples")
    a = ap.parse_args()
    {"setup": cmd_setup, "run": cmd_run, "strip": cmd_strip}[a.cmd](a)


if __name__ == "__main__":
    main()
