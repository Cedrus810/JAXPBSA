#!/usr/bin/env python3
"""Per-frame MM/PBSA of one small-molecule replica with the frozen JAXPBSA analyzer.

    python scripts/score_smallmol_replica.py --run-id BRD4-3_mobley_amber_rep01 --dry-run
    python scripts/score_smallmol_replica.py --run-id BRD4-3_mobley_amber_rep01 --start-ns 0 --end-ns 5

Plan: docs/BRD4_HOU_EXECUTION_HANDOFF.md §4.6–4.7 (frozen default scoring; first round reports
0–5 ns with five 1-ns diagnostic blocks).  Follows scripts/score_peptide_replica.py, except:
  * the analyzer gets the solvated System (setup/system.xml) and the prmtop topology with
    solute_idx = 0..n_solute-1 and the ligand range from setup_record.json -- no separate
    vacuum system is built; OnlineMMPBSA slices charges/LJ/radii(mbondi2) itself;
  * time comes from <replica>/analysis/frame_times.csv (run_smallmol_md.py strip): the
    window is (start, end] in production time, so frame 0 (t = 10 ps) opens 0–5 ns and
    5000 ps closes it; adjacent windows/blocks share no frame;
  * blocks are --block-ns long from the window start (default 1 ns).
Every method/default parameter is left at the frozen main 3737870 values (h 0.5 / ligand
box h 0.25, PBParams(precond="auto"), mbondi2, INP=1 gamma/beta, SA); the grid is sized from
the scored frames (pilot = the window).  --grid h0.4 changes only the spacings (C/R 0.4 /
ligand box 0.2, as grid_sensitivity_smallmol.py's h0.4) and writes under scores/h0.4/
(docs/BRD4_H04_RESCORE_PLAN_20261025.md); --grid default is unchanged.  --pilot npt replays the online
construction instead (pilot = the replica's npt_solute_A.npy, ref = its last frame, as run_smallmol_md.py
start_online) and writes under scores/replay_npt/<grid>/ (docs/HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md §3).  Flagged frames (PBSAReporter._violation) are written
with the reason, never dropped.  Resumable: frames already in frames.csv are skipped.
New output directories record their input/protocol binding before the first frame.
Historical CSVs without that record are retained but cannot be appended to; use --out.

Writes <out>/frames.csv (frame, step, time_ps, analyzer fields, flag, wall_ms) and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys
import time

from smallmol_records import bind_score_protocol, file_identity, row_violation, scored_frames, validate_setup

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/2026-10-17_small_molecule_benchmark"
FIELDS = ["e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand", "delta_g_pb",
          "sasa_complex", "sasa_receptor", "sasa_ligand", "delta_g_sa", "delta_g_mmpbsa",
          "solver_iters", "converged", "margin_A", "margin_lig_A", "sa_ok"]
# h/h_lig per --grid; "default" must stay equal to the frozen values so its protocol (and the
# resume of existing scores/default directories) is unchanged.
GRIDS = {"default": dict(h=0.5, h_lig=0.25), "h0.4": dict(h=0.4, h_lig=0.2)}


def run_row(package: Path, run_id: str) -> dict:
    for r in csv.DictReader((package / "manifests/runs.csv").open()):
        if r["run_id"] == run_id:
            return r
    raise SystemExit(f"run_id {run_id} not in manifests/runs.csv")


def input_row(inputs: Path, sample: str) -> dict:
    for r in csv.DictReader(inputs.open()):
        if r["sample"] == sample:
            return r
    raise SystemExit(f"sample {sample} not in {inputs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--inputs", type=Path, default=PACKAGE / "manifests/brd4_inputs.csv")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--start-ns", type=float, default=0.0, help="window start, exclusive (production time)")
    ap.add_argument("--end-ns", type=float, default=5.0, help="window end, inclusive (production time)")
    ap.add_argument("--block-ns", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=1, help="score every n-th saved frame of the window")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="default",
                    help="PB grid spacings; default = frozen h 0.5 / ligand 0.25")
    ap.add_argument("--pilot", choices=("window", "npt"), default="window",
                    help="grid sizing pilot: the scored window (default) or the online NPT pilot")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <package>/scores/<grid>/<target>/<sample>/<model>/<rep>; "
                         "--pilot npt: <package>/scores/replay_npt/<grid>/...")
    ap.add_argument("--dry-run", action="store_true", help="print inputs/outputs; no JAX/OpenMM import")
    a = ap.parse_args()
    if (not all(math.isfinite(x) for x in (a.start_ns, a.end_ns, a.block_ns))
            or a.start_ns < 0 or a.end_ns <= a.start_ns or a.block_ns <= 0 or a.stride < 1):
        ap.error("require 0 <= start-ns < end-ns, block-ns > 0 and stride >= 1")

    row = run_row(a.package, a.run_id)
    rep = a.package / row["output_directory"]
    setup = rep.parent / "setup"
    dcd, ftimes = (rep / "analysis" / n for n in ("solute_dry.dcd", "frame_times.csv"))
    sub = Path(a.grid) if a.pilot == "window" else Path("replay_npt") / a.grid
    out = a.out or a.package / "scores" / sub / row["target"] / row["sample_id"] / row["model_id"] / rep.name
    plan = dict(run_id=a.run_id, replica=str(rep), setup=str(setup), dcd=str(dcd), dcd_exists=dcd.exists(),
                window_ns=[a.start_ns, a.end_ns], window_rule="(start, end]", block_ns=a.block_ns,
                stride_frames=a.stride, grid=dict(name=a.grid, **GRIDS[a.grid]), pilot=a.pilot, out=str(out))
    if a.dry_run:
        print(json.dumps(plan)); return
    if not dcd.exists() or not ftimes.exists():
        raise SystemExit(f"{a.run_id}: run `run_smallmol_md.py strip` first ({rep / 'analysis'})")
    srec = validate_setup(setup, input_row(a.inputs, row["sample_id"]))

    import numpy as np
    import mdtraj as md
    import openmm as mm
    from openmm import app
    sys.path.insert(0, str(ROOT))
    import jaxpbsa
    from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
    from jaxpbsa.pb import PBParams
    from jaxpbsa.sa import BETA_INP1, GAMMA_INP1

    n = srec["n_solute_atoms"]
    lig = np.arange(*srec["ligand_local_idx"])
    prm = app.AmberPrmtopFile(input_row(a.inputs, row["sample_id"])["prmtop"])
    system = mm.XmlSerializer.deserialize((setup / "system.xml").read_text())
    if system.getNumParticles() != prm.topology.getNumAtoms():
        raise RuntimeError("system.xml and prmtop atom counts differ")
    # topology from the prmtop, not solute_dry.pdb: mdtraj's PDB reader renames atoms to its
    # standard names, which would break the order/name check against the prmtop
    traj = md.load_dcd(str(dcd), top=md.load_topology(input_row(a.inputs, row["sample_id"])["prmtop"]).subset(range(n)))
    names = [x.name for x in list(prm.topology.atoms())[:n]]
    if traj.n_atoms != n or [x.name for x in traj.topology.atoms] != names:
        raise RuntimeError("solute_dry atom order differs from the prmtop solute")
    times = list(csv.DictReader(ftimes.open()))
    if len(times) != traj.n_frames:
        raise RuntimeError(f"frame_times.csv has {len(times)} rows, solute_dry.dcd {traj.n_frames} frames")
    if ([int(x["frame"]) for x in times] != list(range(traj.n_frames))
            or any(not math.isfinite(float(x["time_ps"])) for x in times)
            or any(int(b["step"]) <= int(a0["step"]) or float(b["time_ps"]) <= float(a0["time_ps"])
                   for a0, b in zip(times, times[1:]))):
        raise RuntimeError("frame_times.csv frame IDs, steps or times are invalid")
    xyz = traj.xyz.astype(np.float64) * 10.0     # mdtraj reads the DCD in nm
    t_ps = np.array([float(x["time_ps"]) for x in times])
    steps = [int(x["step"]) for x in times]
    win = [k for k in range(traj.n_frames) if a.start_ns * 1000 + 1e-6 < t_ps[k] <= a.end_ns * 1000 + 1e-6]
    sel = win[::a.stride]
    if not sel:
        raise SystemExit("no frames in the window")
    if t_ps[win[-1]] < a.end_ns * 1000 - 1e-6:
        print(json.dumps({"warning": f"trajectory ends at {t_ps[win[-1]]} ps, before window end {a.end_ns * 1000} ps"}))

    protocol = dict(
        schema=2, run_id=a.run_id, target=row["target"], sample=row["sample_id"], model=row["model_id"],
        inputs={k: file_identity(v) for k, v in dict(
                system=setup / "system.xml", setup_record=setup / "setup_record.json",
                prmtop=input_row(a.inputs, row["sample_id"])["prmtop"]).items()},
        # Not a content digest: strip rewrites solute_dry.dcd on every call (its header holds
        # the creation time) and an extension appends frames. scored_frames checks each saved
        # frame's step/time against frame_times.csv instead.
        trajectory=dict(dcd=str(dcd.resolve()), frame_times=str(ftimes.resolve()), n_atoms=int(traj.n_atoms)),
        window_ns=[a.start_ns, a.end_ns], stride_frames=a.stride, block_ns=a.block_ns,
        analyzer=dict(**GRIDS[a.grid], pb_params=vars(PBParams(precond="auto")),
                      radii_model="mbondi2", gamma=GAMMA_INP1, beta=BETA_INP1,
                      sa_k=192, sa_points=960, precision_bits=np.dtype(jaxpbsa.dtype()).itemsize * 8),
    )
    if a.pilot == "npt":   # only here, so the protocol of existing window-pilot outputs is unchanged
        npt = rep / "npt_solute_A.npy"
        protocol["pilot"] = dict(source="npt", path=str(npt.resolve()), ref="last pilot frame")
    bind_score_protocol(out, protocol)
    fcsv = out / "frames.csv"
    done = scored_frames(fcsv, {k: (steps[k], t_ps[k]) for k in sel}, required_fields=FIELDS)

    jaxpbsa.enable_compilation_cache()
    t0 = time.perf_counter()
    if a.pilot == "npt":
        pilot = np.load(rep / "npt_solute_A.npy").astype(np.float64)
        if pilot.shape[1:] != (n, 3):
            raise RuntimeError(f"npt_solute_A.npy shape {pilot.shape} != ({n}, 3) per frame")
        ref = pilot[-1]
    else:
        pilot, ref = xyz[sel], xyz[sel[0]]
    az = OnlineMMPBSA(system, prm.topology, np.arange(n), lig, ref, pilot_coords_A=pilot, **GRIDS[a.grid])
    t_init = time.perf_counter() - t0
    judge = PBSAReporter(az, interval_steps=1, solute_idx=np.arange(n))   # only for its frame rule
    import jax
    dev = str(jax.devices()[0])

    new = not fcsv.exists() or fcsv.stat().st_size == 0
    walls = []
    with fcsv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "step", "time_ps"] + FIELDS + ["flag", "wall_ms"])
        if new:
            w.writeheader()
        for k in sel:
            if k in done:
                continue
            t1 = time.perf_counter()
            r = az(xyz[k])
            ms = (time.perf_counter() - t1) * 1000
            walls.append(ms)
            w.writerow(dict(frame=k, step=steps[k], time_ps=t_ps[k],
                            **{f: (json.dumps(r[f]) if f == "solver_iters" else r[f]) for f in FIELDS},
                            flag=judge._violation(r) or "", wall_ms=round(ms, 2)))
            fh.flush()

    rows = sorted(csv.DictReader(fcsv.open()), key=lambda x: int(x["frame"]))
    reasons = [row_violation(x, az.margin_flag) for x in rows]
    valid = [x for x, reason in zip(rows, reasons) if not reason]
    g = np.array([float(x["delta_g_mmpbsa"]) for x in valid])
    tv = np.array([float(x["time_ps"]) for x in valid])
    blocks, b0 = {}, a.start_ns
    while b0 < a.end_ns - 1e-9:
        b1 = min(b0 + a.block_ns, a.end_ns)
        m = (tv > b0 * 1000 + 1e-6) & (tv <= b1 * 1000 + 1e-6)
        blocks[f"{b0:g}-{b1:g}ns"] = dict(mean=float(g[m].mean()) if m.any() else None, n=int(m.sum()))
        b0 += a.block_ns
    summary = dict(
        **plan, target=row["target"], sample=row["sample_id"], model=row["model_id"],
        n_frames_in_window=len(win), n_frames_scored=len(rows), n_valid=len(valid), n_flagged=len(rows) - len(valid),
        flags=sorted({reason for reason in reasons if reason})[:10],
        first_scored=dict(frame=int(rows[0]["frame"]), step=int(rows[0]["step"]), time_ps=float(rows[0]["time_ps"])),
        last_scored=dict(frame=int(rows[-1]["frame"]), step=int(rows[-1]["step"]), time_ps=float(rows[-1]["time_ps"])),
        delta_g_mmpbsa_mean=float(g.mean()) if len(g) else None,
        delta_g_mmpbsa_sd=float(g.std(ddof=1)) if len(g) > 1 else None,
        components_mean={f: float(np.mean([float(x[f]) for x in valid])) for f in
                         ("e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand",
                          "delta_g_pb", "delta_g_sa")} if valid else None,
        block_means=blocks,
        solute_charge_e=srec["charge_e"],
        legacy_setup_input_history_unverified="input_contract" not in srec,
        sizing={k: (list(v) if isinstance(v, tuple) else v) for k, v in az.sizing.items()},
        analyzer_defaults=("OnlineMMPBSA defaults (frozen main 3737870); only pilot_coords_A given"
                           + ("" if a.grid == "default" else f" and h/h_lig of --grid {a.grid}")),
        init_incl_warmup_s=round(t_init, 2), warmup_s=round(az.warmup_time_s, 2),
        ms_per_frame_this_call=dict(median=round(float(np.median(walls)), 2), n=len(walls)) if walls else None,
        device=dev, host=platform.node(), jax=jax.__version__, openmm=mm.__version__,
        units="kcal/mol (analyzer output); Å for margins and areas",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("run_id", "n_frames_scored", "n_valid", "n_flagged",
                                              "delta_g_mmpbsa_mean", "block_means", "ms_per_frame_this_call",
                                              "warmup_s", "device")}, default=float), flush=True)


if __name__ == "__main__":
    main()
