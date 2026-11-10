#!/usr/bin/env python3
"""PB grid sensitivity on fixed reference frames of one small-molecule replica (GPU).

    python scripts/grid_sensitivity_smallmol.py --run-id BRD4-3_mobley_amber_rep01 --frames 124,249,374
    python scripts/grid_sensitivity_smallmol.py --run-id ... --frames 124 --configs h0.4 --dry-run

Plan: docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md §8 (PB numerical sensitivity), handoff §4.6.
Configurations (each a separate OnlineMMPBSA; everything else at the frozen defaults):

  h0.4      C/R h 0.4 Å,  ligand box h 0.2 Å     pilot = the replica's (0, 5] ns window
  h0.35     C/R h 0.35 Å, ligand box h 1/6 Å     pilot = the window
  h0.3      C/R h 0.3 Å,  ligand box h 0.15 Å    pilot = the window; beyond design §8's finest level, added for
            the Hou h0.4 acceptance (docs/HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md)
  rot       default h 0.5 / 0.25; each reference frame evaluated unrotated and after
            --n-rot rigid rotations about its centre (seeded); pilot = window + all rotated
            reference frames, so both orientations are scored on the same grids and the
            paired difference is the orientation effect alone
The default configuration's values are the round-1 frames.csv rows (same frames, same
analyzer construction); they are copied into the output for pairing, not recomputed.
Refined grids are a reference, not an analytic truth (design §8).  Rigid translation is not
tested here: TripletSolver re-centres C/R and L on their own centres of mass, so a translation
of the input is removed by construction.  A sub-grid C/R shift can still be imposed through
TripletSolver.__call__(..., shift=) (jaxpbsa/pb/energy.py); OnlineMMPBSA does not expose it, and
scripts/grid_diag_smallmol.py uses it (correction 2026-10-25, RESULTS §19.5).

Writes <out>/sensitivity.csv (frame, config, rotation, analyzer fields, flag, wall_ms) and
<out>/record.json (sizing per config, warm-up, device).  Configs already in sensitivity.csv
are skipped only when every expected (frame, rotation) is present.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

from smallmol_records import (bind_score_protocol, file_identity, online_rows_by_frame, sensitivity_frames,
                             validate_setup)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/2026-10-17_small_molecule_benchmark"
CONFIGS = {"h0.4": dict(h=0.4, h_lig=0.2), "h0.35": dict(h=0.35, h_lig=1.0 / 6.0), "h0.3": dict(h=0.3, h_lig=0.15),
           "rot": dict()}
FIELDS = ["e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand", "delta_g_pb",
          "sasa_complex", "sasa_receptor", "sasa_ligand", "delta_g_sa", "delta_g_mmpbsa",
          "converged", "margin_A", "margin_lig_A", "sa_ok"]


def rows_by(path: Path, key: str) -> dict:
    return {r[key]: r for r in csv.DictReader(path.open())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--inputs", type=Path, default=PACKAGE / "manifests/brd4_inputs.csv")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--frames", required=True, help="comma-separated 0-based frame indices of solute_dry.dcd")
    ap.add_argument("--configs", default="h0.4,h0.35,rot")
    ap.add_argument("--n-rot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261019, help="rotation seed")
    ap.add_argument("--window-ns", default="0,5", help="pilot window (start, end] in ns, as the round-1 score")
    ap.add_argument("--frame-set", default="q3")
    ap.add_argument("--ours", choices=("scores", "online"), default="scores",
                    help="source of the default_round1 rows: offline scores/default frames.csv, or the "
                         "replica's committed online CSV matched by step (Hou)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <package>/reference/grid_sensitivity/<frame-set>/<target>/<sample>/<model>/<rep>")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    run = rows_by(a.package / "manifests/runs.csv", "run_id")[a.run_id]
    inp = rows_by(a.inputs, "sample")[run["sample_id"]]
    rep = a.package / run["output_directory"]
    setup = rep.parent / "setup"
    frames = [int(x) for x in a.frames.split(",")]
    configs = [c for c in a.configs.split(",") if c]
    if not frames or len(set(frames)) != len(frames) or min(frames) < 0 or a.n_rot < 0:
        ap.error("require distinct nonnegative frames and n-rot >= 0")
    if not configs or len(set(configs)) != len(configs):
        ap.error("require distinct configurations")
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; known {list(CONFIGS)}")
    w0, w1 = (float(x) for x in a.window_ns.split(","))
    scores = a.package / "scores/default" / run["target"] / run["sample_id"] / run["model_id"] / rep.name
    out = a.out or (a.package / "reference/grid_sensitivity" / a.frame_set / run["target"] / run["sample_id"]
                    / run["model_id"] / rep.name)
    plan = dict(run_id=a.run_id, frames=frames, configs={c: CONFIGS[c] for c in configs}, n_rot=a.n_rot,
                seed=a.seed, pilot_window_ns=[w0, w1], out=str(out))
    if a.dry_run:
        print(json.dumps(plan)); return
    srec = validate_setup(setup, inp)

    import numpy as np
    import mdtraj as md
    import openmm as mm
    from openmm import app
    sys.path.insert(0, str(ROOT))
    import jaxpbsa
    from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
    from jaxpbsa.pb import PBParams
    import jax

    n = srec["n_solute_atoms"]
    lig = np.arange(*srec["ligand_local_idx"])
    prm = app.AmberPrmtopFile(inp["prmtop"])
    system = mm.XmlSerializer.deserialize((setup / "system.xml").read_text())
    tr = md.load_dcd(str(rep / "analysis/solute_dry.dcd"), top=md.load_topology(inp["prmtop"]).subset(range(n)))
    xyz = tr.xyz.astype(np.float64) * 10.0
    times = list(csv.DictReader((rep / "analysis/frame_times.csv").open()))
    t_ps = np.array([float(x["time_ps"]) for x in times])
    win = [k for k in range(len(t_ps)) if w0 * 1000 + 1e-6 < t_ps[k] <= w1 * 1000 + 1e-6]
    if any(k not in win for k in frames):
        raise SystemExit("reference frames must lie in the pilot window")
    masses = np.array([system.getParticleMass(i).value_in_unit(mm.unit.dalton) for i in range(n)])

    rng = np.random.default_rng(a.seed)
    rots = []
    for _ in range(a.n_rot):          # uniform random rotations (QR of a Gaussian matrix)
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rots.append(q)

    def rotate(x, R):
        com = (masses[:, None] * x).sum(0) / masses.sum()
        return (x - com) @ R.T + com

    if a.ours == "online":
        ours, ours_src = online_rows_by_frame(rep)
    else:
        ours, ours_src = {int(x["frame"]): x for x in csv.DictReader((scores / "frames.csv").open())}, None
    in_files = dict(prmtop=inp["prmtop"], system=setup / "system.xml", setup_record=setup / "setup_record.json")
    if a.ours == "scores":
        in_files["scores"] = scores / "frames.csv"
    protocol = dict(schema=2, run_id=a.run_id, frames=frames, configs={c: CONFIGS[c] for c in configs},
                    n_rot=a.n_rot, seed=a.seed, pilot_window_ns=[w0, w1],
                    pb_params=vars(PBParams(precond="auto")),
                    precision_bits=np.dtype(jaxpbsa.dtype()).itemsize * 8,
                    inputs={k: file_identity(v) for k, v in in_files.items()},
                    # Not a content digest: strip rewrites solute_dry.dcd on every call (its
                    # header holds the creation time); bind the reference frames by step/time.
                    trajectory=dict(dcd=str((rep / "analysis/solute_dry.dcd").resolve()), n_atoms=int(tr.n_atoms),
                                    reference_frames=[[k, int(times[k]["step"]), float(t_ps[k])] for k in frames]))
    if a.ours == "online":   # paths only; online CSVs of a complete run are read through status.json
        protocol["ours_online_csvs"] = ours_src
    bind_score_protocol(out, protocol, csv_name="sensitivity.csv")
    fcsv = out / "sensitivity.csv"
    expected = {("default_round1", k, 0) for k in frames}
    expected.update((c, k, j) for c in configs for k in frames
                    for j in range(a.n_rot + 1 if c == "rot" else 1))
    have = sensitivity_frames(fcsv, expected, t_ps, required_fields=FIELDS)
    record = json.loads((out / "record.json").read_text()) if (out / "record.json").exists() else {}
    record.update(run_id=a.run_id, frames=[times[k] for k in frames], seed=a.seed, n_rot=a.n_rot,
                  legacy_setup_input_history_unverified="input_contract" not in srec,
                  rotations=[R.round(6).tolist() for R in rots], pilot_window_ns=[w0, w1],
                  host=platform.node(), device=str(jax.devices()[0]), jax=jax.__version__,
                  translation="not tested: TripletSolver re-centres C/R and L on their own COM")
    record.setdefault("configs", {})
    jaxpbsa.enable_compilation_cache()
    new = not fcsv.exists() or fcsv.stat().st_size == 0
    with fcsv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "time_ps", "config", "h", "h_lig", "rotation"] + FIELDS
                           + ["flag", "wall_ms"])
        if new:
            w.writeheader()
        for k in frames:            # also recover an interrupted default-row copy
            if ("default_round1", k, 0) in have:
                continue
            o = ours[k]
            w.writerow(dict(frame=k, time_ps=t_ps[k], config="default_round1", h=0.5, h_lig=0.25, rotation=0,
                            **{f: o[f] for f in FIELDS}, flag=o["flag"], wall_ms=o["wall_ms"]))
            fh.flush()
        for c in configs:
            if all(key in have for key in expected if key[0] == c) and c in record["configs"]:
                print(json.dumps({"skip_config_done": c})); continue
            kw = CONFIGS[c]
            pilot = xyz[win]
            evals = [(k, 0, xyz[k]) for k in frames]
            if c == "rot":
                extra = [(k, j + 1, rotate(xyz[k], R)) for k in frames for j, R in enumerate(rots)]
                evals += extra
                if extra:
                    pilot = np.concatenate([pilot, np.stack([x for _, _, x in extra])])
            t0 = time.perf_counter()
            az = OnlineMMPBSA(system, prm.topology, np.arange(n), lig, xyz[frames[0]], pilot_coords_A=pilot, **kw)
            t_init = time.perf_counter() - t0
            judge = PBSAReporter(az, interval_steps=1, solute_idx=np.arange(n))
            walls = []
            for k, j, x in evals:
                if (c, k, j) in have:
                    continue
                t1 = time.perf_counter()
                r = az(x)
                ms = (time.perf_counter() - t1) * 1000
                walls.append(ms)
                w.writerow(dict(frame=k, time_ps=t_ps[k], config=c, h=kw.get("h", 0.5), h_lig=kw.get("h_lig", 0.25),
                                rotation=j, **{f: r[f] for f in FIELDS}, flag=judge._violation(r) or "",
                                wall_ms=round(ms, 2)))
                fh.flush()
            record["configs"][c] = dict(kwargs=kw, n_pilot=len(pilot), n_evals=len(evals),
                                        sizing={k2: (list(v) if isinstance(v, tuple) else v) for k2, v in az.sizing.items()},
                                        init_incl_warmup_s=round(t_init, 2), warmup_s=round(az.warmup_time_s, 2),
                                        new_evals_this_call=len(walls),
                                        ms_per_eval_median=round(float(np.median(walls)), 2) if walls else None)
            (out / "record.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
            print(json.dumps({"run_id": a.run_id, "config": c, **{k2: record["configs"][c][k2] for k2 in
                              ("n_evals", "init_incl_warmup_s", "ms_per_eval_median")}}), flush=True)
            del az, judge
    (out / "record.json").write_text(json.dumps(record, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
