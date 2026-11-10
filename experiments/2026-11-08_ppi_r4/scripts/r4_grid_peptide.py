#!/usr/bin/env python3
"""R4 N1: PB numerical sensitivity on fixed frames of one peptide replica (GPU).

    python experiments/2026-11-08_ppi_r4/scripts/r4_grid_peptide.py --run-id PMI_v21m1_rep01
    python ... --run-id PMI_v21m1_rep01 --frames 100,1900 --configs h0.35 --out <smoke dir>

Protocol: experiments/2026-11-08_ppi_r4/r4_protocol.md §2 (N1a-d).  Analyzer construction is
scripts/score_peptide_replica.py's (system_vacuum.xml, complex_dry.cif topology, ligand_idx of
parameters.npz, pilot = every frame of the 1-20 ns window); only the keyword named per config
changes.  The default row of each frame is copied from the replica's scores_1_20 frames.csv
(same frame, same analyzer), not recomputed.

  h0.4    C/R h 0.4, ligand h 0.2
  h0.35   C/R h 0.35, ligand h 1/6           (finest = reference)
  rot     default grids; 3 seeded rigid rotations about the mass centre (variant 1-3) and the
          unrotated frame (variant 0); pilot = window + the rotated frames, so all share the grids
  shift   default grids; variant 0 = no offset, 1-3 = C/R sub-grid offsets u*h, u ~ U[0,1)^3 seeded by (seed, frame, variant), through
          TripletSolver(..., shift=) (MM and SA do not depend on it; only the PB fields change)
  tol6    default grids, PBParams(precond="auto", tol=1e-6)

Writes <out>/sensitivity.csv (frame, config, variant, h, h_lig, analyzer fields, flag, wall_ms) and
<out>/record.json.  Resumable: rows (config, frame, variant) already present are skipped.
No acceptance threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
R4 = Path(__file__).resolve().parents[1]
FRAME_PS = 10.0
CONFIGS = {"h0.4": dict(h=0.4, h_lig=0.2), "h0.35": dict(h=0.35, h_lig=1.0 / 6.0), "rot": {}, "shift": {}, "tol6": {}}
FIELDS = ["e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand", "delta_g_pb",
          "sasa_complex", "sasa_receptor", "sasa_ligand", "delta_g_sa", "delta_g_mmpbsa",
          "converged", "margin_A", "margin_lig_A", "sa_ok"]
DEFAULT_FRAMES = ",".join(str(k) for k in range(100, 2000, 100))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=PACKET)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--frames", default=DEFAULT_FRAMES, help="0-based solute_dry.dcd frames (frame k = 10k ps)")
    ap.add_argument("--configs", default="h0.4,h0.35,rot,shift,tol6")
    ap.add_argument("--n-var", type=int, default=3, help="rotations / shifts per frame")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--window-ns", default="1,20", help="pilot window [start, end) ns, as scores_1_20")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    row = next(r for r in csv.DictReader((a.packet / "proposed_runs.csv").open()) if r["run_id"] == a.run_id)
    rep = Path(row["output_directory"])
    sample, model = rep.parent.parent.name, rep.parent.name
    prep = a.packet / "prepared" / sample / model
    scores = a.packet / "scores_1_20" / sample / model / rep.name / "frames.csv"
    out = (a.out or R4 / "n1_grid" / sample / model / rep.name).resolve()
    frames = [int(x) for x in a.frames.split(",") if x]
    configs = [c for c in a.configs.split(",") if c]
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}")
    w0, w1 = (float(x) for x in a.window_ns.split(","))
    plan = dict(run_id=a.run_id, frames=frames, config_list=configs, n_var=a.n_var, seed=a.seed,
                pilot_window_ns=[w0, w1], scores=str(scores), out=str(out))
    if a.dry_run:
        print(json.dumps(plan)); return

    import numpy as np
    import mdtraj as md
    import openmm as mm
    from openmm import app
    sys.path.insert(0, str(ROOT))
    import jaxpbsa
    from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
    from jaxpbsa.pb import PBParams
    import jax

    ref = np.load(prep / "parameters.npz")
    n = len(ref["atom_name"])
    cif = app.PDBxFile(str(prep / "complex_dry.cif"))
    system = mm.XmlSerializer.deserialize((prep / "system_vacuum.xml").read_text())
    tr = md.load_dcd(str(rep / "analysis/solute_dry.dcd"), top=str(rep / "analysis/solute_dry.pdb"))
    if [x.name for x in tr.topology.atoms] != list(ref["atom_name"]):
        raise RuntimeError("solute_dry atom order differs from parameters.npz")
    xyz = tr.xyz.astype(np.float64) * 10.0
    t_ps = np.arange(tr.n_frames) * FRAME_PS
    win = [k for k in range(tr.n_frames) if w0 * 1000 - 1e-6 <= t_ps[k] < w1 * 1000 - 1e-6]
    if any(k not in win for k in frames):
        raise SystemExit("frames must lie in the pilot window")
    lig = ref["ligand_idx"]
    masses = np.array([system.getParticleMass(i).value_in_unit(mm.unit.dalton) for i in range(n)])
    ours = {int(x["frame"]): x for x in csv.DictReader(scores.open())}

    rng = np.random.default_rng(a.seed)
    rots = []
    for _ in range(a.n_var):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rots.append(q)

    def rotate(x, R):
        com = (masses[:, None] * x).sum(0) / masses.sum()
        return (x - com) @ R.T + com

    out.mkdir(parents=True, exist_ok=True)
    fcsv = out / "sensitivity.csv"
    have = set()
    if fcsv.exists() and fcsv.stat().st_size:
        have = {(r["config"], int(r["frame"]), int(r["variant"])) for r in csv.DictReader(fcsv.open())}
    record = json.loads((out / "record.json").read_text()) if (out / "record.json").exists() else {}
    record.update(plan, rotations=[R.round(6).tolist() for R in rots], host=platform.node(),
                  device=str(jax.devices()[0]), jax=jax.__version__, prepared=str(prep))
    if not isinstance(record.get("configs"), dict):
        record["configs"] = {}
    jaxpbsa.enable_compilation_cache()
    new = not have and not (fcsv.exists() and fcsv.stat().st_size)
    with fcsv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "time_ps", "config", "variant", "h", "h_lig", "tol",
                                           "ux", "uy", "uz"] + FIELDS + ["flag", "wall_ms"])
        if new:
            w.writeheader()
        for k in frames:
            if ("default", k, 0) not in have:
                o = ours[k]
                w.writerow(dict(frame=k, time_ps=t_ps[k], config="default", variant=0, h=0.5, h_lig=0.25, tol=1e-5,
                                **{f: o[f] for f in FIELDS}, flag=o["flag"], wall_ms=o["wall_ms"]))
        fh.flush()
        for c in configs:
            # variant 0 = the unrotated / unshifted frame on this config's own grids, so each rotation /
            # shift is paired with the same grid (grid_sensitivity_smallmol.py does the same for rot)
            todo = [(k, j) for k in frames for j in range(a.n_var + 1 if c in ("rot", "shift") else 1)]
            todo = [(k, j) for k, j in todo if (c, k, j) not in have]
            if not todo:
                print(json.dumps({"skip_config_done": c})); continue
            kw = dict(CONFIGS[c])
            if c == "tol6":
                kw["pb_params"] = PBParams(precond="auto", tol=1e-6)
            pilot = xyz[win]
            if c == "rot":
                pilot = np.concatenate([pilot, np.stack([rotate(xyz[k], R) for k in frames for R in rots])])
            t0 = time.perf_counter()
            az = OnlineMMPBSA(system, cif.topology, np.arange(n), lig, xyz[frames[0]], pilot_coords_A=pilot, **kw)
            t_init = time.perf_counter() - t0
            judge = PBSAReporter(az, interval_steps=1, solute_idx=np.arange(n))
            h = kw.get("h", 0.5)
            walls = []
            for k, j in todo:
                u = (None, None, None)
                t1 = time.perf_counter()
                if c == "rot":
                    r = az(rotate(xyz[k], rots[j - 1]) if j else xyz[k])
                elif c == "shift":
                    u = np.random.default_rng([a.seed, k, j]).random(3) if j else np.zeros(3)
                    base = ours[k]
                    pb = az.triplet_solver(xyz[k], az._q, shift=u * h)
                    num = lambda v: v in (True, "True") if v in ("True", "False", True, False) else float(v)
                    r = {f: num(base[f]) for f in FIELDS}
                    for f in ("g_pb_complex", "g_pb_receptor", "g_pb_ligand", "delta_g_pb", "margin_A", "margin_lig_A"):
                        r[f] = float(pb[f])
                    r["converged"] = bool(np.asarray(pb["converged"]))
                    r["delta_g_mmpbsa"] = (float(base["e_coul_rl"]) + float(base["e_lj_rl"]) + r["delta_g_pb"]
                                           + float(base["delta_g_sa"]))
                else:
                    r = az(xyz[k])
                ms = (time.perf_counter() - t1) * 1000
                walls.append(ms)
                w.writerow(dict(frame=k, time_ps=t_ps[k], config=c, variant=j, h=h, h_lig=kw.get("h_lig", 0.25),
                                tol=1e-6 if c == "tol6" else 1e-5, ux=u[0], uy=u[1], uz=u[2],
                                **{f: r[f] for f in FIELDS}, flag=judge._violation(r) or "", wall_ms=round(ms, 2)))
                fh.flush()
            record["configs"][c] = dict(kwargs={k2: (str(v) if k2 == "pb_params" else v) for k2, v in kw.items()},
                                        n_pilot=len(pilot), new_evals=len(walls),
                                        sizing={k2: (list(v) if isinstance(v, tuple) else v) for k2, v in az.sizing.items()},
                                        init_incl_warmup_s=round(t_init, 2), warmup_s=round(az.warmup_time_s, 2),
                                        ms_per_eval_median=round(float(np.median(walls)), 2))
            (out / "record.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
            print(json.dumps({"run_id": a.run_id, "config": c, **record["configs"][c]}, default=float), flush=True)
            del az, judge
    (out / "record.json").write_text(json.dumps(record, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
