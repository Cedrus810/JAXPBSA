#!/usr/bin/env python3
"""D1 of docs/BRD4_GRID_DIAGNOSIS_PLAN_20261025.md: split the C/R grid error of ΔG_PB (GPU).

    python scripts/grid_diag_smallmol.py --samples BRD4-3,BRD4-6,BRD4-4,BRD4-8
    python scripts/grid_diag_smallmol.py --samples BRD4-3 --replicas 1 --frames 124 --h 0.5,0.35   # smoke

Calls the frozen solver directly (TripletSolver internals, no OnlineMMPBSA, no ligand box, no SA).
One C/R grid per (sample, h), sized like OnlineMMPBSA (padding = reach + 1.0 κ⁻¹ around the
re-centred frames of all its replicas); the grid extent does not change C−R (D0: rot config grid
vs round-1 grid, same frames, |ΔCR| ≤ 0.001).  Per frame and h, rows of <out>/diag.csv:

  calc=full    G_C(q), G_R(q_R)              baseline; at h 0.5 / 0.35 it repeats the round-1 /
                                             h0.35 rows of the grid sensitivity (same frame)
  calc=ronly   G_C(q_R)                      ligand charges zeroed, ligand dielectric kept
  calc=lonly   G_C(q_L)                      receptor charges zeroed
  calc=shift   G_C(q), G_R(q_R) after a      sub-grid phase; the fractional offsets u ~ U[0,1)^3 are
               C/R offset u·h, shift=1..n    seeded by (sample, replica, frame), same u at every h
  calc=tol6    full with PBParams(tol=1e-6)  only at h 0.5 (design §8 tolerance check)

Linear PB with a charge-independent dielectric map (binary SES default) makes G a quadratic form
in q, so G_C(q) = RR_C + LL_C + RL_C with RR_C = G_C(q_R), LL_C = G_C(q_L) and
RL_C = G_C(q) − G_C(q_R) − G_C(q_L).  C − R = (RR_C − G_R) + LL_C + RL_C.
Resumable: rows already in diag.csv (sample, replica, frame, h, calc, shift) are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/2026-10-17_small_molecule_benchmark"
FIELDS = ["sample", "replica", "frame", "time_ps", "h", "calc", "shift", "ux", "uy", "uz",
          "g_c", "g_r", "iters", "relres", "converged", "wall_ms"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--inputs", type=Path, default=PACKAGE / "manifests/brd4_inputs.csv")
    ap.add_argument("--target", default="brd4")
    ap.add_argument("--model", default="mobley_amber")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--replicas", default="1,2,3")
    ap.add_argument("--frames", default="124,249,374", help="q3 frame set (reference/frame_sets/q3.csv)")
    ap.add_argument("--frame-set", default="q3")
    ap.add_argument("--h", default="0.5,0.35")
    ap.add_argument("--n-shift", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default <package>/reference/grid_diag/<frame-set>/<target>; one diag.csv per sample")
    a = ap.parse_args()

    import numpy as np
    import mdtraj as md
    import openmm as mm
    from openmm import app
    sys.path.insert(0, str(ROOT))
    import jaxpbsa
    from jaxpbsa.online import boundary_margin_min
    from jaxpbsa.openmm_io import assign_radii, extract_nonbonded
    from jaxpbsa.pb import PBParams, TripletSolver, recenter_com
    import jax

    jaxpbsa.enable_compilation_cache()
    samples = [s for s in a.samples.split(",") if s]
    reps = [int(x) for x in a.replicas.split(",")]
    frames = [int(x) for x in a.frames.split(",")]
    hs = [float(x) for x in a.h.split(",")]
    runs = list(csv.DictReader((a.package / "manifests/runs.csv").open()))
    inputs = {r["sample"]: r for r in csv.DictReader(a.inputs.open())}
    out_root = a.out_root or (a.package / "reference/grid_diag" / a.frame_set / a.target)
    base = PBParams(precond="auto")             # OnlineMMPBSA default
    tol6 = PBParams(precond="auto", tol=1e-6)

    for si, s in enumerate(samples):
        inp = inputs[s]
        rows = {int(r["replica"]): r for r in runs
                if r["target"] == a.target and r["sample_id"] == s and r["model_id"] == a.model}
        rep_dirs = {k: a.package / rows[k]["output_directory"] for k in reps}
        setup = rep_dirs[reps[0]].parent / "setup"
        srec = json.loads((setup / "setup_record.json").read_text())
        n = srec["n_solute_atoms"]
        lig = np.arange(*srec["ligand_local_idx"])
        rec = np.setdiff1d(np.arange(n), lig)
        system = mm.XmlSerializer.deserialize((setup / "system.xml").read_text())
        prm = app.AmberPrmtopFile(inp["prmtop"])
        q = np.asarray(extract_nonbonded(system).charge)[:n]
        radii = np.asarray(assign_radii(prm.topology, "mbondi2", atom_indices=np.arange(n)))[:n]
        masses = np.array([system.getParticleMass(i).value_in_unit(mm.unit.dalton) for i in range(n)])
        q_r = np.where(np.isin(np.arange(n), rec), q, 0.0)
        q_l = np.where(np.isin(np.arange(n), lig), q, 0.0)
        top = md.load_topology(inp["prmtop"]).subset(range(n))
        evals = []                                # (replica, frame, time_ps, coords Å)
        for k in reps:
            rep = rep_dirs[k]
            tr = md.load_dcd(str(rep / "analysis/solute_dry.dcd"), top=top)
            times = list(csv.DictReader((rep / "analysis/frame_times.csv").open()))
            for f in frames:
                evals.append((k, f, float(times[f]["time_ps"]), tr.xyz[f].astype(np.float64) * 10.0))
        out = out_root / s
        out.mkdir(parents=True, exist_ok=True)
        fcsv = out / "diag.csv"
        have = set()
        if fcsv.exists() and fcsv.stat().st_size:
            have = {(int(r["replica"]), int(r["frame"]), float(r["h"]), r["calc"], int(r["shift"]))
                    for r in csv.DictReader(fcsv.open())}
        record = json.loads((out / "record.json").read_text()) if (out / "record.json").exists() else {}
        record.update(sample=s, replicas=reps, frames=frames, n_shift=a.n_shift, seed=a.seed,
                      host=platform.node(), device=str(jax.devices()[0]), jax=jax.__version__,
                      precision_bits=np.dtype(jaxpbsa.dtype()).itemsize * 8,
                      net_charge=float(q.sum()), q_ligand=float(q[lig].sum()))
        record.setdefault("grids", {})
        new = not fcsv.exists() or fcsv.stat().st_size == 0
        with fcsv.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            ref = np.stack([x for *_, x in evals])
            for h in hs:
                todo = [(k, f) for k, f, _, _ in evals for calc, nsh in
                        [("full", 0), ("ronly", 0), ("lonly", 0)] + [("shift", j) for j in range(1, a.n_shift + 1)]
                        + ([("tol6", 0)] if abs(h - 0.5) < 1e-9 else [])
                        if (k, f, h, calc, nsh) not in have]
                if not todo:
                    print(json.dumps({"sample": s, "h": h, "skip": "done"}), flush=True); continue
                walls = {}
                for tag, params in (("base", base), ("tol6", tol6)):
                    if tag == "tol6" and abs(h - 0.5) >= 1e-9:
                        continue
                    pad = float(radii.max()) + params.probe_radius + params.ion_radius + max(params.swin, 0.0) \
                        + boundary_margin_min(params)
                    tri = TripletSolver(ref, masses, radii, rec, lig, params, h=h, padding=pad, h_lig=None)
                    record["grids"][f"{h}_{tag}"] = dict(shape=list(tri.grid.shape), padding=pad)
                    sv = tri._sv

                    def put(k, f, t, calc, j, u, res, ms):
                        w.writerow(dict(sample=s, replica=k, frame=f, time_ps=t, h=h, calc=calc, shift=j,
                                        ux=u[0], uy=u[1], uz=u[2], g_c=float(res["g_c"]),
                                        g_r=("" if res.get("g_r") is None else float(res["g_r"])),
                                        iters=json.dumps(np.asarray(res["iters"]).tolist()),
                                        relres=json.dumps(np.asarray(res["relres"]).tolist()),
                                        converged=bool(res["converged"]), wall_ms=round(ms, 2)))
                        fh.flush()
                        walls.setdefault(calc, []).append(ms)

                    def pair(c, qq):
                        p = sv.pair(c, qq, rec)
                        return dict(g_c=p["g_pb_complex"], g_r=p["g_pb_receptor"], iters=p["iters"],
                                    relres=p["relres"], converged=p["converged"])

                    def single(c, qq):
                        o = sv(c, qq)
                        return dict(g_c=o["g_pb"], g_r=None, iters=o["iters_solvent"],
                                    relres=o["relres_solvent"], converged=o["converged"])

                    t_first = None
                    for k, f, t, x in evals:
                        c = recenter_com(x, masses, tri.grid.center)
                        rng = np.random.default_rng([a.seed, int(s.split("-")[-1]), k, f])
                        us = rng.random((a.n_shift, 3))
                        if tag == "tol6":
                            jobs = [("tol6", 0, (0, 0, 0), lambda: pair(c, q))]
                        else:
                            jobs = [("full", 0, (0, 0, 0), lambda: pair(c, q)),
                                    ("ronly", 0, (0, 0, 0), lambda: single(c, q_r)),
                                    ("lonly", 0, (0, 0, 0), lambda: single(c, q_l))]
                            jobs += [("shift", j + 1, tuple(us[j]), (lambda cc: (lambda: pair(cc, q)))(c + us[j] * h))
                                     for j in range(a.n_shift)]
                        for calc, j, u, fn in jobs:
                            if (k, f, h, calc, j) in have:
                                continue
                            t1 = time.perf_counter()
                            res = jax.block_until_ready(fn())
                            ms = (time.perf_counter() - t1) * 1000
                            if t_first is None:
                                t_first = ms
                            put(k, f, t, calc, j, u, res, ms)
                    record["grids"][f"{h}_{tag}"].update(
                        first_call_incl_compile_s=None if t_first is None else round(t_first / 1000, 1),
                        ms_median={c2: round(float(np.median(v[1:] if len(v) > 1 else v)), 1)
                                   for c2, v in walls.items()})
                    print(json.dumps({"sample": s, "h": h, "params": tag, **record["grids"][f"{h}_{tag}"]}), flush=True)
                    walls = {}
                    del tri, sv
                (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
