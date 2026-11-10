#!/usr/bin/env python3
"""B1 baseline: MM/PBSA of each sample's minimised start with the frozen JAXPBSA analyzer.

    python scripts/score_smallmol_start.py --target brd4 --samples BRD4-3 --dry-run
    python scripts/score_smallmol_start.py --target hou --samples a1 a2 a3 a4 a5 a6 a7

Plan: docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §4.2 (B1: one start per candidate, same R/L
ranges, frozen model).  The start is md/<target>/<sample>/<model>/setup/solute_reference.npz,
the solute of the unrestrained minimisation written by run_smallmol_md.py setup (the state MD
starts from); it is not re-minimised here.  Analyzer construction is score_smallmol_replica.py's:
setup/system.xml + prmtop topology, solute_idx = 0..n_solute-1, ligand range from setup_record.json,
OnlineMMPBSA defaults; --grid h0.4 changes only the spacings.  The grid is sized from the start
itself: no pilot, fluctuation_allowance = 0, so the scored structure has the default boundary
margin (margin_min) by construction.  OnlineMMPBSA rejects a 1-frame pilot because online frames
fluctuate beyond it; here the start is the only frame ever scored.  Boxes are therefore smaller
than the B2 window/NPT-pilot boxes at the same spacings; sizing is kept in summary.json.  One row per sample in
<out>/frames.csv, labelled frame 0 / step 0 / time_ps 0 (= the minimised start, before NVT);
a single structure has no replica SE.  Flagged rows are kept with the reason.  Resumable: a
sample whose row is already in frames.csv is skipped before JAX/OpenMM are imported.

Writes scores/start_<grid>/<target>/<sample>/<model>/{protocol.json,frames.csv,summary.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

from smallmol_records import bind_score_protocol, file_identity, row_violation, scored_frames, validate_setup
from score_smallmol_replica import FIELDS, GRIDS, PACKAGE, ROOT, input_row

MODELS = {"brd4": ("mobley_amber", "manifests/brd4_inputs.csv"), "hou": ("ph74_gaff2", "manifests/hou_inputs.csv")}
START = {0: (0, 0.0)}   # frame 0 = minimised start: step 0, time 0 ps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--target", choices=sorted(MODELS), required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--model-id", default=None, help="default per target: brd4 mobley_amber, hou ph74_gaff2")
    ap.add_argument("--inputs", type=Path, default=None,
                    help="default <package>/manifests/<target>_inputs.csv (brd4_inputs.csv layout)")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="default",
                    help="PB grid spacings; default = frozen h 0.5 / ligand 0.25")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default <package>/scores/start_<grid>; per sample <out-root>/<target>/<sample>/<model>")
    ap.add_argument("--dry-run", action="store_true", help="print inputs/outputs; no JAX/OpenMM import")
    a = ap.parse_args()

    model, inputs = MODELS[a.target]
    model = a.model_id or model
    inputs = a.inputs.resolve() if a.inputs else a.package / inputs
    out_root = a.out_root or a.package / "scores" / f"start_{a.grid}"
    jobs = []
    for s in a.samples:
        setup = a.package / "md" / a.target / s / model / "setup"
        out = out_root / a.target / s / model
        done = scored_frames(out / "frames.csv", START, required_fields=FIELDS)
        jobs.append(dict(sample=s, setup=str(setup), start=str(setup / "solute_reference.npz"),
                         grid=dict(name=a.grid, **GRIDS[a.grid]), pilot="none (fluctuation_allowance 0)", out=str(out),
                         done=bool(done)))
    if a.dry_run:
        for j in jobs:
            print(json.dumps(j))
        return
    todo = [j for j in jobs if not j["done"]]
    for j in jobs:
        if j["done"]:
            print(json.dumps({"sample": j["sample"], "skip": "start already scored"}), flush=True)
    if not todo:
        return

    import numpy as np
    import openmm as mm
    from openmm import app
    sys.path.insert(0, str(ROOT))
    import jaxpbsa
    from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
    from jaxpbsa.pb import PBParams
    from jaxpbsa.sa import BETA_INP1, GAMMA_INP1
    import jax
    jaxpbsa.enable_compilation_cache()
    dev = str(jax.devices()[0])

    for j in todo:
        s, setup, out = j["sample"], Path(j["setup"]), Path(j["out"])
        row = input_row(inputs, s)
        srec = validate_setup(setup, row)
        n = srec["n_solute_atoms"]
        lig = np.arange(*srec["ligand_local_idx"])
        prm = app.AmberPrmtopFile(row["prmtop"])
        system = mm.XmlSerializer.deserialize((setup / "system.xml").read_text())
        if system.getNumParticles() != prm.topology.getNumAtoms():
            raise RuntimeError(f"{s}: system.xml and prmtop atom counts differ")
        ref = np.load(setup / "solute_reference.npz")
        x = ref["coords_A"].astype(np.float64)
        if x.shape != (n, 3) or list(ref["atom_name"]) != [t.name for t in list(prm.topology.atoms())[:n]]:
            raise RuntimeError(f"{s}: solute_reference.npz shape/atom order differs from the prmtop solute")
        if not np.array_equal(ref["ligand_local_idx"], lig):
            raise RuntimeError(f"{s}: solute_reference.npz ligand range differs from setup_record.json")

        protocol = dict(
            schema=2, kind="minimised_start", target=a.target, sample=s, model=model,
            inputs={k: file_identity(v) for k, v in dict(
                    system=setup / "system.xml", setup_record=setup / "setup_record.json",
                    start=setup / "solute_reference.npz", prmtop=row["prmtop"]).items()},
            start_label=dict(frame=0, step=0, time_ps=0.0, meaning="minimised start, before NVT"),
            analyzer=dict(**GRIDS[a.grid], pb_params=vars(PBParams(precond="auto")),
                          radii_model="mbondi2", gamma=GAMMA_INP1, beta=BETA_INP1,
                          sa_k=192, sa_points=960, precision_bits=np.dtype(jaxpbsa.dtype()).itemsize * 8),
            pilot=dict(source="none: grid fitted to the start", fluctuation_allowance_A=0.0),
        )
        bind_score_protocol(out, protocol)
        t0 = time.perf_counter()
        az = OnlineMMPBSA(system, prm.topology, np.arange(n), lig, x, fluctuation_allowance=0.0, **GRIDS[a.grid])
        t_init = time.perf_counter() - t0
        judge = PBSAReporter(az, interval_steps=1, solute_idx=np.arange(n))   # only for its frame rule
        t1 = time.perf_counter()
        r = az(x)
        ms = (time.perf_counter() - t1) * 1000
        fcsv = out / "frames.csv"
        with fcsv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame", "step", "time_ps"] + FIELDS + ["flag", "wall_ms"])
            w.writeheader()
            w.writerow(dict(frame=0, step=0, time_ps=0.0,
                            **{f: (json.dumps(r[f]) if f == "solver_iters" else r[f]) for f in FIELDS},
                            flag=judge._violation(r) or "", wall_ms=round(ms, 2)))
        saved = next(csv.DictReader(fcsv.open()))
        reason = row_violation(saved, az.margin_flag)
        summary = dict(
            **j, target=a.target, model=model, valid=not reason, flag=reason or None,
            delta_g_mmpbsa=float(saved["delta_g_mmpbsa"]),
            components={f: float(saved[f]) for f in ("e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor",
                                                     "g_pb_ligand", "delta_g_pb", "delta_g_sa")},
            solute_charge_e=srec["charge_e"],
            sizing={k: (list(v) if isinstance(v, tuple) else v) for k, v in az.sizing.items()},
            analyzer_defaults=("OnlineMMPBSA defaults (frozen main 3737870); only fluctuation_allowance=0 given"
                               + ("" if a.grid == "default" else f" and h/h_lig of --grid {a.grid}")),
            init_incl_warmup_s=round(t_init, 2), warmup_s=round(az.warmup_time_s, 2), ms_score=round(ms, 2),
            device=dev, host=platform.node(), jax=jax.__version__, openmm=mm.__version__,
            units="kcal/mol (analyzer output); Å for margins and areas",
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        print(json.dumps({k: summary[k] for k in ("sample", "valid", "flag", "delta_g_mmpbsa", "components",
                                                  "init_incl_warmup_s", "ms_score", "device")},
                         default=float), flush=True)


if __name__ == "__main__":
    main()
