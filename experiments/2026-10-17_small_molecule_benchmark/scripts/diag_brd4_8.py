#!/usr/bin/env python3
"""BRD4-8 vs the other 15 C same-protocol ITC entries: score components and pose; read-only.

    python3 experiments/2026-10-17_small_molecule_benchmark/scripts/diag_brd4_8.py

docs/P1_P2_SESSION_HANDOFF_20261101.md §6 (BRD4-8 scored weaker than experiment in every grid and
resample).  Per sample: mean over 3 replicas x 500 frames of E_coul, E_LJ, dG_PB, dG_SA and their
sums (elec = E_coul + dG_PB), default and h0.4 grids, plus pose diagnostics from
reports/brd4_round1_replicas.csv.  BRD4-5 from the alprazolam package.  Descriptive only: a
component difference is not a cause.  Writes reports/brd4_statistics_20261101_v2/brd4_8_components.csv.
"""
import csv
import statistics as st
from pathlib import Path

P = Path(__file__).resolve().parents[1]
SAMPLES = ["BRD4-5", "BRD4-6", "BRD4-7", "BRD4-8", "BRD4-9"]
EXP = {"BRD4-5": -7.40, "BRD4-6": -7.84, "BRD4-7": -8.16, "BRD4-8": -8.99, "BRD4-9": -9.64}
F = ["e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa"]


def rep_dir(grid, s, r):
    if s == "BRD4-5":
        return P / f"brd4_5_alprazolam_20261101/scores/{grid}/brd4/{s}/alprazolam_gaff_fix/rep{r:02d}"
    return P / f"scores/{grid}/brd4/{s}/mobley_amber/rep{r:02d}"


pose = {(x["sample"], int(x["replica"])): x for x in csv.DictReader((P / "reports/brd4_round1_replicas.csv").open())}
out = []
for grid in ("default", "h0.4"):
    for s in SAMPLES:
        reps = []
        for r in (1, 2, 3):
            rows = list(csv.DictReader((rep_dir(grid, s, r) / "frames.csv").open()))
            reps.append({f: st.mean(float(x[f]) for x in rows) for f in F})
        m = {f: st.mean(x[f] for x in reps) for f in F}
        row = dict(grid=grid, sample=s, exp_dg=EXP[s], **{f: round(m[f], 3) for f in F},
                   elec=round(m["e_coul_rl"] + m["delta_g_pb"], 3))
        if grid == "default" and s != "BRD4-5":
            row.update(lig_rmsd_p50_A=" ".join(pose[(s, r)]["lig_rmsd_p50_A"] for r in (1, 2, 3)),
                       lig_rmsd_p95_A=" ".join(pose[(s, r)]["lig_rmsd_p95_A"] for r in (1, 2, 3)))
        out.append(row)
dst = P / "reports/brd4_statistics_20261101_v2/brd4_8_components.csv"
with dst.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in out for k in r)))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r)
