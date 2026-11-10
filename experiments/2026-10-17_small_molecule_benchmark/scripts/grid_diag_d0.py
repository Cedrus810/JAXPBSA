#!/usr/bin/env python3
"""D0 of docs/BRD4_GRID_DIAGNOSIS_PLAN_20261025.md: read-only split of the q3 grid sensitivity.

    python experiments/2026-10-17_small_molecule_benchmark/scripts/grid_diag_d0.py

Reads reference/grid_sensitivity/q3/brd4/<sample>/mobley_amber/rep*/sensitivity.csv (no new
computation) and writes reports/grid_diag_d0_frames_20261025.csv (one row per frame) and
reports/grid_diag_d0_20261025.csv (one row per sample).  All energies kcal/mol.

Per frame, with CR = g_pb_complex - g_pb_receptor:
  d035_C, d035_R, d035_CR, d035_L   h0.35 minus default_round1 (same frame)
  d04_CR                            h0.4 minus default_round1
  rot0_vs_default_CR                rot config, rotation 0, minus default_round1 (grid-size check)
  rotavg_minus_rot0_CR              mean over rotations 1..n of CR minus rotation 0 (same grid)
Per sample: mean, SD over frames, n frames with d035_CR > 0.
"""
from __future__ import annotations

import csv
from pathlib import Path
import statistics as st

PACKAGE = Path(__file__).resolve().parents[1]
SAMPLES = [f"BRD4-{i}" for i in range(1, 11)]


def main():
    frows, srows = [], []
    for s in SAMPLES:
        per = []
        for rep in ("rep01", "rep02", "rep03"):
            f = PACKAGE / "reference/grid_sensitivity/q3/brd4" / s / "mobley_amber" / rep / "sensitivity.csv"
            v = {}
            for r in csv.DictReader(f.open()):
                c, cr = float(r["g_pb_complex"]), float(r["g_pb_receptor"])
                v[(r["config"], int(r["frame"]), int(r["rotation"]))] = dict(
                    C=c, R=cr, L=float(r["g_pb_ligand"]), CR=c - cr)
            for k in sorted({k for (_, k, _) in v}):
                d, h35, h4 = v[("default_round1", k, 0)], v[("h0.35", k, 0)], v[("h0.4", k, 0)]
                r0 = v[("rot", k, 0)]
                rots = [v[key]["CR"] for key in v if key[0] == "rot" and key[1] == k and key[2] > 0]
                row = dict(sample=s, replica=rep, frame=k,
                           d035_C=h35["C"] - d["C"], d035_R=h35["R"] - d["R"], d035_CR=h35["CR"] - d["CR"],
                           d035_L=h35["L"] - d["L"], d04_CR=h4["CR"] - d["CR"],
                           rot0_vs_default_CR=r0["CR"] - d["CR"],
                           rotavg_minus_rot0_CR=st.fmean(rots) - r0["CR"], n_rot=len(rots))
                per.append(row)
        frows += per
        out = dict(sample=s, n_frames=len(per))
        for q in ("d035_C", "d035_R", "d035_CR", "d035_L", "d04_CR", "rot0_vs_default_CR", "rotavg_minus_rot0_CR"):
            xs = [r[q] for r in per]
            out[q + "_mean"], out[q + "_sd"] = st.fmean(xs), st.stdev(xs)
        out["n_d035_CR_pos"] = sum(r["d035_CR"] > 0 for r in per)
        srows.append(out)
    rep = PACKAGE / "reports"
    for name, rows in (("grid_diag_d0_frames_20261025.csv", frows), ("grid_diag_d0_20261025.csv", srows)):
        with (rep / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
    for r in srows:
        print(f"{r['sample']:8s} C {r['d035_C_mean']:+7.2f}±{r['d035_C_sd']:.2f}  R {r['d035_R_mean']:+7.2f}±{r['d035_R_sd']:.2f}"
              f"  C-R {r['d035_CR_mean']:+5.2f}±{r['d035_CR_sd']:.2f} ({r['n_d035_CR_pos']}/9 >0)"
              f"  L {r['d035_L_mean']:+5.2f}  h0.4 C-R {r['d04_CR_mean']:+5.2f}"
              f"  rot0-def {r['rot0_vs_default_CR_mean']:+5.2f}±{r['rot0_vs_default_CR_sd']:.2f}"
              f"  rotavg-rot0 {r['rotavg_minus_rot0_CR_mean']:+5.2f}±{r['rotavg_minus_rot0_CR_sd']:.2f}")


if __name__ == "__main__":
    main()
