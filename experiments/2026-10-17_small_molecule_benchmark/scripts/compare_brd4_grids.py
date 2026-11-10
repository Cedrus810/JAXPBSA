#!/usr/bin/env python3
"""Compare the BRD4 h0.4 rescore with round 1 (default grid); read-only, no computation.

    python experiments/2026-10-17_small_molecule_benchmark/scripts/compare_brd4_grids.py

Plan docs/BRD4_H04_RESCORE_PLAN_20261025.md §1.  Inputs: scores/{default,h0.4}/brd4/.../frames.csv,
reports/brd4_{round1,h0.4}_samples.csv, reference/grid_sensitivity/q3/... (config h0.4),
reports/reference_q3_samples.csv.  Writes reports/brd4_h04_vs_default_20261025.csv and prints:
  1. q3 reproduction: h0.4 rescore minus the h0.4 reference at the same 90 frames (same config, same pilot);
  2. per sample: h0.4 minus default over 500 frames x 3 replicas (ΔG_MMPBSA and ΔG_PB), next to
     the 9-frame q3 estimate (reference_q3_samples tot_h04 - ours_total);
  3. Spearman rank correlation of the 10 sample means, default vs h0.4.
No experimental statistics (plan §5).
"""
from __future__ import annotations

import csv
from pathlib import Path
import statistics as st

P = Path(__file__).resolve().parents[1]
SAMPLES = [f"BRD4-{i}" for i in range(1, 11)]


def frames(grid, s, r):
    return {int(x["frame"]): x for x in csv.DictReader(
        (P / f"scores/{grid}/brd4/{s}/mobley_amber/rep{r:02d}/frames.csv").open())}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0] * len(xs)
    for k, i in enumerate(order):
        rk[i] = k
    return rk


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def main():
    rep_diff = []
    for s in SAMPLES:
        for r in (1, 2, 3):
            h4 = frames("h0.4", s, r)
            for x in csv.DictReader((P / f"reference/grid_sensitivity/q3/brd4/{s}/mobley_amber/rep{r:02d}/sensitivity.csv").open()):
                if x["config"] == "h0.4":
                    y = h4[int(x["frame"])]
                    rep_diff.append(abs(float(y["delta_g_mmpbsa"]) - float(x["delta_g_mmpbsa"])))
    print(f"1. q3 reproduction: {len(rep_diff)} frames, max |h0.4 rescore - h0.4 reference| ΔG_MMPBSA = {max(rep_diff):.2e}")

    d = {r["sample"]: r for r in csv.DictReader((P / "reports/brd4_round1_samples.csv").open())}
    h = {r["sample"]: r for r in csv.DictReader((P / "reports/brd4_h0.4_samples.csv").open())}
    q3 = {r["sample"]: r for r in csv.DictReader((P / "reports/reference_q3_samples.csv").open())}
    rows = []
    for s in SAMPLES:
        dpb = []
        for r in (1, 2, 3):
            a, b = frames("default", s, r), frames("h0.4", s, r)
            dpb.append(st.fmean(float(b[k]["delta_g_pb"]) - float(a[k]["delta_g_pb"]) for k in a))
        dd, hh = float(d[s]["dG_mmpbsa_mean_of_replicas"]), float(h[s]["dG_mmpbsa_mean_of_replicas"])
        rows.append(dict(sample=s, default=dd, h04=hh, h04_minus_default=hh - dd,
                         d_pb_rep_mean=st.fmean(dpb), d_pb_rep_range=max(dpb) - min(dpb),
                         q3_estimate=float(q3[s]["tot_h04"]) - float(q3[s]["ours_total"]),
                         range_default=float(d[s]["replica_range"]), range_h04=float(h[s]["replica_range"])))
    with (P / "reports/brd4_h04_vs_default_20261025.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("2. sample      default    h0.4   h0.4-def  dPB(rep mean, range)  q3 9-frame est  rep range def/h0.4")
    for r in rows:
        print(f"   {r['sample']:8s} {r['default']:8.2f} {r['h04']:8.2f} {r['h04_minus_default']:+8.2f}"
              f"   {r['d_pb_rep_mean']:+6.2f} ({r['d_pb_rep_range']:.2f})      {r['q3_estimate']:+6.2f}"
              f"          {r['range_default']:.2f} / {r['range_h04']:.2f}")
    sp = spearman([r["default"] for r in rows], [r["h04"] for r in rows])
    print(f"3. Spearman default vs h0.4 (10 samples): {sp:.3f}")
    print("   order default:", [r["sample"] for r in sorted(rows, key=lambda r: r["default"])])
    print("   order h0.4:   ", [r["sample"] for r in sorted(rows, key=lambda r: r["h04"])])


if __name__ == "__main__":
    main()
