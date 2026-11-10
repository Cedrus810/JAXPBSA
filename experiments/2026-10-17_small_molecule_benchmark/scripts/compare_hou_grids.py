#!/usr/bin/env python3
"""Compare the Hou h0.4 rescore with the online round (default grid); read-only, run on 183.

    python experiments/2026-10-17_small_molecule_benchmark/scripts/compare_hou_grids.py

Default values: committed online rows of each replica (smallmol_records.online_rows_by_frame), window
(0, 5] ns, as summarize_hou_online.py.  h0.4: scores/h0.4/hou/<s>/ph74_gaff2/rep<NN>/frames.csv.
Replica score = time mean of delta_g_mmpbsa over the frames valid in both; sample = equal-weight mean of
replicas.  Prints and writes reports/hou_h04_vs_default_20261026.csv:
  1. q3 reproduction (h0.4 rescore vs reference/grid_sensitivity h0.4 at the 63 frames);
  2. per sample: default, h0.4, difference (and its ΔG_PB part), replica range of the difference;
  3. Spearman of the 7 sample means, default vs h0.4.
No experimental statistics.
"""
from __future__ import annotations

import csv
from pathlib import Path
import statistics as st
import sys

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P.parents[1] / "scripts"))
from smallmol_records import online_rows_by_frame, row_violation  # noqa: E402

SAMPLES = [f"a{i}" for i in range(1, 8)]


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb, n = rk(a), rk(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def main():
    rep_max, n_flag, rows = 0.0, [0, 0], []
    for s in SAMPLES:
        per = []
        for r in (1, 2, 3):
            rep = P / f"md/hou/{s}/ph74_gaff2/rep{r:02d}"
            dflt, _ = online_rows_by_frame(rep)
            h4 = {int(x["frame"]): x for x in csv.DictReader((P / f"scores/h0.4/hou/{s}/ph74_gaff2/rep{r:02d}/frames.csv").open())}
            for x in csv.DictReader((P / f"reference/grid_sensitivity/q3/hou/{s}/ph74_gaff2/rep{r:02d}/sensitivity.csv").open()):
                if x["config"] == "h0.4":
                    rep_max = max(rep_max, abs(float(h4[int(x["frame"])]["delta_g_mmpbsa"]) - float(x["delta_g_mmpbsa"])))
            win = [k for k, x in dflt.items() if 0 < float(x["time_ps"]) <= 5000 + 1e-6]
            n_flag[0] += sum(1 for k in win if dflt[k]["flag"])
            n_flag[1] += sum(1 for k in win if k in h4 and h4[k]["flag"])
            ok = [k for k in win if k in h4 and not dflt[k]["flag"] and not h4[k]["flag"]]
            g0 = st.fmean(float(dflt[k]["delta_g_mmpbsa"]) for k in ok)
            g4 = st.fmean(float(h4[k]["delta_g_mmpbsa"]) for k in ok)
            pb = st.fmean(float(h4[k]["delta_g_pb"]) - float(dflt[k]["delta_g_pb"]) for k in ok)
            per.append(dict(n=len(ok), g0=g0, g4=g4, pb=pb))
        d = [x["g4"] - x["g0"] for x in per]
        rows.append(dict(sample=s, n_frames=sum(x["n"] for x in per), default=st.fmean(x["g0"] for x in per),
                         h04=st.fmean(x["g4"] for x in per), h04_minus_default=st.fmean(d),
                         d_pb=st.fmean(x["pb"] for x in per), d_rep_range=max(d) - min(d),
                         range_default=max(x["g0"] for x in per) - min(x["g0"] for x in per),
                         range_h04=max(x["g4"] for x in per) - min(x["g4"] for x in per)))
    print(f"1. q3 reproduction: max |h0.4 rescore - h0.4 reference| ΔG_MMPBSA = {rep_max:.2e}; "
          f"flagged frames default {n_flag[0]}, h0.4 {n_flag[1]}")
    with (P / "reports/hou_h04_vs_default_20261026.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("2. sample  frames  default    h0.4  h0.4-def  dPB   rep-range(diff)  rep range def/h0.4")
    for r in rows:
        print(f"   {r['sample']:4s} {r['n_frames']:6d} {r['default']:8.2f} {r['h04']:8.2f} {r['h04_minus_default']:+7.2f}"
              f" {r['d_pb']:+6.2f}      {r['d_rep_range']:.2f}           {r['range_default']:.2f} / {r['range_h04']:.2f}")
    print(f"3. Spearman default vs h0.4 (7 samples): {spearman([r['default'] for r in rows], [r['h04'] for r in rows]):.3f}")
    print("   order default:", [r["sample"] for r in sorted(rows, key=lambda r: r["default"])])
    print("   order h0.4:   ", [r["sample"] for r in sorted(rows, key=lambda r: r["h04"])])


if __name__ == "__main__":
    main()
