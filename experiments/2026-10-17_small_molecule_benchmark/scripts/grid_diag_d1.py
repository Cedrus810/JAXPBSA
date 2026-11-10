#!/usr/bin/env python3
"""Summarise D1 (scripts/grid_diag_smallmol.py) of docs/BRD4_GRID_DIAGNOSIS_PLAN_20261025.md.

    python experiments/2026-10-17_small_molecule_benchmark/scripts/grid_diag_d1.py [--root DIR] [--tag 20261025]

Per frame and h (kcal/mol):  CR = G_C − G_R (unshifted);  RRmR = G_C(q_R) − G_R;  LL = G_C(q_L);
RL = G_C(q) − G_C(q_R) − G_C(q_L);  CRsh = mean C−R over the shifts;  phase = CRsh − CR.
Per frame, d_* = value at h_fine minus value at 0.5 (same frame).  Per sample: mean and SE over frames.
Writes reports/grid_diag_d1_frames_<tag>.csv and reports/grid_diag_d1_<tag>.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics as st

PACKAGE = Path(__file__).resolve().parents[1]
Q = ["CR", "RRmR", "LL", "RL", "CRsh", "phase", "shift_sd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PACKAGE / "reference/grid_diag/q3/brd4")
    ap.add_argument("--tag", default="20261025")
    ap.add_argument("--h-fine", type=float, default=0.35)
    a = ap.parse_args()
    frows, srows = [], []
    for f in sorted(a.root.glob("*/diag.csv")):
        v = {}
        for r in csv.DictReader(f.open()):
            v.setdefault((r["sample"], int(r["replica"]), int(r["frame"]), float(r["h"])), []).append(r)
        per = {}
        for (s, k, fr, h), rs in v.items():
            by = {(r["calc"], int(r["shift"])): r for r in rs}
            if not all(c in by for c in [("full", 0), ("ronly", 0), ("lonly", 0)]):
                continue
            gc, gr = float(by[("full", 0)]["g_c"]), float(by[("full", 0)]["g_r"])
            rr, ll = float(by[("ronly", 0)]["g_c"]), float(by[("lonly", 0)]["g_c"])
            sh = [float(r["g_c"]) - float(r["g_r"]) for (c, j), r in by.items() if c == "shift"]
            x = dict(CR=gc - gr, RRmR=rr - gr, LL=ll, RL=gc - rr - ll,
                     CRsh=st.fmean(sh) if sh else float("nan"), shift_sd=st.stdev(sh) if len(sh) > 1 else float("nan"))
            x["phase"] = x["CRsh"] - x["CR"]
            if ("tol6", 0) in by:
                x["tol6_dCR"] = float(by[("tol6", 0)]["g_c"]) - float(by[("tol6", 0)]["g_r"]) - x["CR"]
            per[(s, k, fr, h)] = x
        for (s, k, fr, h), x in sorted(per.items()):
            if abs(h - 0.5) > 1e-9 or (s, k, fr, a.h_fine) not in per:
                continue
            y = per[(s, k, fr, a.h_fine)]
            row = dict(sample=s, replica=k, frame=fr, **{f"{q}_05": x[q] for q in Q},
                       **{f"{q}_fine": y[q] for q in Q}, **{f"d_{q}": y[q] - x[q] for q in ("CR", "RRmR", "LL", "RL", "CRsh")},
                       tol6_dCR=x.get("tol6_dCR", float("nan")))
            frows.append(row)
    for s in sorted({r["sample"] for r in frows}, key=lambda t: int(t.split("-")[-1])):
        rs = [r for r in frows if r["sample"] == s]
        out = dict(sample=s, n_frames=len(rs))
        for q in ["d_CR", "d_RRmR", "d_LL", "d_RL", "d_CRsh", "phase_05", "phase_fine", "shift_sd_05", "shift_sd_fine", "tol6_dCR"]:
            xs = [r[q] for r in rs]
            out[q] = st.fmean(xs)
            out[q + "_se"] = st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
        srows.append(out)
    rep = PACKAGE / "reports"
    for name, rows in ((f"grid_diag_d1_frames_{a.tag}.csv", frows), (f"grid_diag_d1_{a.tag}.csv", srows)):
        with (rep / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
    for r in srows:
        print(f"{r['sample']:8s} n={r['n_frames']}  dCR {r['d_CR']:+.2f}±{r['d_CR_se']:.2f} = RR-R {r['d_RRmR']:+.2f}±{r['d_RRmR_se']:.2f}"
              f" + LL {r['d_LL']:+.2f}±{r['d_LL_se']:.2f} + RL {r['d_RL']:+.2f}±{r['d_RL_se']:.2f}"
              f" | phase-avg dCR {r['d_CRsh']:+.2f}±{r['d_CRsh_se']:.2f}"
              f" | shift-mean - unshifted: h0.5 {r['phase_05']:+.2f}±{r['phase_05_se']:.2f}, fine {r['phase_fine']:+.2f}"
              f" | shift SD h0.5 {r['shift_sd_05']:.2f}, fine {r['shift_sd_fine']:.2f} | tol6 {r['tol6_dCR']:+.4f}")


if __name__ == "__main__":
    main()
