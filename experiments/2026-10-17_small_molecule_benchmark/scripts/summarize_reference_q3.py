#!/usr/bin/env python3
"""Summarise the BRD4 same-frame reference (frame set q3) -- read-only, stdlib only.

Reads reference/amber_mmpbsa/q3/.../comparison.csv and reference/grid_sensitivity/q3/.../
sensitivity.csv for all BRD4 replicas and writes:
  reports/reference_q3_frames.csv    one row per frame: ours vs Amber terms and differences,
                                     grid configs and rotations (dG_PB and dG_MMPBSA)
  reports/reference_q3_samples.csv   per sample, mean over its 9 frames, for each method/config
and prints a short text summary (difference statistics, sample-order agreement).
Sample means here are over the 9 reference frames, not the 500-frame round-1 means.
No thresholds are applied.
Optional argument "hou": the 7 Hou samples a1-a7 (model ph74_gaff2, ours = online values),
written to reports/reference_q3_hou_{frames,samples}.csv; default is BRD4 as before.
"""
import csv
import statistics as st
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
TARGET = sys.argv[1] if len(sys.argv) > 1 else "brd4"
SAMPLES, MODEL, TAG = {"brd4": ([f"BRD4-{i}" for i in range(1, 11)], "mobley_amber", "reference_q3"),
                       "hou": ([f"a{i}" for i in range(1, 8)], "ph74_gaff2", "reference_q3_hou")}[TARGET]


def f(x):
    return float(x) if x not in ("", None) else None


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


rows = []
for s in SAMPLES:
    for r in (1, 2, 3):
        sub = f"{TARGET}/{s}/{MODEL}/rep{r:02d}"
        cmp_ = list(csv.DictReader(open(P / "reference/amber_mmpbsa/q3" / sub / "comparison.csv")))
        sens = list(csv.DictReader(open(P / "reference/grid_sensitivity/q3" / sub / "sensitivity.csv")))
        for c in cmp_:
            k = c["frame"]
            g = {(x["config"], int(x["rotation"])): x for x in sens if x["frame"] == k}
            d = g[("default_round1", 0)]
            rot = [f(g[("rot", j)]["delta_g_pb"]) for j in range(4) if ("rot", j) in g]
            row = dict(sample=s, replica=r, frame=int(k), time_ps=f(c["time_ps"]),
                       ours_e_coul=f(c["ours_e_coul"]), amber_EEL=f(c["amber_EEL"]),
                       ours_e_lj=f(c["ours_e_lj"]), amber_VDW=f(c["amber_VDW"]),
                       ours_dG_pb=f(c["ours_dG_pb"]), amber_EPB=f(c["amber_EPB"]),
                       ours_dG_sa=f(c["ours_dG_sa"]), amber_ENPOLAR=f(c["amber_ENPOLAR"]),
                       ours_total=f(c["ours_dG_mmpbsa"]), amber_PB_total=f(c["amber_PB_TOTAL"]),
                       amber_GB_total=f(c["amber_GB_TOTAL"]), amber_EGB=f(c["amber_EGB"]),
                       pb_h04=f(g[("h0.4", 0)]["delta_g_pb"]), pb_h035=f(g[("h0.35", 0)]["delta_g_pb"]),
                       pb_rot_unrot=rot[0], pb_rot_min=min(rot), pb_rot_max=max(rot),
                       tot_h04=f(g[("h0.4", 0)]["delta_g_mmpbsa"]), tot_h035=f(g[("h0.35", 0)]["delta_g_mmpbsa"]),
                       flags=";".join(sorted({x["flag"] for x in sens if x["frame"] == k and x["flag"]})))
            row.update(d_eel=row["ours_e_coul"] - row["amber_EEL"], d_vdw=row["ours_e_lj"] - row["amber_VDW"],
                       d_pb_amber=row["ours_dG_pb"] - row["amber_EPB"], d_sa=row["ours_dG_sa"] - row["amber_ENPOLAR"],
                       d_total_amber=row["ours_total"] - row["amber_PB_total"],
                       d_pb_h04=row["pb_h04"] - row["ours_dG_pb"], d_pb_h035=row["pb_h035"] - row["ours_dG_pb"],
                       rot_span=row["pb_rot_max"] - row["pb_rot_min"],
                       rot_reproduce=row["pb_rot_unrot"] - row["ours_dG_pb"])
            rows.append(row)
(P / "reports").mkdir(exist_ok=True)
with open(P / f"reports/{TAG}_frames.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

cols = ["ours_total", "amber_PB_total", "amber_GB_total", "tot_h04", "tot_h035", "ours_dG_pb", "amber_EPB", "pb_h04", "pb_h035"]
samples = []
for s in SAMPLES:
    sr = [x for x in rows if x["sample"] == s]
    samples.append(dict(sample=s, n_frames=len(sr), **{c: round(st.mean(x[c] for x in sr), 3) for c in cols},
                        rot_span_mean=round(st.mean(x["rot_span"] for x in sr), 3)))
with open(P / f"reports/{TAG}_samples.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(samples[0])); w.writeheader(); w.writerows(samples)


def desc(key):
    v = [x[key] for x in rows]
    return f"mean {st.mean(v):+.3f} sd {st.stdev(v):.3f} min {min(v):+.3f} max {max(v):+.3f}"


print(f"frames {len(rows)}; flagged frames in any config: {sum(1 for x in rows if x['flags'])}")
for k, lab in (("d_eel", "E_coul ours-Amber"), ("d_vdw", "E_LJ ours-Amber"), ("d_pb_amber", "dG_PB ours-pbsa"),
               ("d_sa", "dG_SA ours-LCPO"), ("d_total_amber", "total ours-AmberPB"), ("d_pb_h04", "dG_PB h0.4-default"),
               ("d_pb_h035", "dG_PB h0.35-default"), ("rot_span", "dG_PB rotation span (4 orient.)"),
               ("rot_reproduce", "dG_PB fresh-analyzer reproduce")):
    print(f"{lab:34s} {desc(k)}")
print("relative dG_PB ours vs pbsa: mean %+.2f%%" % (100 * st.mean(x["d_pb_amber"] / x["amber_EPB"] for x in rows)))
base = [x["ours_total"] for x in samples]
for c in ("amber_PB_total", "amber_GB_total", "tot_h04", "tot_h035"):
    print(f"sample order ({len(samples)} samples, 9-frame means): Spearman ours-default vs {c}: {spearman(base, [x[c] for x in samples]):.3f}")
print("sample-mean spread (max-min) ours default: %.2f; per-sample |h0.35-default| mean %.2f" %
      (max(base) - min(base), st.mean(abs(x["tot_h035"] - x["ours_total"]) for x in samples)))
for c in ("ours_dG_pb", "pb_h04", "pb_h035"):
    d = [x[c] - x["amber_EPB"] for x in samples]
    print(f"per-sample {c} - pbsa: {min(d):+.2f} .. {max(d):+.2f} (spread {max(d) - min(d):.2f})")
for x in samples:
    print(x)
