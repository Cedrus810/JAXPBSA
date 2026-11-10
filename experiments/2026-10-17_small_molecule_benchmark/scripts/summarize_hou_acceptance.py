#!/usr/bin/env python3
"""Hou h0.4 bounded numerical acceptance: summary (read-only; run on 183).

Plan: docs/HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md §1, §6.
  * per frame (s25, 21 replicas): ΔG_PB at default (online copy), h0.4 (scores/h0.4 frames.csv, the same analyzer
    construction as the grid-sensitivity h0.4: §20.5 matched q3 bit for bit), h0.35, and h0.3 for a1/a2/a6;
  * per sample: mean of the paired differences (replica = frame mean, sample = equal-weight replica mean);
    SE from the 15 (replica x 5 time-block) means, each block 5 consecutive s25 frames = 1 ns;
  * adjacent samples in the h0.4 500-frame order: gap at h0.4, gap after adding each sample's s25 mean
    h0.35 - h0.4, and the replica SEs (h0.4 500-frame replica means);
  * Amber (q3, 63 frames): ours h0.4 - pbsa at pbsa h 0.5 / 0.35 / 0.25 per sample and the span over samples.
Writes reports/hou_acceptance_{frames,samples,pairs,amber}_20261028.csv.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
TAG = "20261028"
SAMPLES = [f"a{i}" for i in range(1, 8)]

rows = []
for s in SAMPLES:
    for r in (1, 2, 3):
        rep = f"rep{r:02d}"
        g = pd.read_csv(P / f"reference/grid_sensitivity/s25/hou/{s}/ph74_gaff2/{rep}/sensitivity.csv")
        h4 = pd.read_csv(P / f"scores/h0.4/hou/{s}/ph74_gaff2/{rep}/frames.csv").set_index("frame")
        piv = g.pivot_table(index="frame", columns="config", values="delta_g_pb")
        for f in piv.index:
            rows.append(dict(sample=s, replica=r, frame=int(f), block=int(f // 100) + 1,
                             pb_default=piv.loc[f, "default_round1"], pb_h04=float(h4.loc[f, "delta_g_pb"]),
                             pb_h035=piv.loc[f, "h0.35"],
                             pb_h03=piv.loc[f, "h0.3"] if "h0.3" in piv.columns else np.nan))
fr = pd.DataFrame(rows)
fr["d35_40"] = fr.pb_h035 - fr.pb_h04
fr["d30_35"] = fr.pb_h03 - fr.pb_h035
fr["d40_def"] = fr.pb_h04 - fr.pb_default
if sorted(fr.groupby(["sample", "replica"]).size().unique()) != [25]:
    raise RuntimeError("expected 25 s25 frames per replica")


def mean_se(g, col):
    b = g.groupby(["replica", "block"])[col].mean()
    if b.isna().all():
        return np.nan, np.nan, np.nan
    rep_means = g.groupby("replica")[col].mean()
    return rep_means.mean(), b.std(ddof=1) / np.sqrt(len(b)), g[col].std(ddof=1)


full = {}
for s in SAMPLES:
    v = [pd.read_csv(P / f"scores/h0.4/hou/{s}/ph74_gaff2/rep{r:02d}/frames.csv").delta_g_mmpbsa.mean() for r in (1, 2, 3)]
    full[s] = (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(3)))

samp = []
for s, g in fr.groupby("sample"):
    d1, se1, sd1 = mean_se(g, "d35_40")
    d2, se2, sd2 = mean_se(g, "d30_35")
    d0, se0, _ = mean_se(g, "d40_def")
    samp.append(dict(sample=s, score_h04_500=round(full[s][0], 3), se_rep_h04=round(full[s][1], 3),
                     d40_def_s25=round(d0, 3), se_d40_def=round(se0, 3),
                     d35_40=round(d1, 3), se_d35_40=round(se1, 3), sd_frame_d35_40=round(sd1, 3),
                     d30_35=round(d2, 3) if d2 == d2 else "", se_d30_35=round(se2, 3) if se2 == se2 else "",
                     sd_frame_d30_35=round(sd2, 3) if sd2 == sd2 else ""))
sm = pd.DataFrame(samp).set_index("sample")
sm["score_h035_est"] = (sm.score_h04_500 + sm.d35_40).round(3)

order = sm.sort_values("score_h04_500").index.tolist()
pairs = []
for a, b in zip(order, order[1:]):
    pairs.append(dict(stronger=a, weaker=b, gap_h04=round(sm.score_h04_500[b] - sm.score_h04_500[a], 3),
                      gap_h035_est=round(sm.score_h035_est[b] - sm.score_h035_est[a], 3),
                      d_gap_numerical=round(sm.d35_40[b] - sm.d35_40[a], 3),
                      se_d_gap_numerical=round(float(np.hypot(sm.se_d35_40[a], sm.se_d35_40[b])), 3),
                      se_rep_stronger=sm.se_rep_h04[a], se_rep_weaker=sm.se_rep_h04[b],
                      se_rep_combined=round(float(np.hypot(sm.se_rep_h04[a], sm.se_rep_h04[b])), 3)))

q3 = pd.read_csv(P / "reports/reference_q3_hou_frames.csv")
am = []
for _, x in q3.iterrows():
    row = dict(sample=x["sample"], replica=int(x["replica"]), frame=int(x["frame"]), ours_h04=x["pb_h04"],
               ours_h035=x["pb_h035"], pbsa_0_5=x["amber_EPB"])
    for h in ("0.35", "0.25"):
        c = pd.read_csv(P / f"reference/amber_mmpbsa/q3_pbsa{h}/hou/{x['sample']}/ph74_gaff2/rep{int(x['replica']):02d}/comparison.csv")
        c = c.set_index("frame")
        if abs(c.loc[int(x["frame"]), "ours_dG_pb"] - x["ours_dG_pb"]) > 1e-9:
            raise RuntimeError("q3 default ΔG_PB differs between the two Amber runs")
        row[f"pbsa_{h.replace('.', '_')}"] = c.loc[int(x["frame"]), "amber_EPB"]
    am.append(row)
am = pd.DataFrame(am)
amb = am.groupby("sample")[["ours_h04", "ours_h035", "pbsa_0_5", "pbsa_0_35", "pbsa_0_25"]].mean()
for h in ("0_5", "0_35", "0_25"):
    amb[f"h04_minus_pbsa_{h}"] = amb.ours_h04 - amb[f"pbsa_{h}"]
amb["pbsa_0_25_minus_0_5"] = amb.pbsa_0_25 - amb.pbsa_0_5
span = {c: float(amb[c].max() - amb[c].min()) for c in amb.columns if c.startswith("h04_minus")}

fr.round(4).to_csv(P / f"reports/hou_acceptance_frames_{TAG}.csv", index=False)
sm.reset_index().to_csv(P / f"reports/hou_acceptance_samples_{TAG}.csv", index=False)
pd.DataFrame(pairs).to_csv(P / f"reports/hou_acceptance_pairs_{TAG}.csv", index=False)
amb.round(3).reset_index().to_csv(P / f"reports/hou_acceptance_amber_{TAG}.csv", index=False)

pd.set_option("display.width", 250)
print(sm.to_string())
print(pd.DataFrame(pairs).to_string(index=False))
print(amb.round(2).to_string())
print("span over samples of ours h0.4 - pbsa:", {k: round(v, 2) for k, v in span.items()})
print("pbsa 0.25 - 0.5 per frame: mean %.3f sd %.3f max|.| %.3f" % (
    (am.pbsa_0_25 - am.pbsa_0_5).mean(), (am.pbsa_0_25 - am.pbsa_0_5).std(), (am.pbsa_0_25 - am.pbsa_0_5).abs().max()))
x = fr[fr.frame == 249].merge(q3[["sample", "replica", "frame", "pb_h035"]], on=["sample", "replica", "frame"])
print("frame 249 h0.35 vs q3 reference: max |diff| = %.2e (n=%d)" % ((x.pb_h035_x - x.pb_h035_y).abs().max(), len(x)))
