#!/usr/bin/env python3
"""R4 numerical summary (no experimental labels are read).  r4_protocol.md §2.

All quantities in kcal/mol, paired on the same frame.  What matters for ranking is not a common
offset but how an error differs between samples, so every error is also averaged per sample
(over its replicas x frames) and the spread of these sample means is set beside the adjacent
sample-score gaps and replica SE of the main score (2026-10-11 packet scores_1_20_summary).

  N1 grid     e = X(config) - X(h0.35) for X = delta_g_pb, C/R part (G_C - G_R), ligand part (-G_L)
  N1 rot/shift per frame: SD over the 4 variants (0 = unperturbed on the same grid) and
              variant - variant0
  N1 tol6     tol 1e-6 - tol 1e-5
  N2/N3       ours - Amber per term (EEL, VDW correctness; PB, SA, total method comparison); GB total
  rank        Spearman between sample means of two scorers on the same frames (8 or 7 samples;
              label-free agreement of orderings, not an experimental endpoint)
Writes stats/n1_frame_errors.csv, stats/n1_summary.csv, stats/n1_sample_errors.csv,
stats/n2n3_frames.csv, stats/n2n3_summary.csv, stats/sample_means_methods.csv, stats/summary.json.
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

R4 = Path(__file__).resolve().parents[1]
ROOT = R4.parents[1]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
OUT = R4 / "stats"
F = lambda x: float(x)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def describe(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=round(float(v.mean()), 3), sd=round(float(v.std(ddof=1)), 3) if v.size > 1 else None,
                mae=round(float(np.abs(v).mean()), 3), max_abs=round(float(np.abs(v).max()), 3))


def main():
    OUT.mkdir(exist_ok=True)
    ws = {r["sample"]: r for r in csv.DictReader((PACKET / "scores_1_20_summary/window_samples.csv").open()) if r["window"] == "1-20"}
    order = sorted(ws, key=lambda s: float(ws[s]["score"]))
    gaps = [float(ws[order[i + 1]]["score"]) - float(ws[order[i]]["score"]) for i in range(len(order) - 1)]
    ref = dict(sample_scores={s: float(ws[s]["score"]) for s in order},
               adjacent_gaps=dict(min=round(min(gaps), 3), max=round(max(gaps), 3)),
               replica_se=dict(min=round(min(float(ws[s]["se_rep"]) for s in order), 3),
                               max=round(max(float(ws[s]["se_rep"]) for s in order), 3)),
               score_range=round(float(ws[order[-1]]["score"]) - float(ws[order[0]]["score"]), 3))

    # ---------------- N1
    fe, var_rows = [], []
    for f in sorted(glob.glob(str(R4 / "n1_grid/*/*/rep*/sensitivity.csv"))):
        p = Path(f); s, m, rp = p.parts[-4], p.parts[-3], p.parts[-2]
        rows = list(csv.DictReader(p.open()))
        by = defaultdict(dict)
        for r in rows:
            by[int(r["frame"])][(r["config"], int(r["variant"]))] = r
        for k, d in by.items():
            parts = lambda r: (F(r["delta_g_pb"]), F(r["g_pb_complex"]) - F(r["g_pb_receptor"]), -F(r["g_pb_ligand"]),
                               F(r["delta_g_mmpbsa"]))
            fine = parts(d[("h0.35", 0)])
            row = dict(sample=s, model=m, replica=rp, frame=k)
            for c in ("default", "h0.4"):
                x = parts(d[(c, 0)])
                row.update({f"{c}_err_dGpb": x[0] - fine[0], f"{c}_err_cr": x[1] - fine[1], f"{c}_err_lig": x[2] - fine[2],
                            f"{c}_err_total": x[3] - fine[3]})
            row["tol6_minus_default_dGpb"] = F(d[("tol6", 0)]["delta_g_pb"]) - F(d[("default", 0)]["delta_g_pb"])
            for c in ("rot", "shift"):
                vals = [F(d[(c, j)]["delta_g_pb"]) for j in range(4)]
                row[f"{c}_sd_dGpb"] = float(np.std(vals, ddof=1))
                row[f"{c}_maxdev_dGpb"] = float(max(abs(v - vals[0]) for v in vals[1:]))
                row[f"{c}_v0_minus_default"] = vals[0] - F(d[("default", 0)]["delta_g_pb"])
                for j in (1, 2, 3):
                    var_rows.append(dict(sample=s, model=m, replica=rp, frame=k, config=c, variant=j, d_dGpb=vals[j] - vals[0]))
            fe.append(row)
    keys = [k for k in fe[0] if k not in ("sample", "model", "replica", "frame")]
    with (OUT / "n1_frame_errors.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fe[0])); w.writeheader(); w.writerows(fe)
    n1sum = [dict(quantity=k, **describe([r[k] for r in fe])) for k in keys]
    n1sum += [dict(quantity=f"{c}_variant_minus_v0_dGpb", **describe([r["d_dGpb"] for r in var_rows if r["config"] == c]))
              for c in ("rot", "shift")]
    with (OUT / "n1_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(n1sum[0])); w.writeheader(); w.writerows(n1sum)
    # per-sample error means: replica means first, then the sample mean and replica SE
    samp = []
    for s in sorted({r["sample"] for r in fe}):
        d = dict(sample=s)
        for k in ("default_err_total", "default_err_cr", "default_err_lig", "h0.4_err_total", "rot_sd_dGpb", "shift_sd_dGpb"):
            reps = defaultdict(list)
            for r in fe:
                if r["sample"] == s:
                    reps[(r["model"], r["replica"])].append(r[k])
            rm = [float(np.mean(v)) for v in reps.values()]
            d[k + "_mean"] = round(float(np.mean(rm)), 3)
            d[k + "_se_rep"] = round(float(np.std(rm, ddof=1) / len(rm) ** 0.5), 3)
        samp.append(d)
    with (OUT / "n1_sample_errors.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(samp[0])); w.writeheader(); w.writerows(samp)
    # same-frame sample means: default vs h0.35 (114 frames per sample)
    sm = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(R4 / "n1_grid/*/*/rep*/sensitivity.csv"))):
        p = Path(f); s = p.parts[-4]
        for r in csv.DictReader(p.open()):
            if r["config"] in ("default", "h0.35", "h0.4") and r["variant"] == "0":
                sm[s][r["config"]].append(F(r["delta_g_mmpbsa"]))
    n1_samples = sorted(sm)
    n1_rank = {c: spearman([np.mean(sm[s]["default"]) for s in n1_samples], [np.mean(sm[s][c]) for s in n1_samples])
               for c in ("h0.4", "h0.35")}
    err_means = [d["default_err_total_mean"] for d in samp]

    # ---------------- N2/N3
    am = []
    for f in sorted(glob.glob(str(R4 / "n2n3_amber/*/*/rep*/comparison.csv"))):
        p = Path(f); s, m, rp = p.parts[-4], p.parts[-3], p.parts[-2]
        for r in csv.DictReader(p.open()):
            am.append(dict(sample=s, model=m, replica=rp, frame=int(r["frame"]),
                           d_eel=F(r["ours_e_coul"]) - F(r["amber_EEL"]), d_vdw=F(r["ours_e_lj"]) - F(r["amber_VDW"]),
                           d_pb=F(r["ours_dG_pb"]) - F(r["amber_EPB"]), d_sa=F(r["ours_dG_sa"]) - F(r["amber_ENPOLAR"]),
                           d_total=F(r["ours_dG_mmpbsa"]) - F(r["amber_PB_TOTAL"]),
                           ours_total=F(r["ours_dG_mmpbsa"]), amber_pb_total=F(r["amber_PB_TOTAL"]),
                           amber_gb_total=F(r["amber_GB_TOTAL"]), rel_eel=abs(F(r["ours_e_coul"]) - F(r["amber_EEL"])) / abs(F(r["amber_EEL"]))))
    with (OUT / "n2n3_frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(am[0])); w.writeheader(); w.writerows(am)
    n2sum = [dict(quantity=k, **describe([r[k] for r in am])) for k in ("d_eel", "d_vdw", "d_pb", "d_sa", "d_total", "rel_eel")]
    with (OUT / "n2n3_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(n2sum[0])); w.writeheader(); w.writerows(n2sum)

    # ---------------- per-sample means of each method on its own frames (label-free)
    meth = {}
    def rep_mean(rows, key, s):
        reps = defaultdict(list)
        for r in rows:
            if r["sample"] == s:
                reps[(r["model"], r["replica"])].append(r[key])
        rm = [float(np.mean(v)) for v in reps.values()]
        return float(np.mean(rm)), float(np.std(rm, ddof=1) / len(rm) ** 0.5)
    n4 = {(r["sample"], r["model"]): r for r in csv.DictReader((R4 / "n4_baselines/start_baselines.csv").open())}
    mds = list(csv.DictReader((R4 / "n4_baselines/md_dsasa.csv").open()))
    out = []
    for s in order:
        row = dict(sample=s, main_score=ref["sample_scores"][s], main_se_rep=float(ws[s]["se_rep"]))
        for key, lab in (("ours_total", "ours_amberframes"), ("amber_pb_total", "amber_pb"), ("amber_gb_total", "amber_gb")):
            mu, se = rep_mean(am, key, s); row[lab] = round(mu, 3); row[lab + "_se"] = round(se, 3)
        row["start_score_mean2"] = round(st.mean(F(n4[(s, m)]["delta_g_mmpbsa"]) for m in ("v21m1", "v21m2")), 3)
        row["start_dsasa_mean2"] = round(st.mean(F(n4[(s, m)]["delta_sasa"]) for m in ("v21m1", "v21m2")), 2)
        row["start_contacts_mean2"] = st.mean(int(n4[(s, m)]["contacts_4A_heavy"]) for m in ("v21m1", "v21m2"))
        row["md_dsasa"] = round(st.mean(F(x["delta_sasa_mean"]) for x in mds if x["sample"] == s), 2)
        row["peptide_net_charge"] = n4[(s, "v21m1")]["peptide_net_charge"]
        out.append(row)
    with (OUT / "sample_means_methods.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    main = [r["main_score"] for r in out]
    rank_vs_main = {k: round(spearman(main, [r[k] for r in out]), 3) for k in
                    ("ours_amberframes", "amber_pb", "amber_gb", "start_score_mean2")}
    # delta SASA = complex - receptor - ligand is negative and more negative when more is buried, the same
    # direction as the score (more negative = predicted stronger): compared as is.  Contacts grow with
    # burial, so their negative is compared.
    rank_vs_main.update({"start_dsasa_mean2": round(spearman(main, [r["start_dsasa_mean2"] for r in out]), 3),
                         "md_dsasa": round(spearman(main, [r["md_dsasa"] for r in out]), 3),
                         "start_contacts_mean2 (negated)": round(spearman(main, [-r["start_contacts_mean2"] for r in out]), 3)})
    summary = dict(reference=ref,
                   n1=dict(frames=len(fe), samples=n1_samples,
                           sample_mean_default_err_total=dict(min=min(err_means), max=max(err_means),
                                                              spread=round(max(err_means) - min(err_means), 3)),
                           spearman_samples_default_vs=n1_rank),
                   n2n3=dict(frames=len(am)), rank_vs_main_8samples=rank_vs_main,
                   note="label-free: no experimental values read; Spearman values compare orderings of scorers only")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
