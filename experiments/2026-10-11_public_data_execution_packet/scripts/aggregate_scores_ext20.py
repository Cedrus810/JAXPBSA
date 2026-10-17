#!/usr/bin/env python3
"""Score diagnostics after extending the 48 v2.1 replicas to 20 ns (design §4, 2026-10-16).

    python scripts/aggregate_scores_ext20.py                     # main: scores_1_20/ -> scores_1_20_summary/
    python scripts/aggregate_scores_ext20.py --source split      # superseded: scores/ + scores_10_20/ -> scores_ext20/

Main source (2026-10-17): scores_1_20/, one grid sizing per replica from all 1900 frames of
1-20 ns.  "split" joins scores/ (1-10 ns) and scores_10_20/ (10-20 ns), which were sized
separately (39/48 replicas differ) -- kept only to reproduce the 2026-10-16 numbers.
Valid frames only; no experimental label is read.  Writes into the output directory:
    block_means.csv     replica x 3-ns block (1-4 ... 16-19 ns) mean delta_g_mmpbsa
    window_samples.csv  per window (1-10, 10-20, 1-20 ns) and sample: mean of the 6 replica
                        means, SD/SE over replicas, per-start means
    structure.csv       per replica, 0-10 vs 10-20 ns from analysis/diagnostics.csv: stereo
                        class changes, junction >4 sigma frames, peptide CA RMSD median,
                        min receptor-peptide heavy distance p5
    summary.json        block trend over 48 replicas, variance split and rank stability per
                        window, Spearman between windows
    sizing_paired.csv   (main only) per replica: mean per-frame difference single sizing minus
                        split sizing, for delta_g_mmpbsa and delta_g_pb, in 1-10 and 10-20 ns
Statistical unit: replica mean (protocol §2.1).  Frames are never treated as independent.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from aggregate_scores import DRIFT, MODELS, REPS, SAMPLES, sokal_tau, spearman

P = Path(__file__).resolve().parents[1]
BLOCKS = [(b, b + 3.0) for b in (1.0, 4.0, 7.0, 10.0, 13.0, 16.0)]      # full 3-ns blocks in 1-19 ns
WINDOWS = {"1-10": (1.0, 10.0), "10-20": (10.0, 20.0), "1-20": (1.0, 20.0)}
N_BOOT = 20000


def read(root, s, m, r, field="delta_g_mmpbsa"):
    """{frame: value} over valid frames."""
    return {int(x["frame"]): float(x[field]) for x in csv.DictReader((P / root / s / m / r / "frames.csv").open())
            if not x["flag"]}


def frames(s, m, r, roots):
    t, g = [], []
    for root in roots:
        for x in csv.DictReader((P / root / s / m / r / "frames.csv").open()):
            if not x["flag"]:
                t.append(float(x["time_ps"]) / 1000.0)
                g.append(float(x["delta_g_mmpbsa"]))
    t, g = np.array(t), np.array(g)
    if len(np.unique(t)) != len(t):
        raise RuntimeError(f"{s}/{m}/{r}: a frame appears in both windows")
    o = np.argsort(t)
    return t[o], g[o]


def boot(rep_mean, score, rng):
    """Hierarchical bootstrap over starts then replicas: Spearman and pairwise order keep."""
    rho, keep = [], np.zeros((8, 8))
    for _ in range(N_BOOT):
        bs = np.array([np.mean([rep_mean[i, k, rng.integers(0, 3, 3)].mean() for k in rng.integers(0, 2, 2)])
                       for i in range(8)])
        rho.append(spearman(score, bs))
        keep += np.sign(bs[:, None] - bs[None, :]) == np.sign(score[:, None] - score[None, :])
    return np.array(rho), keep / N_BOOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["main", "split"], default="main")
    args = ap.parse_args()
    roots = ("scores_1_20",) if args.source == "main" else ("scores", "scores_10_20")
    OUT = P / ("scores_1_20_summary" if args.source == "main" else "scores_ext20")
    OUT.mkdir(exist_ok=True)
    keys = list(itertools.product(SAMPLES, MODELS, REPS))
    data = {k: frames(*k, roots) for k in keys}

    # 3-ns block means 1-19 ns
    B = np.array([[data[k][1][(data[k][0] >= b0) & (data[k][0] < b1)].mean() for b0, b1 in BLOCKS] for k in keys])
    with (OUT / "block_means.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "model", "replica", "drift_v5"] + [f"{b0:g}-{b1:g}ns" for b0, b1 in BLOCKS])
        for k, row in zip(keys, B):
            w.writerow(list(k) + [k in DRIFT] + [round(v, 3) for v in row])
    se = B.std(0, ddof=1) / np.sqrt(len(keys))
    d = lambda j, i: (float((B[:, j] - B[:, i]).mean()), float((B[:, j] - B[:, i]).std(ddof=1) / np.sqrt(len(keys))),
                      int(((B[:, j] - B[:, i]) < 0).sum()))
    trend = dict(block_mean_over_48=[round(float(v), 3) for v in B.mean(0)], block_se_over_48=[round(float(v), 3) for v in se],
                 blocks=[f"{b0:g}-{b1:g}" for b0, b1 in BLOCKS],
                 d_7_10_minus_1_4=d(2, 0), d_16_19_minus_10_13=d(5, 3), d_10_19_minus_1_10=None,
                 note="d = (mean, SE over 48 replicas, n replicas with negative difference)")
    late, early = B[:, 3:].mean(1), B[:, :3].mean(1)
    trend["d_10_19_minus_1_10"] = (float((late - early).mean()), float((late - early).std(ddof=1) / np.sqrt(48)),
                                   int(((late - early) < 0).sum()))

    rng = np.random.default_rng(20261016)
    win_out, rows, sample_scores = {}, [], {}
    for wname, (w0, w1) in WINDOWS.items():
        rm = np.array([data[k][1][(data[k][0] >= w0 - 1e-9) & (data[k][0] < w1 - 1e-9)].mean() for k in keys]).reshape(8, 2, 3)
        nfr = [int(((data[k][0] >= w0 - 1e-9) & (data[k][0] < w1 - 1e-9)).sum()) for k in keys]
        taus = [sokal_tau(data[k][1][(data[k][0] >= w0 - 1e-9) & (data[k][0] < w1 - 1e-9)])[0] * 10.0 for k in keys]
        score = rm.reshape(8, 6).mean(1)
        sample_scores[wname] = score
        sd = rm.reshape(8, 6).std(1, ddof=1)
        for i, s in enumerate(SAMPLES):
            rows.append(dict(window=wname, sample=s, score=score[i], sd_rep=sd[i], se_rep=sd[i] / np.sqrt(6),
                             m1=rm[i, 0].mean(), m2=rm[i, 1].mean(), m1_minus_m2=rm[i, 0].mean() - rm[i, 1].mean()))
        # variance split on full 3-ns blocks inside the window
        bi = [j for j, (b0, b1) in enumerate(BLOCKS) if b0 >= w0 - 1e-9 and b1 <= w1 + 1e-9]
        M = B[:, bi].reshape(8, 2, 3, len(bi))
        rmb = M.mean(-1)
        smb = rmb.mean(-1)
        vb = float(((M - rmb[..., None]) ** 2).sum() / (48 * (len(bi) - 1)))
        vr = float(((rmb - smb[..., None]) ** 2).sum() / 32)
        vs = float(((smb - smb.mean(-1, keepdims=True)) ** 2).sum() / 8)
        rho, keep = boot(rm, score, rng)
        order = list(np.argsort(score))
        pairs = [dict(better=SAMPLES[i], worse=SAMPLES[j], diff=float(score[j] - score[i]), boot_keep=float(keep[i, j]))
                 for i, j in itertools.combinations(order, 2)]
        win_out[wname] = dict(
            frames_per_replica=sorted(set(nfr)), tau_ps_median=float(np.median(taus)), tau_ps_max=float(max(taus)),
            blocks_in_variance_split=[f"{BLOCKS[j][0]:g}-{BLOCKS[j][1]:g}" for j in bi],
            sd_block_within_replica=vb ** .5, sd_replica_within_start=vr ** .5, sd_start_within_sample=vs ** .5,
            replica_excess_var_over_block_noise=vr - vb / len(bi), start_excess_var_over_replica_noise=vs - vr / 3,
            order_best_first=[SAMPLES[i] for i in order], score_best_first=[round(float(score[i]), 2) for i in order],
            spearman_m1_vs_m2=spearman(rm[:, 0].mean(1), rm[:, 1].mean(1)),
            boot_spearman=dict(median=float(np.median(rho)), p5=float(np.percentile(rho, 5)), p95=float(np.percentile(rho, 95))),
            n_pairs_keep_ge_0_95=sum(p["boot_keep"] >= 0.95 for p in pairs), n_pairs=len(pairs), pairs=pairs)
    with (OUT / "window_samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for x in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})

    # structure 0-10 vs 10-20 ns by frame index (frame f is 10 ps x (f+1) into production); the
    # time_ps column of diagnostics.csv is mdtraj's per-chunk DCD time (0-99 per chunk), not used
    srows = []
    for s, m, r in keys:
        D = list(csv.DictReader((P / "md" / s / m / r / "analysis" / "diagnostics.csv").open()))
        out = dict(sample=s, model=m, replica=r, drift_v5=(s, m, r) in DRIFT, n_frames=len(D))
        for lab, (a, b) in (("0_10", (0, 1000)), ("10_20", (1000, 2000))):
            X = [x for x in D if a <= int(x["frame"]) < b]
            out[f"stereo_change_frames_{lab}"] = sum(int(x["stereo_class_changes"]) > 0 for x in X)
            out[f"junction_4sigma_frames_{lab}"] = sum(int(x["junction_terms_outside_4sigma"]) > 0 for x in X)
            out[f"pep_rmsd_median_{lab}"] = float(np.median([float(x["peptide_CA_rmsd_receptor_frame_A"]) for x in X]))
            out[f"min_contact_p5_{lab}"] = float(np.percentile([float(x["min_rec_pep_heavy_A"]) for x in X], 5))
        srows.append(out)
    with (OUT / "structure.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0]))
        w.writeheader()
        for x in srows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
    dr = np.array([x["pep_rmsd_median_10_20"] - x["pep_rmsd_median_0_10"] for x in srows])
    structure = dict(
        n_frames=sorted({x["n_frames"] for x in srows}),
        stereo_change_frames_10_20=sum(x["stereo_change_frames_10_20"] for x in srows),
        junction_4sigma_frames_10_20=sum(x["junction_4sigma_frames_10_20"] for x in srows),
        min_contact_p5_10_20_range=[min(x["min_contact_p5_10_20"] for x in srows), max(x["min_contact_p5_10_20"] for x in srows)],
        pep_rmsd_median_change_10_20_vs_0_10=dict(median=float(np.median(dr)), min=float(dr.min()), max=float(dr.max())),
        pep_rmsd_rise_gt_0_5_A=[f"{x['sample']}/{x['model']}/{x['replica']}: {x['pep_rmsd_median_0_10']:.2f}->{x['pep_rmsd_median_10_20']:.2f}"
                                for x in srows if x["pep_rmsd_median_10_20"] - x["pep_rmsd_median_0_10"] > 0.5],
        note="0.5 Å groups the description only (as in V5); not a criterion")

    spear = {f"{a}_vs_{b}": spearman(sample_scores[a], sample_scores[b]) for a, b in itertools.combinations(WINDOWS, 2)}
    paired = None
    if args.source == "main":
        prow, agg = [], {}
        for s, m, r in keys:
            row = dict(sample=s, model=m, replica=r)
            for field in ("delta_g_mmpbsa", "delta_g_pb"):
                new = read("scores_1_20", s, m, r, field)
                for lab, old_root in (("1_10", "scores"), ("10_20", "scores_10_20")):
                    old = read(old_root, s, m, r, field)
                    common = sorted(set(new) & set(old))
                    dd = np.array([new[f] - old[f] for f in common])
                    row[f"n_{lab}"] = len(common)
                    row[f"{field}_mean_diff_{lab}"] = float(dd.mean())
                    row[f"{field}_sd_diff_{lab}"] = float(dd.std(ddof=1))
                    agg.setdefault((field, lab), []).append(dd)
            prow.append(row)
        with (OUT / "sizing_paired.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(prow[0]))
            w.writeheader()
            for x in prow:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
        paired = {}
        for (field, lab), lst in agg.items():
            allf = np.concatenate(lst)
            rmeans = np.array([x.mean() for x in lst])
            paired[f"{field}_{lab}"] = dict(
                n_frames=int(len(allf)), frame_diff_mean=float(allf.mean()), frame_diff_sd=float(allf.std(ddof=1)),
                frame_abs_diff_p95=float(np.percentile(np.abs(allf), 95)), frame_abs_diff_max=float(np.abs(allf).max()),
                replica_mean_diff_range=[float(rmeans.min()), float(rmeans.max())],
                replica_mean_abs_diff_median=float(np.median(np.abs(rmeans))))
        paired["note"] = "single sizing (scores_1_20) minus split sizing (scores / scores_10_20), same frames, kcal/mol"
    summary = dict(source=args.source, roots=list(roots), trend=trend, windows=win_out, spearman_between_windows=spear,
                   structure=structure, sizing_paired=paired,
                   note="no experimental label read; replica means are the unit; bootstrap resamples starts then replicas")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    brief = dict(source=args.source, trend=trend, spearman_between_windows=spear, sizing_paired=paired,
                 windows={k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in win_out.items()})
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
