#!/usr/bin/env python3
"""Aggregate the P4 per-frame scores of the 48 v2.1 replicas (design §4, 2026-10-15 plan).

    python scripts/aggregate_scores.py            # run from the packet directory

Reads scores/<sample>/<model>/<rep>/frames.csv (authoritative; summary.json only for timing)
and writes, without reading any experimental label:
    scores/replica_table.csv    one row per replica: valid/flagged frames, mean, SD, 3-ns block
                                means, b3-b1, Sokal tau, N_eff, ms/frame, drift note (V5)
    scores/sample_table.csv     one row per sample: mean of the 6 replica means, SD/SE over
                                replicas, per-start means, start difference, per-block means
    scores/aggregate.json       variance split (frames->blocks->replicas->starts), block-trend
                                counts, rank stability (by block, by start, replica bootstrap)
Statistical unit: replica mean (protocol §2.1).  Frames are never treated as independent.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
SAMPLES = ["PMI", "PMI-L10A", "T1L_L10A", "A4L_L10A", "N8A_L10A", "S11M_L10A", "P12Q_L10A", "PMI-M4"]
MODELS = ["v21m1", "v21m2"]
REPS = ["rep01", "rep02", "rep03"]
BLOCKS = ((1000, 4000), (4000, 7000), (7000, 10000))       # ps
# V5 (design §3, 2026-10-15): peptide CA RMSD rose by >0.5 Å in the second half
DRIFT = {("A4L_L10A", "v21m2", "rep03"), ("A4L_L10A", "v21m1", "rep01"), ("A4L_L10A", "v21m1", "rep02"),
         ("PMI-L10A", "v21m1", "rep03"), ("S11M_L10A", "v21m1", "rep01")}
N_BOOT = 20000


def sokal_tau(g):
    x = g - g.mean()
    v = x @ x / len(x)
    acf = [1.0] + [(x[:-k] @ x[k:]) / (len(x) - k) / v for k in range(1, min(len(g) // 2, 450) + 1)]
    tau = win = None
    for M in range(1, len(acf)):
        tau, win = 1.0 + 2.0 * sum(acf[1:M + 1]), M
        if M >= 5 * tau:
            break
    return tau, win


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    reps = []
    for s, m, r in itertools.product(SAMPLES, MODELS, REPS):
        d = P / "scores" / s / m / r
        rows = sorted(csv.DictReader((d / "frames.csv").open()), key=lambda x: int(x["frame"]))
        valid = [x for x in rows if not x["flag"]]
        g = np.array([float(x["delta_g_mmpbsa"]) for x in valid])
        t = np.array([float(x["time_ps"]) for x in valid])
        b = [float(g[(t >= b0) & (t < b1)].mean()) for b0, b1 in BLOCKS]
        tau, win = sokal_tau(g)
        sm = json.loads((d / "summary.json").read_text())
        ms = sm.get("ms_per_frame_this_call") or {}
        reps.append(dict(sample=s, model=m, replica=r, n_frames=len(rows), n_valid=len(valid),
                         n_flagged=len(rows) - len(valid), t_first_ps=t.min(), t_last_ps=t.max(),
                         mean=g.mean(), sd=g.std(ddof=1), b1=b[0], b2=b[1], b3=b[2], b3_minus_b1=b[2] - b[0],
                         block_se=np.std(b, ddof=1) / np.sqrt(3), tau_ps=tau * 10.0, sokal_window_ps=win * 10.0,
                         n_eff=len(g) / tau, ms_per_frame=ms.get("median"), warmup_s=sm.get("warmup_s"),
                         drift_v5=(s, m, r) in DRIFT,
                         **{f"{k}_mean": float(np.mean([float(x[k]) for x in valid]))
                            for k in ("e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa")}))

    keys = list(reps[0])
    with (P / "scores" / "replica_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in reps:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})

    def arr(s, m=None, key="mean"):
        return np.array([x[key] for x in reps if x["sample"] == s and (m is None or x["model"] == m)])

    samp = []
    for s in SAMPLES:
        a, a1, a2 = arr(s), arr(s, "v21m1"), arr(s, "v21m2")
        samp.append(dict(sample=s, n_rep=len(a), score=a.mean(), sd_rep=a.std(ddof=1), se_rep=a.std(ddof=1) / np.sqrt(len(a)),
                         m1=a1.mean(), m2=a2.mean(), m1_minus_m2=a1.mean() - a2.mean(),
                         sd_within_m1=a1.std(ddof=1), sd_within_m2=a2.std(ddof=1),
                         score_b1=arr(s, key="b1").mean(), score_b2=arr(s, key="b2").mean(), score_b3=arr(s, key="b3").mean(),
                         score_no_drift=np.mean([x["mean"] for x in reps if x["sample"] == s and not x["drift_v5"]]),
                         n_drift=sum(x["drift_v5"] for x in reps if x["sample"] == s)))
    with (P / "scores" / "sample_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(samp[0]))
        w.writeheader()
        for x in samp:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})

    # variance split (one-way nested, balanced): blocks in replica, replicas in start, starts in sample
    M = np.array([[[[next(x for x in reps if (x["sample"], x["model"], x["replica"]) == (s, m, r))[k]
                     for k in ("b1", "b2", "b3")] for r in REPS] for m in MODELS] for s in SAMPLES])  # 8x2x3x3
    rep_mean = M.mean(-1)
    start_mean = rep_mean.mean(-1)
    var_block = float(((M - rep_mean[..., None]) ** 2).sum() / (8 * 2 * 3 * 2))           # within replica
    var_rep = float(((rep_mean - start_mean[..., None]) ** 2).sum() / (8 * 2 * 2))         # replica means in start
    var_start = float(((start_mean - start_mean.mean(-1, keepdims=True)) ** 2).sum() / (8 * 1))  # start means in sample
    score = np.array([x["score"] for x in samp])

    # rank stability
    by_block = {f"b{j}": spearman(score, [x[f"score_b{j}"] for x in samp]) for j in (1, 2, 3)}
    by_start = spearman([x["m1"] for x in samp], [x["m2"] for x in samp])
    rng = np.random.default_rng(20261015)
    R = rep_mean.reshape(8, 6)
    boot_rho, pair_keep = [], np.zeros((8, 8))
    for _ in range(N_BOOT):                      # hierarchical: resample starts, then replicas in each
        bs = np.empty(8)
        for i in range(8):
            st = rng.integers(0, 2, 2)
            bs[i] = np.mean([rep_mean[i, k, rng.integers(0, 3, 3)].mean() for k in st])
        boot_rho.append(spearman(score, bs))
        pair_keep += (np.sign(bs[:, None] - bs[None, :]) == np.sign(score[:, None] - score[None, :]))
    pair_keep /= N_BOOT
    order = list(np.argsort(score))
    pairs = []
    for i, j in itertools.combinations(order, 2):
        se = np.hypot(samp[i]["se_rep"], samp[j]["se_rep"])
        pairs.append(dict(better=SAMPLES[i], worse=SAMPLES[j], diff=float(score[j] - score[i]),
                          se=float(se), z=float((score[j] - score[i]) / se), boot_keep=float(pair_keep[i, j])))

    d31 = np.array([x["b3_minus_b1"] for x in reps])
    mono_down = sum(x["b1"] > x["b2"] > x["b3"] for x in reps)
    mono_up = sum(x["b1"] < x["b2"] < x["b3"] for x in reps)
    out = dict(
        n_replicas=len(reps), n_frames=sum(x["n_frames"] for x in reps), n_valid=sum(x["n_valid"] for x in reps),
        ms_per_frame_median=float(np.median([x["ms_per_frame"] for x in reps if x["ms_per_frame"]])),
        tau_ps=dict(median=float(np.median([x["tau_ps"] for x in reps])), min=min(x["tau_ps"] for x in reps),
                    max=max(x["tau_ps"] for x in reps)),
        variance_kcal2=dict(block_means_within_replica=var_block, replica_means_within_start=var_rep,
                            start_means_within_sample=var_start),
        sd_kcal=dict(block_within_replica=var_block ** .5, replica_within_start=var_rep ** .5,
                     start_within_sample=var_start ** .5),
        block_trend=dict(b3_minus_b1_mean=float(d31.mean()), b3_minus_b1_se=float(d31.std(ddof=1) / np.sqrt(len(d31))),
                         n_b3_more_negative=int((d31 < 0).sum()), n_monotone_down=mono_down, n_monotone_up=mono_up,
                         n_monotone_expected_by_chance=len(reps) / 3),
        rank=dict(order_best_first=[SAMPLES[i] for i in order],
                  spearman_full_vs_block=by_block, spearman_m1_vs_m2=by_start,
                  spearman_full_vs_no_drift=spearman(score, [x["score_no_drift"] for x in samp]),
                  boot_spearman=dict(median=float(np.median(boot_rho)), p5=float(np.percentile(boot_rho, 5)),
                                     p95=float(np.percentile(boot_rho, 95))),
                  pairs=pairs, n_boot=N_BOOT),
        note="no experimental label read; replica means are the unit; bootstrap resamples starts then replicas",
    )
    (P / "scores" / "aggregate.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rank"} | {"rank": {k: v for k, v in out["rank"].items() if k != "pairs"}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
