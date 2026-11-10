#!/usr/bin/env python3
"""Hou alpha-thrombin a1-a7 first round: online MM/PBSA scores -> per-replica / per-sample tables and
the ranking comparison with Table S1 (read-only; run on 183, 181's NFS view lags).

Plan: docs/BRD4_HOU_EXECUTION_HANDOFF.md §3.1 (a1-a4 same series first, all seven as extra),
§4.7 (0-5 ns, five 1-ns blocks), §7.12; design docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md §9.
  * rows: online/pbsa_from_chunk_NNNN.csv of each replica, segments taken from status.json
    "online" (a segment owns steps (from_chunk, to_chunk] * chunk_steps), window (0, 5] ns;
  * replica score = time mean of delta_g_mmpbsa; sample score = equal-weight mean of replicas;
  * the score is an MM/PBSA score without entropy: only its ORDER is compared with the
    experimental dG (Spearman rho main, Kendall tau-b and pairwise order as supplements).
Writes reports/hou_online_replicas.csv, reports/hou_online_samples.csv, reports/hou_online_ranking.json.
"""
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P.parents[1] / "scripts"))
from smallmol_records import online_rows, row_violation
SAMPLES = [f"a{i}" for i in range(1, 8)]
COMP = ["e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa"]
T0, T1, BLOCK = 0.0, 5000.0, 1000.0

exp = {}
for r in csv.DictReader((P / "manifests/hou2011_table_s1.csv").open()):
    if r["sample_id"] in SAMPLES:
        exp[r["sample_id"]] = float(r["experimental_dg_kcal_mol"])

rep_rows = []
for run in csv.DictReader((P / "manifests/runs.csv").open()):
    if run["target"] != "hou":
        continue
    out = P / run["output_directory"]
    st = json.loads((out / "status.json").read_text())
    cs = st["chunk_steps"]
    required_step = round(T1 / st["dt_ps"])
    if st["chunks_done"] * cs < required_step:
        raise RuntimeError(f"{run['run_id']}: MD has not reached {T1} ps")
    rows = online_rows(st, through_step=required_step)
    # Step-derived analysis times avoid double-counting a boundary whose OpenMM
    # accumulated time lies a few ulps above the exact 1000/2000/... ps value.
    t = np.array([int(x["step"]) * st["dt_ps"] for x in rows])
    bad = [x for x in rows if row_violation(x, x["_margin_flag"])]
    if bad:
        raise RuntimeError(f"{run['run_id']}: {len(bad)} invalid frames; ranking not written")
    v = {k: np.array([float(x[k]) for x in rows]) for k in COMP}
    blocks = [float(v["delta_g_mmpbsa"][(t > b + 1e-6) & (t <= b + BLOCK + 1e-6)].mean())
              for b in np.arange(T0, T1, BLOCK)]
    diag = json.loads((out / "analysis/diagnostics_summary.json").read_text())["quantiles_p5_p50_p95"]
    seg0 = st["online"][-1]
    n_rep = sum(s["n_reports"] for s in st["online"])
    prod_s = sum(w for k, w in st["stage_wall_s"].items() if k.startswith("prod"))
    rep_rows.append(dict(
        run_id=run["run_id"], sample=run["sample_id"], replica=int(run["replica"]), n_frames=len(rows),
        first_ps=t.min(), last_ps=t.max(), frames_flagged_by_reporter=sum(s["n_flagged"] for s in st["online"]),
        frames_bad_by_columns=len(bad), **{f"mean_{k}": round(float(v[k].mean()), 3) for k in COMP},
        sd_dg_mmpbsa=round(float(v["delta_g_mmpbsa"].std(ddof=1)), 3),
        **{f"block{i + 1}_dg": round(b, 3) for i, b in enumerate(blocks)},
        lig_rmsd_p50=diag["ligand_heavy_rmsd_receptor_frame_A"]["50"],
        lig_rmsd_p95=diag["ligand_heavy_rmsd_receptor_frame_A"]["95"],
        lig_com_shift_p95=diag["ligand_com_shift_A"]["95"], min_lig_rec_p50=diag["min_lig_rec_heavy_A"]["50"],
        prod_s_per_ns=round(prod_s / (st["chunks_done"] * cs * st["dt_ps"] / 1000), 1),
        pbsa_s_per_frame=round(sum(s["total_pbsa_s"] for s in st["online"]) / max(n_rep, 1), 3),
        max_latency_s=max(s["max_latency_s"] for s in st["online"]), warmup_s=seg0["warmup_time_s"],
        mps=seg0["mps_thread_percentage"]))

(P / "reports").mkdir(exist_ok=True)
with (P / "reports/hou_online_replicas.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rep_rows[0]))
    w.writeheader(); w.writerows(rep_rows)

samp = []
for s in SAMPLES:
    rr = [r for r in rep_rows if r["sample"] == s]
    sc = np.array([r["mean_delta_g_mmpbsa"] for r in rr])
    samp.append(dict(sample=s, n_replicas=len(rr), score=round(float(sc.mean()), 3),
                     replica_range=round(float(sc.max() - sc.min()), 3),
                     **{f"mean_{k}": round(float(np.mean([r[f'mean_{k}'] for r in rr])), 3) for k in COMP[:-1]},
                     exp_dg=exp[s]))
with (P / "reports/hou_online_samples.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(samp[0]))
    w.writeheader(); w.writerows(samp)


def ranking(ids):
    sc = np.array([next(x["score"] for x in samp if x["sample"] == i) for i in ids])
    ex = np.array([exp[i] for i in ids])
    rho, p_rho = stats.spearmanr(sc, ex)
    tau, p_tau = stats.kendalltau(sc, ex)
    pairs = []
    for a, b in itertools.combinations(range(len(ids)), 2):
        pairs.append(dict(pair=f"{ids[a]}/{ids[b]}", exp_stronger=ids[a] if ex[a] < ex[b] else ids[b],
                          score_stronger=ids[a] if sc[a] < sc[b] else ids[b],
                          concordant=bool((ex[a] - ex[b]) * (sc[a] - sc[b]) > 0),
                          score_gap=round(float(abs(sc[a] - sc[b])), 3)))
    return dict(samples=ids, spearman_rho=round(float(rho), 3), spearman_p=round(float(p_rho), 4),
                kendall_tau_b=round(float(tau), 3), kendall_p=round(float(p_tau), 4),
                concordant_pairs=f"{sum(p['concordant'] for p in pairs)}/{len(pairs)}", pairs=pairs)


res = {"a1_a4_same_series": ranking(SAMPLES[:4]), "all_seven": ranking(SAMPLES),
       "note": "score = MM/PBSA without entropy; order only, no absolute comparison (design §9)"}
(P / "reports/hou_online_ranking.json").write_text(json.dumps(res, indent=2) + "\n")
for r in samp:
    print(json.dumps(r))
for k in ("a1_a4_same_series", "all_seven"):
    print(k, {x: res[k][x] for x in ("spearman_rho", "spearman_p", "kendall_tau_b", "concordant_pairs")})
