#!/usr/bin/env python3
"""Hou fixed 6 x 10 ns experiment ranking, benchmark design section 9.

Consumes the strict sampling audit; no MD/JAX/OpenMM imports. Reports separate
replica/time-block and candidate-composition resampling, not frame-iid errors.
Experimental values are Table S1 dG labels, not newly verified Ki/Kd assays.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from smallmol_records import file_identity

PACKAGE = Path(__file__).resolve().parents[1] / "experiments/2026-10-17_small_molecule_benchmark"
SAMPLES = tuple(f"a{i}" for i in range(1, 8))
GROUPS = {"a1_a4_same_series": (0, 1, 2, 3),
          "all_seven_exploratory": tuple(range(7)), "a5_a7_secondary": (4, 5, 6)}


def correlations(x, y):
    """Batched Spearman and Kendall tau-b, including candidate-bootstrap ties."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    rx, ry = rankdata(x, axis=-1), rankdata(y, axis=-1)
    rx, ry = rx - rx.mean(axis=-1, keepdims=True), ry - ry.mean(axis=-1, keepdims=True)
    denom = np.sqrt((rx * rx).sum(axis=-1) * (ry * ry).sum(axis=-1))
    rho = np.divide((rx * ry).sum(axis=-1), denom,
                    out=np.full(denom.shape, np.nan), where=denom > 0)
    i, j = np.triu_indices(x.shape[-1], 1)
    dx, dy = np.sign(x[..., i] - x[..., j]), np.sign(y[..., i] - y[..., j])
    denom = np.sqrt((dx != 0).sum(axis=-1) * (dy != 0).sum(axis=-1))
    tau = np.divide((dx * dy).sum(axis=-1), denom,
                    out=np.full(denom.shape, np.nan), where=denom > 0)
    return rho, tau


def exact_statistics(score, experiment):
    rho, tau = correlations(score, experiment)
    permutations = np.array(list(itertools.permutations(experiment)))
    null_rho, null_tau = correlations(score, permutations)
    return dict(spearman_rho=float(rho), kendall_tau_b=float(tau),
                spearman_p_exact_two_sided=float(np.mean(abs(null_rho) >= abs(rho) - 1e-12)),
                kendall_p_exact_two_sided=float(np.mean(abs(null_tau) >= abs(tau) - 1e-12)),
                n_label_permutations=len(permutations))


def interval(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    return dict(percentile_2_5_97_5=np.quantile(values[finite], [.025, .975]).tolist(),
                defined_draws=int(finite.sum()), undefined_draws=int((~finite).sum()))


def sampling_draws(blocks, rng, draws, block_ns):
    """Resample six replicas, then circular contiguous 1/2/5-ns blocks per draw.

    The ten 1-ns means retain temporal order. Every sampled replica still spans
    ten ns; a block is never counted as an independent biological replica.
    """
    output = np.empty((draws, len(blocks)))
    for s, block in enumerate(blocks):
        replicas = rng.integers(0, 6, size=(draws, 6))
        starts = rng.integers(0, 10, size=(draws, 6, 10 // block_ns))
        indices = (starts[..., None] + np.arange(block_ns)) % 10
        values = block[replicas[..., None, None], indices]
        output[:, s] = values.mean(axis=(1, 2, 3))
    return output


def load_inputs(package, analysis):
    audit = json.loads((analysis / "audit.json").read_text())
    if (audit.get("n_frames") != 42000 or audit.get("full_window_usable_replicas") != 42
            or audit.get("full_window_available_samples") != list(SAMPLES) or audit.get("recovery_required")):
        raise RuntimeError("complete valid scores for all seven samples are required")
    for receipt in [audit["manifest"], audit["analysis_script"], *audit["sources"]]:
        if file_identity(receipt["path"]) != receipt:
            raise RuntimeError(f"sampling audit is stale: {receipt['path']}")
    replicas = list(csv.DictReader((analysis / "replicas.csv").open()))
    records = list(csv.DictReader((analysis / "blocks.csv").open()))
    if len(replicas) != 42 or len(records) != 420:
        raise RuntimeError("expected 42 replicas and 420 one-ns blocks")
    scores = np.empty((7, 6))
    blocks = np.empty((7, 6, 10))
    seen_reps, seen_blocks = set(), set()
    for r in replicas:
        k = (SAMPLES.index(r["sample"]), int(r["replica"]) - 1)
        if k in seen_reps or not 0 <= k[1] < 6 or r["usable_for_full_window"] != "True":
            raise RuntimeError("invalid/duplicate replica in sampling analysis")
        seen_reps.add(k)
        scores[k] = float(r["mean"])
    for r in records:
        k = (SAMPLES.index(r["sample"]), int(r["replica"]) - 1, int(r["block_start_ns"]))
        if (k in seen_blocks or not 0 <= k[1] < 6 or not 0 <= k[2] < 10
                or int(r["block_end_ns"]) != k[2] + 1
                or int(r["n_frames"]) != 100 or int(r["invalid_frames"]) != 0):
            raise RuntimeError("invalid/duplicate/incomplete analysis block")
        seen_blocks.add(k)
        blocks[k] = float(r["mean_delta_g_mmpbsa"])
    if (not np.isfinite(scores).all() or not np.isfinite(blocks).all()
            or not np.allclose(scores, blocks.mean(axis=-1), rtol=0, atol=1e-10)):
        raise RuntimeError("replica means and complete time blocks disagree")
    experiment_path = package / "manifests/hou2011_table_s1.csv"
    exp_rows = [r for r in csv.DictReader(experiment_path.open()) if r["sample_id"] in SAMPLES]
    if len(exp_rows) != 7 or {r["sample_id"] for r in exp_rows} != set(SAMPLES):
        raise RuntimeError("missing or duplicate experimental labels")
    experiment = np.array([float(next(r for r in exp_rows if r["sample_id"] == s)["experimental_dg_kcal_mol"])
                           for s in SAMPLES])
    if not np.isfinite(experiment).all():
        raise RuntimeError("nonfinite experimental label")
    return audit, replicas, scores, blocks, experiment, experiment_path


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(package, analysis, out, draws, seed):
    if out.exists():
        raise RuntimeError(f"preserve existing results; use a new --out: {out}")
    if draws < 1000:
        raise ValueError("use at least 1000 bootstrap draws")
    audit, replicas, scores, blocks, exp, experiment_path = load_inputs(package, analysis)
    means = scores.mean(axis=1)
    errors = scores.std(axis=1, ddof=1) / math.sqrt(6)
    rng = np.random.default_rng(seed)
    sampling = {b: sampling_draws(blocks, rng, draws, b) for b in (1, 2, 5)}
    result = dict(protocol=dict(grid="frozen default h=0.5 / h_lig=0.25", window_ns=[0, 10],
                                n_replicas_per_sample=6, bootstrap_draws=draws, seed=seed,
                                circular_block_lengths_ns=[1, 2, 5],
                                score_sources=dict(online=audit["online_valid_replicas"],
                                                   offline_recovered=audit["recovered_offline_replicas"])), groups={})
    for name, ids in GROUPS.items():
        ids = np.array(ids)
        group = dict(samples=[SAMPLES[i] for i in ids], n_samples=len(ids),
                     **exact_statistics(means[ids], exp[ids]), sampling_bootstrap={})
        for b, values in sampling.items():
            rho, tau = correlations(values[:, ids], exp[ids])
            group["sampling_bootstrap"][str(b)] = dict(spearman=interval(rho), kendall=interval(tau))
        candidates = rng.integers(0, len(ids), size=(draws, len(ids)))
        rho, tau = correlations(means[ids][candidates], exp[ids][candidates])
        group["candidate_composition_bootstrap"] = dict(spearman=interval(rho), kendall=interval(tau))
        signs = [(means[a] - means[b]) * (exp[a] - exp[b]) for a, b in itertools.combinations(ids, 2)]
        group["concordant_pairs"] = int(sum(x > 0 for x in signs))
        group["discordant_pairs"] = int(sum(x < 0 for x in signs))
        group["tied_pairs"] = int(sum(x == 0 for x in signs))
        result["groups"][name] = group
    pairs = []
    for a, b in itertools.combinations(range(7), 2):
        pairs.append(dict(sample_a=SAMPLES[a], sample_b=SAMPLES[b],
                          experiment_stronger=SAMPLES[a] if exp[a] < exp[b] else SAMPLES[b],
                          score_stronger=SAMPLES[a] if means[a] < means[b] else SAMPLES[b],
                          concordant=bool((means[a] - means[b]) * (exp[a] - exp[b]) > 0),
                          score_a_minus_b=float(means[a] - means[b]),
                          combined_replica_se=float(np.hypot(errors[a], errors[b])),
                          **{f"bootstrap_fraction_a_stronger_block{block_ns}ns":
                             float(np.mean(values[:, a] < values[:, b]) + .5 * np.mean(values[:, a] == values[:, b]))
                             for block_ns, values in sampling.items()}))
    result["sources"] = [file_identity(__file__), file_identity(experiment_path)] + [
        file_identity(analysis / filename) for filename in ("audit.json", "replicas.csv", "blocks.csv", "samples.csv")]
    result["limitations"] = [
        "Primary chemical-series comparison a1-a4; all-seven and a5-a7 comparisons are supplementary.",
        "Table S1 labels are used as reported; original assay types/conditions are not newly established as uniform.",
        "Unknown experimental errors are not assigned an invented distribution.",
        "Score is entropy-free MM/PBSA, not an absolute binding free-energy prediction.",
        "Permutation p values condition on these fixed candidate labels; small n limits interpretation.",
        "Sampling bootstrap conditions on visited states, with six replicas and circular contiguous time blocks.",
        "Bootstrap percentiles are sensitivity diagnostics; unvisited states and numerical/model errors are absent.",
        "Candidate bootstrap is separate from MD bootstrap; constant-candidate draws have undefined correlation and are counted.",
        "h0.4 has no complete 6x10-ns rescoring here and is not declared converged; no grid is chosen from correlation."]
    out.mkdir(parents=True)
    write_csv(out / "pairs.csv", pairs)
    sample_rows = [dict(sample=s, experimental_dg=float(exp[i]), score=float(means[i]),
                        replica_se=float(errors[i]), score_rank=int(rankdata(means)[i]),
                        experiment_rank=int(rankdata(exp)[i]),
                        offline_recovered_replicas=sum(r["sample"] == s and r["recovered_offline"] == "True"
                                                       for r in replicas)) for i, s in enumerate(SAMPLES)]
    write_csv(out / "samples.csv", sample_rows)
    (out / "ranking.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    text = ["# Hou 6 × 10 ns experimental ranking", "",
            "Default grid; 39 online replicas plus 3 whole-replica offline recoveries. "
            "Table S1 experimental dG labels; lower score and lower label both mean stronger binding.", "",
            "| Group | n | Spearman rho | Kendall tau-b | Concordant pairs | Exact two-sided p (rho) |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, g in result["groups"].items():
        total = g["concordant_pairs"] + g["discordant_pairs"] + g["tied_pairs"]
        text.append(f"| {name} | {g['n_samples']} | {g['spearman_rho']:.3f} | {g['kendall_tau_b']:.3f} | "
                    f"{g['concordant_pairs']}/{total} | {g['spearman_p_exact_two_sided']:.4f} |")
    text += ["", "| Group | Block ns | Sampling-bootstrap rho 2.5–97.5 percentiles |",
             "|---|---:|---:|"]
    for name, g in result["groups"].items():
        for b, v in g["sampling_bootstrap"].items():
            lo, hi = v["spearman"]["percentile_2_5_97_5"]
            text.append(f"| {name} | {b} | {lo:.3f} … {hi:.3f} |")
    text += ["", "All pairwise orders and resampling frequencies: `pairs.csv`. "
             "Candidate-composition bootstrap and its undefined-draw counts: `ranking.json`.", ""]
    text += [f"- {note}" for note in result["limitations"]]
    (out / "report.md").write_text("\n".join(text) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxpbsa-ranking-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, s in enumerate(SAMPLES):
        color = "#2166ac" if i < 4 else "#777777"
        ax.scatter(np.full(6, exp[i]), scores[i], color=color, alpha=.32, s=23)
        ax.errorbar(exp[i], means[i], yerr=errors[i], fmt="o", color=color, capsize=3)
        ax.annotate(s, (exp[i], means[i]), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Table S1 experimental dG (kcal/mol)",
           ylabel="Default-grid MM/PBSA score (kcal/mol; no entropy)",
           title="Hou: 6 replicas x 10 ns\nPoints: replicas; bars: SE of replica means")
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out / "score_vs_experiment.png", dpi=180)
    fig.savefig(out / "score_vs_experiment.pdf")
    plt.close(fig)
    print(json.dumps({name: {k: g[k] for k in ("spearman_rho", "kendall_tau_b", "concordant_pairs",
                                               "spearman_p_exact_two_sided")}
                      for name, g in result["groups"].items()}))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    run(args.package, args.analysis_dir, args.out, args.bootstrap_draws, args.seed)
