#!/usr/bin/env python3
"""P2 BRD4 experimental-order statistics for the default and h0.4 grids; read-only.

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-10-17_small_molecule_benchmark/scripts/brd4_statistics.py

Plan docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §5.  Groups, exclusion (BRD4-5 identity mismatch),
pairs and statistics are fixed in reports/brd4_statistics_20261101/protocol.json; labels and
their audit in label_audit.csv.  Replica means and 1-ns block means come from each replica's
frames.csv ((0,5] ns, 500 frames, 100 per block); they must agree with
reports/brd4_{round1,h0.4}_replicas.csv to the 3 decimals stored there.  Point statistics,
candidate bootstrap and exact p reuse scripts/rank_hou_sampling.py; the sampling bootstrap is its
replica + circular-block scheme for 3 replicas x 5 one-ns blocks (block 1 or 5 ns).
Writes replicas_<grid>.csv, samples_<grid>.csv, pairs_<grid>.csv, statistics.json and
score_vs_experiment.{png,pdf} in --out (default reports/brd4_statistics_20261101/), which must hold
protocol.json and label_audit.csv.  A protocol "sample_sources" entry {sample: {package, model}}
reads that sample's frames.csv from <package>/scores/<grid>/brd4/<sample>/<model>/ instead
(BRD4-5 alprazolam re-prep, docs/BRD4_5_REPREP_PLAN_20261101.md §6); its replica means are checked
against that replica's summary.json.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rank_hou_sampling import correlations, exact_statistics, interval  # noqa: E402
from smallmol_records import file_identity  # noqa: E402

OUT = P / "reports/brd4_statistics_20261101"
SAMPLES = [f"BRD4-{i}" for i in range(1, 11)]
GRIDS = {"default": ("scores/default", "reports/brd4_round1_replicas.csv"),
         "h0.4": ("scores/h0.4", "reports/brd4_h0.4_replicas.csv")}
DRAWS, SEED = 10000, 20261101


def replica_dir(grid_dir, sample, r, sources):
    src = sources.get(sample)
    if src:
        return P / src["package"] / grid_dir / f"brd4/{sample}/{src['model']}/rep{r + 1:02d}"
    return P / grid_dir / f"brd4/{sample}/mobley_amber/rep{r + 1:02d}"


def blocks_of(grid_dir, sources):
    """[10 samples, 3 replicas, 5 blocks] of 1-ns block means, plus replica means."""
    blocks = np.empty((len(SAMPLES), 3, 5))
    for s, sample in enumerate(SAMPLES):
        for r in range(3):
            rows = list(csv.DictReader((replica_dir(grid_dir, sample, r, sources) / "frames.csv").open()))
            t = np.array([float(x["time_ps"]) for x in rows])
            g = np.array([float(x["delta_g_mmpbsa"]) for x in rows])
            if (len(rows) != 500 or any(x["flag"] for x in rows) or any(x["converged"] != "True" for x in rows)
                    or not np.isfinite(g).all()):
                raise RuntimeError(f"{grid_dir} {sample} rep{r + 1}: expected 500 valid frames")
            for b in range(5):
                m = (t > b * 1000 + 1e-6) & (t <= (b + 1) * 1000 + 1e-6)
                if m.sum() != 100:
                    raise RuntimeError(f"{sample} rep{r + 1} block {b}: {m.sum()} frames")
                blocks[s, r, b] = g[m].mean()
    return blocks


def sampling_draws(blocks, rng, block_ns):
    """rank_hou_sampling.sampling_draws for 3 replicas x 5 one-ns blocks."""
    out = np.empty((DRAWS, blocks.shape[0]))
    nr, nb = blocks.shape[1:]
    for s in range(blocks.shape[0]):
        reps = rng.integers(0, nr, size=(DRAWS, nr))
        starts = rng.integers(0, nb, size=(DRAWS, nr, nb // block_ns))
        idx = (starts[..., None] + np.arange(block_ns)) % nb
        out[:, s] = blocks[s][reps[..., None, None], idx].mean(axis=(1, 2, 3))
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    OUT = ap.parse_args().out.resolve()
    proto = json.loads((OUT / "protocol.json").read_text())
    sources = proto.get("sample_sources", {})
    labels = {x["sample"]: x for x in csv.DictReader((OUT / "label_audit.csv").open())}
    exp = {s: float(x["exp_dg"]) for s, x in labels.items() if x["exp_dg"]}
    rng = np.random.default_rng(SEED)
    stats, grid_means = {}, {}
    for grid, (gdir, summary) in GRIDS.items():
        blocks = blocks_of(gdir, sources)
        reps = blocks.mean(axis=2)
        old = {(x["sample"], int(x["replica"])): float(x["dG_mmpbsa_mean"]) for x in csv.DictReader((P / summary).open())}
        for sample in sources:
            for r in range(3):
                old[(sample, r + 1)] = json.loads(
                    (replica_dir(gdir, sample, r, sources) / "summary.json").read_text())["delta_g_mmpbsa_mean"]
        for s, sample in enumerate(SAMPLES):
            for r in range(3):
                if abs(reps[s, r] - old[(sample, r + 1)]) > 6e-4:
                    raise RuntimeError(f"{grid} {sample} rep{r + 1}: frames mean {reps[s, r]} vs {summary}")
        means, se = reps.mean(axis=1), reps.std(axis=1, ddof=1) / math.sqrt(3)
        grid_means[grid] = (reps, means, se)
        draws = {b: sampling_draws(blocks, rng, b) for b in (1, 5)}
        write_csv(OUT / f"replicas_{grid}.csv", [
            dict(sample=sample, replica=r + 1, mean=reps[s, r], **{f"block_{b}-{b + 1}ns": blocks[s, r, b] for b in range(5)})
            for s, sample in enumerate(SAMPLES) for r in range(3)])
        write_csv(OUT / f"samples_{grid}.csv", [
            dict(sample=sample, score=means[s], replica_se=se[s], replica_range=np.ptp(reps[s]),
                 experimental_dg=exp.get(sample, ""), assay=labels[sample]["assay"],
                 label=labels[sample]["maintainer_label"], exact_label_use=labels[sample]["use_in_exact_label_statistics"])
            for s, sample in enumerate(SAMPLES)])

        gstats = {}
        for name, spec in proto["groups"].items():
            ids = np.array([SAMPLES.index(x) for x in spec["samples"]])
            y = np.array([exp[x] for x in spec["samples"]])
            g = dict(samples=spec["samples"], role=spec["role"], n=len(ids), **exact_statistics(means[ids], y))
            sign = [(means[a] - means[b]) * (exp[SAMPLES[a]] - exp[SAMPLES[b]]) for a, b in itertools.combinations(ids, 2)]
            g.update(concordant_pairs=int(sum(v > 0 for v in sign)), discordant_pairs=int(sum(v < 0 for v in sign)),
                     tied_pairs=int(sum(v == 0 for v in sign)), sampling_bootstrap={})
            for b, v in draws.items():
                rho, tau = correlations(v[:, ids], y)
                g["sampling_bootstrap"][f"{b}ns"] = dict(spearman=interval(rho), kendall=interval(tau))
            cand = rng.integers(0, len(ids), size=(DRAWS, len(ids)))
            rho, tau = correlations(means[ids][cand], y[cand])
            g["candidate_bootstrap"] = dict(spearman=interval(rho), kendall=interval(tau))
            gstats[name] = g
        stats[grid] = gstats

        pairs = []
        quant = [x for x in SAMPLES if x in exp]
        listed = {tuple(p[:2]): p[2] for k in ("same_source", "alphascreen_cross_paper") for p in proto["pairs"][k]}
        for a, b in itertools.combinations(quant, 2):
            ia, ib = SAMPLES.index(a), SAMPLES.index(b)
            pairs.append(dict(
                sample_a=a, sample_b=b, kind=listed.get((a, b), listed.get((b, a), "cross-source")),
                involves_identity_mismatch=any(labels[x]["chemical_identity_check"].startswith("MISMATCH") for x in (a, b)), assays=f"{labels[a]['assay']}/{labels[b]['assay']}",
                experiment_stronger=a if exp[a] < exp[b] else b, exp_a_minus_b=exp[a] - exp[b],
                score_stronger=a if means[ia] < means[ib] else b, score_a_minus_b=means[ia] - means[ib],
                concordant=bool((means[ia] - means[ib]) * (exp[a] - exp[b]) > 0),
                combined_replica_se=math.hypot(se[ia], se[ib]),
                bootstrap_fraction_a_stronger_1ns=float(np.mean(draws[1][:, ia] < draws[1][:, ib]))))
        for a, b, note in proto["pairs"]["censored_same_assay"]:
            ia, ib = SAMPLES.index(a), SAMPLES.index(b)
            pairs.append(dict(sample_a=a, sample_b=b, kind=note, involves_identity_mismatch=False,
                              assays="AlphaScreen/AlphaScreen", experiment_stronger=a, exp_a_minus_b="",
                              score_stronger=a if means[ia] < means[ib] else b, score_a_minus_b=means[ia] - means[ib],
                              concordant=bool(means[ia] < means[ib]), combined_replica_se=math.hypot(se[ia], se[ib]),
                              bootstrap_fraction_a_stronger_1ns=float(np.mean(draws[1][:, ia] < draws[1][:, ib]))))
        write_csv(OUT / f"pairs_{grid}.csv", pairs)

    srcs = [OUT / "protocol.json", OUT / "label_audit.csv", P / "reports/brd4_round1_replicas.csv",
            P / "reports/brd4_h0.4_replicas.csv", Path(__file__)]
    srcs += [replica_dir(g, s, r, sources) / "frames.csv" for g, _ in GRIDS.values() for s in sources for r in range(3)]
    (OUT / "statistics.json").write_text(json.dumps(dict(
        draws=DRAWS, seed=SEED, groups=stats, sources=[file_identity(x) for x in srcs]), indent=1, allow_nan=False) + "\n")
    figure(grid_means, exp, labels, sources)
    for grid, gs in stats.items():
        for name, g in gs.items():
            bs = g["sampling_bootstrap"]["1ns"]["spearman"]["percentile_2_5_97_5"]
            print(f"{grid:7s} {name:46s} n={g['n']} rho={g['spearman_rho']:.3f} tau={g['kendall_tau_b']:.3f} "
                  f"pairs {g['concordant_pairs']}/{g['discordant_pairs']}/{g['tied_pairs']} p={g['spearman_p_exact_two_sided']:.3f} "
                  f"MD-boot rho [{bs[0]:.2f},{bs[1]:.2f}]")


def figure(grid_means, exp, labels, sources):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {"ITC": "#2a78d6", "AlphaScreen": "#eb6834"}
    marker = {"ITC": "o", "AlphaScreen": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, (grid, (reps, means, se)) in zip(axes, grid_means.items()):
        for s, sample in enumerate(SAMPLES):
            if sample not in exp:
                continue
            a = labels[sample]["assay"]
            mismatch = labels[sample]["chemical_identity_check"].startswith("MISMATCH")
            c = "#8a8a85" if mismatch else color[a]
            ax.scatter(reps[s], [exp[sample]] * 3, s=14, color=c, alpha=.45, linewidths=0, marker=marker[a])
            ax.errorbar(means[s], exp[sample], xerr=se[s], fmt=marker[a], ms=8, color=c,
                        mfc="white" if mismatch else c, mec=c, elinewidth=1.5, capsize=0)
            ax.annotate(sample.replace("BRD4-", "") + (" (identity mismatch)" if mismatch else "")
                        + (" (alprazolam re-prep)" if sample in sources else ""),
                        (means[s], exp[sample]), xytext=(6, 4), textcoords="offset points", fontsize=8, color="#444441")
        ax.set_title(f"{grid} grid, 3 x (0,5] ns", fontsize=10)
        ax.set_xlabel("MM/PBSA score (kcal/mol, no entropy)")
        ax.grid(color="#e6e5e0", lw=.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("experimental dG (kcal/mol)")
    handles = [plt.Line2D([], [], marker=marker[k], ls="", color=v, ms=7, label=k) for k, v in color.items()]
    if any(labels[s]["chemical_identity_check"].startswith("MISMATCH") for s in exp):
        handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#8a8a85", ms=7, label="BRD4-5 (excluded)"))
    axes[1].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.01, "Points: replica means; markers: sample mean +/- replica SE. BRD4-1/2 have no exact label.",
             fontsize=7, color="#6b6a65")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"score_vs_experiment.{ext}", dpi=200)


if __name__ == "__main__":
    main()
