#!/usr/bin/env python3
"""Build manuscript figures and tables from archived reports, without MD/PBSA.

Run from any directory. --reports can point to the reports in the review bundle.
Every plotted data panel has an adjacent CSV; historical frame sets stay separate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "experiments/2026-10-17_small_molecule_benchmark/reports"
BLUE, ORANGE, RED = "#25618d", "#dc8b32", "#b94c53"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def check_inputs(reports):
    """Cross-check plotted means, rank direction and flags against separate reports."""
    samples = read_csv(reports / "hou_ranking_20261030/samples.csv")
    replicas = read_csv(reports / "hou_sampling_20261030_verified/replicas.csv")
    if len(samples) != 7 or len(replicas) != 42 or len({r["run_id"] for r in replicas}) != 42:
        raise ValueError("Hou requires seven samples and 42 replicas")
    if sum(int(r["n_frames"]) for r in replicas) != 42000:
        raise ValueError("incomplete Hou coverage")
    if sum(int(r["online_flagged"]) for r in replicas) != 97:
        raise ValueError("original online flags must be retained")
    if sum(r["recovered_offline"] == "True" for r in replicas) != 3:
        raise ValueError("expected three full-replica recoveries")
    for s in samples:
        values = [float(r["mean"]) for r in replicas if r["sample"] == s["sample"]]
        if len(values) != 6 or not np.isclose(np.mean(values), float(s["score"]), atol=1e-10, rtol=0):
            raise ValueError(f"Hou replica mean mismatch: {s['sample']}")
        if not np.isclose(np.std(values, ddof=1) / np.sqrt(6), float(s["replica_se"]), atol=1e-10, rtol=0):
            raise ValueError(f"Hou SE mismatch: {s['sample']}")
    rho = spearmanr([float(s["score"]) for s in samples],
                    [float(s["experimental_dg"]) for s in samples]).statistic
    if not np.isclose(rho, .75):
        raise ValueError("Hou rank direction or values changed")
    for grid in ("default", "h0.4"):
        ss = read_csv(reports / f"brd4_statistics_20261101_v2/samples_{grid}.csv")
        rr = read_csv(reports / f"brd4_statistics_20261101_v2/replicas_{grid}.csv")
        if len(ss) != 10 or len(rr) != 30:
            raise ValueError("BRD4 v2 requires ten samples and 30 replicas")
        for s in ss:
            vals = [float(r["mean"]) for r in rr if r["sample"] == s["sample"]]
            if len(vals) != 3 or not np.isclose(np.mean(vals), float(s["score"]), atol=1e-10, rtol=0):
                raise ValueError(f"BRD4 v2 mean mismatch: {s['sample']}")
        br5 = next(s for s in ss if s["sample"] == "BRD4-5")
        if not (-21 < float(br5["score"]) < -20):
            raise ValueError("BRD4-5 must use the corrected alprazolam run")
        primary = [s for s in ss if s["sample"] in {f"BRD4-{i}" for i in range(5,10)}]
        rho_brd4 = spearmanr([float(s["score"]) for s in primary],
                              [float(s["experimental_dg"]) for s in primary]).statistic
        if not np.isclose(rho_brd4, .4 if grid == "default" else .3):
            raise ValueError("BRD4 v2 primary-group rank mismatch")
    return {"hou_samples": 7, "hou_replicas": 42, "frames": 42000,
            "original_online_flags": 97, "recovered_replicas": 3,
            "hou_spearman": float(rho), "brd4_v2_samples": 10,
            "checks": "replica means/SE, corrected input, rank direction, coverage"}


def workflow(out):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set(xlim=(0, 11), ylim=(0, 5)); ax.axis("off")
    boxes = [(0.2, 3.0, 2.4, "OpenMM MD\nperiodic snapshots\ncoordinates + step + time"),
             (3.2, 3.0, 3.6, "Same GPU: analysis\nsync callback OR bounded async queue\noptional MPS worker cap"),
             (7.4, 3.0, 3.2, "Per-frame CSV\nMM + PB(C − R − L) + SA\nmargin / convergence / finite flags"),
             (1.0, .7, 3.6, "Run-script layer\ninput / protocol identity\ncheckpoint + committed chunk clock"),
             (6.1, .7, 3.8, "Audit and recovery\nsubmitted steps vs scored rows\nretain flags; record offline replacement")]
    for x, y, width, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, 1.3, boxstyle="round,pad=0.15",
                                   facecolor="#edf3f7", edgecolor=BLUE))
        ax.text(x + width / 2, y + .65, text, ha="center", va="center", fontsize=10)
    for a, b in [((2.8, 3.65), (3.0, 3.65)), ((7.0, 3.65), (7.2, 3.65)),
                 ((4.8, 1.35), (5.9, 1.35)), ((9, 2.8), (9, 2.2))]:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2})
    ax.text(5.5, 4.75, "Single-trajectory endpoint score; entropy omitted", ha="center", fontsize=12)
    save(fig, out, "fig1_workflow")


def numerical(reports, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    data = []
    for k, (target, file, color) in enumerate([
        ("BRD4 historical", "reference_q3_frames.csv", BLUE),
        ("Hou q3", "reference_q3_hou_frames.csv", ORANGE)]):
        rows = read_csv(reports / file)
        for i, (key, label) in enumerate([("d_eel", "Coulomb"), ("d_vdw", "LJ"), ("d_sa", "SA")]):
            vals = [float(r[key]) for r in rows]
            axes[0].scatter(np.full(len(vals), i + (k-.5)*.22), vals, s=7,
                            alpha=.4, color=color, label=f"{target}, n={len(rows)}" if i == 0 else None)
            for r, val in zip(rows, vals):
                data.append(dict(target=target, sample=r["sample"], replica=r["replica"],
                                 frame=r["frame"], component=label, residual=val))
    axes[0].set(xticks=range(3), xticklabels=["Coulomb", "LJ", "SA"], ylabel="JAXPBSA − Amber (kcal/mol)", title="A  Same-frame residuals")
    axes[0].legend(fontsize=8)
    brd = read_csv(reports / "brd4_h04_vs_default_20261025.csv")
    axes[1].bar(range(10), [float(r["h04_minus_default"]) for r in brd], color=BLUE)
    axes[1].set(xticks=range(10), xticklabels=[r["sample"].split("-")[-1] for r in brd],
                xlabel="BRD4 ID (historical inputs, 15,000 frames)", ylabel="h0.4 − default (kcal/mol)", title="B  Historical grid sensitivity")
    acc = read_csv(reports / "hou_acceptance_samples_20261028.csv")
    for off, key, err, color, label in [(-.13, "d35_40", "se_d35_40", BLUE, "h0.35 − h0.4"),
                                       (.13, "d30_35", "se_d30_35", ORANGE, "h0.3 − h0.35")]:
        rows = [r for r in acc if r[key] != ""]
        axes[2].errorbar([acc.index(r)+off for r in rows], [float(r[key]) for r in rows],
                         yerr=[float(r[err]) for r in rows], fmt="o", color=color, capsize=3, label=label)
    axes[2].set(xticks=range(7), xticklabels=[r["sample"] for r in acc],
                ylabel="Correction (kcal/mol)", title="C  Hou bounded s25 refinement")
    axes[2].legend(fontsize=8)
    for ax in axes: ax.axhline(0, color=".5", lw=.7)
    write_csv(out / "fig2_residuals.csv", data)
    write_csv(out / "fig2_brd4_historical_grid.csv", brd)
    write_csv(out / "fig2_hou_grid.csv", acc)
    save(fig, out, "fig2_numerical")


def online(reports, out):
    cost = read_csv(reports / "hou_online_acc_cost_19854.csv")
    rows = [r for r in cost if r["scheme"] != "baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    keys = list(dict.fromkeys((r["scheme"], r["grid"]) for r in rows))
    for x, key in enumerate(keys):
        for r in rows:
            if (r["scheme"], r["grid"]) == key:
                axes[0].scatter(x + (-.08 if r["order"] == "openmm" else .08), float(r["overhead_pct"]),
                                color=BLUE if r["order"] == "openmm" else ORANGE,
                                label=r["order"] if x == 0 else None)
    axes[0].set(xticks=range(len(keys)), xticklabels=[f"{k[0]}\n{k[1]}" for k in keys],
                ylabel="MD overhead (%)", title="A  Two initialization orders")
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].legend(title="init order", fontsize=7, title_fontsize=8)
    for r in rows:
        axes[1].scatter(float(r["analyzer_warmup"]), float(r["pbsa_ms_per_report"]),
                        color=BLUE if r["grid"] == "default" else ORANGE, s=30)
    axes[1].set(xlabel="Analyzer warmup (s)", ylabel="PBSA busy time (ms/frame)",
                title="B  Cold/cache-hit starts (3090)")
    production = read_csv(reports / "hou_online_acc_production_19854.csv")
    online_rows = [r for r in production if r["online_frames"]]
    run = online_rows[0]
    if run["steps_unique"] != "True":
        raise ValueError("restart audit reports duplicate steps")
    axes[2].bar(["expected", "scored"],
                [int(float(run[k])) for k in ("expected_frames", "online_frames")], color=BLUE)
    axes[2].set(ylabel="Frames / step identities", title="C  SIGTERM + resume audit")
    axes[2].text(.5, .15, "Restart: 100/100; no gaps or duplicates\nReplay: 174/175 within 0.0038 kcal/mol\nRetained exception: a4, about 0.10 kcal/mol", transform=axes[2].transAxes,
                 ha="center", fontsize=8, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    write_csv(out / "fig3_online_cost.csv", cost)
    write_csv(out / "fig3_production.csv", production)
    write_csv(out / "fig3_replay.csv", read_csv(reports / "hou_online_acc_replay_19854.csv"))
    save(fig, out, "fig3_online")


def hou(reports, out):
    ss = read_csv(reports / "hou_ranking_20261030/samples.csv")
    rr = read_csv(reports / "hou_sampling_20261030_verified/replicas.csv")
    pairs = read_csv(reports / "hou_ranking_20261030/pairs.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for s in ss:
        x, y, se = (float(s[k]) for k in ("experimental_dg", "score", "replica_se"))
        vals = [float(r["mean"]) for r in rr if r["sample"] == s["sample"]]
        axes[0].scatter(np.full(6, x), vals, alpha=.45, s=15, color=BLUE)
        axes[0].errorbar(x, y, yerr=se, color=BLUE, fmt="o", capsize=4)
        axes[0].annotate(s["sample"], (x, y), xytext=(5, 3), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="Experimental ΔG (kcal/mol)", ylabel="MM/PBSA score (kcal/mol)", title="A  Six replicas; mean ± replica SE")
    groups = json.loads((reports / "hou_ranking_20261030/ranking.json").read_text())["groups"]
    for i, (key, label) in enumerate([("a1_a4_same_series", "a1–a4\nprimary"), ("all_seven_exploratory", "all seven\nexploratory")]):
        g = groups[key]; lo, hi = g["sampling_bootstrap"]["1"]["spearman"]["percentile_2_5_97_5"]
        val = g["spearman_rho"]
        axes[1].errorbar(i, val, yerr=[[val-lo], [hi-val]], color=BLUE, fmt="o", capsize=5)
        axes[1].text(i, lo-.13, f"n={g['n_samples']}\np={g['spearman_p_exact_two_sided']:.4f}", ha="center", fontsize=9)
    axes[1].set(xticks=[0,1], xticklabels=["a1–a4\nprimary", "all seven\nexploratory"],
                ylim=(0,1.15), ylabel="Spearman ρ", title="B  Replica + 1 ns block sensitivity")
    oriented = []
    for p in pairs:
        f = float(p["bootstrap_fraction_a_stronger_block1ns"])
        if p["experiment_stronger"] != p["sample_a"]: f = 1-f
        oriented.append(dict(pair=f"{p['sample_a']}/{p['sample_b']}", concordant=p["concordant"],
                             fraction_matching_experiment=f))
    axes[2].barh(range(21), [p["fraction_matching_experiment"] for p in oriented],
                 color=[BLUE if p["concordant"] == "True" else RED for p in oriented])
    axes[2].set(yticks=range(21), yticklabels=[p["pair"] for p in oriented], xlim=(0,1),
                xlabel="Resampling fraction matching experiment", title="C  All pairs; 1 ns blocks")
    axes[2].tick_params(axis="y", labelsize=7)
    write_csv(out / "fig4_samples.csv", ss); write_csv(out / "fig4_replicas.csv", rr)
    write_csv(out / "fig4_pairs.csv", oriented)
    save(fig, out, "fig4_hou")


def limits(reports, out):
    rr = [r for r in read_csv(reports / "hou_sampling_replicas_20261030.csv") if r["sample"] == "a6"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for r in rr:
        x, y = float(r["asp189_med_h2"]), float(r["dg_0_10"])
        axes[0].scatter(x, y, color=BLUE)
        axes[0].annotate(f"rep{r['replica']}", (x,y), xytext=(3,3), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Asp189 contact median, 5–10 ns (Å)", ylabel="Score, 0–10 ns (kcal/mol)", title="A  a6 visited states (no population claim)")
    for grid, offset, color in [("default", -.08, BLUE), ("h0.4", .08, ORANGE)]:
        ss = read_csv(reports / f"brd4_statistics_20261101_v2/samples_{grid}.csv")
        chosen = [r for r in ss if r["sample"] in [f"BRD4-{i}" for i in range(5,10)]]
        for i, r in enumerate(chosen):
            axes[1].errorbar(i+offset, float(r["score"]), yerr=float(r["replica_se"]),
                             fmt="o", capsize=3, color=color, label=grid if i==0 else None)
    axes[1].set(xticks=range(5), xticklabels=["5*","6","7","8","9"], xlabel="BRD4 ITC at 15 °C (*corrected input)",
                ylabel="Score (kcal/mol)", title="B  BRD4 v2 primary group, n=5")
    axes[1].legend(fontsize=8)
    baseline = read_csv(reports / "baseline_20261101_v2/method_comparison.csv")
    groups = [("hou_a1_a4_same_series", "Hou a1–a4"), ("hou_all_seven_exploratory", "Hou all"),
              ("brd4_itc6_cross_source_exploratory", "BRD4 ITC6")]
    for off, method, label, color in [(-.23, "B0_size", "size", ".65"), (0, "B1_start_default", "start", ORANGE), (.23, "B2_md_default", "MD", BLUE)]:
        vals = [float(next(r for r in baseline if r["group"]==g and r["method"]==method)["spearman_rho"]) for g,_ in groups]
        axes[2].bar(np.arange(3)+off, vals, .22, label=label, color=color)
    axes[2].set(xticks=range(3), xticklabels=[l for _,l in groups], ylim=(-.6,1.2),
                ylabel="Spearman ρ", title="C  Baselines; BRD4 cross-source exploratory")
    axes[2].tick_params(axis="x", labelsize=8); axes[2].legend(fontsize=8)
    write_csv(out / "fig5_a6.csv", rr)
    write_csv(out / "table_baselines_v2.csv", baseline)
    save(fig, out, "fig5_limits")


def supplement(reports, out):
    for name, source in {
        "table_labels_verified.csv": "brd4_statistics_20261101_v2/label_verification_20261102.csv",
        "table_brd4_samples_default.csv": "brd4_statistics_20261101_v2/samples_default.csv",
        "table_brd4_samples_h04.csv": "brd4_statistics_20261101_v2/samples_h0.4.csv",
        "table_b1_repeat.csv": "baseline_20261101_v2/b1_repeat_samples.csv",
        "table_cost_md.csv": "evidence_20261102/cost_md.csv",
        "table_cost_prep.csv": "evidence_20261102/cost_prep.csv",
        "table_cost_setup.csv": "evidence_20261102/cost_setup.csv",
        "table_cost_scoring.csv": "evidence_20261102/cost_scoring.csv",
        "table_cost_online.csv": "evidence_20261102/cost_online.csv",
    }.items():
        write_csv(out / name, read_csv(reports / source))
    stats = json.loads((reports / "brd4_statistics_20261101_v2/statistics.json").read_text())
    rows = []
    for grid, groups in stats["groups"].items():
        for group, g in groups.items():
            rows.append(dict(grid=grid, group=group, samples=" ".join(g["samples"]),
                             n=g["n"], rho=g["spearman_rho"], tau_b=g["kendall_tau_b"],
                             exact_p=g["spearman_p_exact_two_sided"],
                             concordant=g["concordant_pairs"], discordant=g["discordant_pairs"]))
    write_csv(out / "table_brd4_statistics_v2.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    checks = check_inputs(args.reports)
    workflow(args.out); numerical(args.reports, args.out); online(args.reports, args.out)
    hou(args.reports, args.out); limits(args.reports, args.out)
    supplement(args.reports, args.out)
    write_csv(args.out / "table_S_replicas.csv", read_csv(args.reports / "hou_sampling_20261030_verified/replicas.csv"))
    (args.out / "validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    (args.out / "README.md").write_text(
        "# Manuscript figures\n\nGenerated from archived reports with `scripts/build_paper_assets.py`.\n\n"
        "Figures 1–5: PDF (vector) and PNG. Adjacent CSVs contain plotted data.\n"
        "Figure 2B is explicitly historical (original BRD4-5); final application statistics use v2.\n"
        "Figure 2C uses the bounded s25 frame set, separate from final 6 × 10 ns sampling.\n"
        "Figure 3 points are initialization orders, not uncertainty intervals.\n"
        "Figure 4 error bars are replica SE; rank intervals condition on visited states.\n"
        "Figure 5 uses corrected alprazolam and does not infer equilibrium state populations.\n"
        "`table_S_replicas.csv` is the full 42-replica coverage/recovery supplement.\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
