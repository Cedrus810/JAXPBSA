#!/usr/bin/env python3
"""R4 experimental comparison, exactly as frozen in r4_protocol.md §8 (2026-11-08, before labels).

Endpoints: Spearman rho(-score, pKd) (average ranks), Kendall tau-b, pairwise concordance over
pairs with distinct pKd.  Hierarchical bootstrap 20000 draws, seed 20261108: per sample draw 2
starts with replacement, then 3 replicas with replacement inside each drawn start; sample score
= mean of the 6 drawn replica values.  Start-only baselines resample starts only.
Sets: all8, l10a_double5 (T1L, A4L, N8A, S11M, P12Q), no_p12q7.  Fine-grid sensitivity on N1
frames: 7 samples (no P12Q) and l10a_double4.
Writes stats/experiment_endpoints.csv, stats/experiment_samples.csv, stats/experiment_meta.json.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as ss

R4 = Path(__file__).resolve().parents[1]
ROOT = R4.parents[1]
P = ROOT / "experiments/2026-10-11_public_data_execution_packet"
SAMPLES = ["PMI", "PMI-L10A", "T1L_L10A", "A4L_L10A", "N8A_L10A", "S11M_L10A", "P12Q_L10A", "PMI-M4"]
SETS = {"all8": SAMPLES, "l10a_double5": ["T1L_L10A", "A4L_L10A", "N8A_L10A", "S11M_L10A", "P12Q_L10A"],
        "no_p12q7": [s for s in SAMPLES if s != "P12Q_L10A"]}
SETS_N1 = {"no_p12q7": SETS["no_p12q7"], "l10a_double4": ["T1L_L10A", "A4L_L10A", "N8A_L10A", "S11M_L10A"]}
MODELS, REPS = ("v21m1", "v21m2"), ("rep01", "rep02", "rep03")
N_BOOT, SEED = 20000, 20261108


def endpoints(neg_score, pkd):
    x, y = np.asarray(neg_score, float), np.asarray(pkd, float)
    rho = ss.spearmanr(x, y).statistic
    tau = ss.kendalltau(x, y, variant="b").statistic
    agree = tot = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if y[i] == y[j]:
                continue
            tot += 1
            agree += (x[i] - x[j]) * (y[i] - y[j]) > 0
    return float(rho), float(tau), agree / tot


def boot(values, pkd, samples, sign, start_only=False):
    """values[sample][model] = list of replica values (or one start value when start_only)."""
    rng = np.random.default_rng(SEED)
    out = np.empty((N_BOOT, 3))
    for b in range(N_BOOT):
        sc = []
        for s in samples:
            ms = rng.integers(0, 2, 2)
            if start_only:
                sc.append(np.mean([values[s][MODELS[k]][0] for k in ms]))
            else:
                v = []
                for k in ms:
                    reps = values[s][MODELS[k]]
                    v += [reps[i] for i in rng.integers(0, len(reps), len(reps))]
                sc.append(np.mean(v))
        out[b] = endpoints(sign * np.asarray(sc), [pkd[s] for s in samples])
    return out


def point(values, samples):
    return [np.mean([x for m in MODELS for x in values[s][m]]) for s in samples]


def main():
    lab = {r["peptide_name"]: r for r in csv.DictReader((P / "pilot_measurements.csv").open())}
    pkd = {s: 9.0 - np.log10(float(lab[s]["value_nM"])) for s in SAMPLES}
    meth = {}
    for win, d in (("1-20", "scores_1_20"), ("1-10", "scores"), ("10-20", "scores_10_20")):
        meth[f"jaxpbsa_{win}"] = ({s: {m: [json.loads((P / d / s / m / r / "summary.json").read_text())["delta_g_mmpbsa_mean"]
                                           for r in REPS] for m in MODELS} for s in SAMPLES}, -1, False)
    am = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(str(R4 / "n2n3_amber/*/*/rep*/comparison.csv")):
        pth = Path(f); s, m, r = pth.parts[-4], pth.parts[-3], pth.parts[-2]
        for row in csv.DictReader(pth.open()):
            am["amber_pb"][s][(m, r)].append(float(row["amber_PB_TOTAL"]))
            am["amber_gb"][s][(m, r)].append(float(row["amber_GB_TOTAL"]))
    for k in ("amber_pb", "amber_gb"):
        meth[k] = ({s: {m: [float(np.mean(am[k][s][(m, r)])) for r in REPS] for m in MODELS} for s in SAMPLES}, -1, False)
    md = defaultdict(dict)
    for row in csv.DictReader((R4 / "n4_baselines/md_dsasa.csv").open()):
        md[row["sample"]].setdefault(row["model"], []).append(float(row["delta_sasa_mean"]))
    meth["md_dsasa"] = ({s: {m: md[s][m] for m in MODELS} for s in SAMPLES}, -1, False)
    n4 = {(r["sample"], r["model"]): r for r in csv.DictReader((R4 / "n4_baselines/start_baselines.csv").open())}
    meth["start_score"] = ({s: {m: [float(n4[(s, m)]["delta_g_mmpbsa"])] for m in MODELS} for s in SAMPLES}, -1, True)
    meth["start_dsasa"] = ({s: {m: [float(n4[(s, m)]["delta_sasa"])] for m in MODELS} for s in SAMPLES}, -1, True)
    meth["start_contacts"] = ({s: {m: [float(n4[(s, m)]["contacts_4A_heavy"])] for m in MODELS} for s in SAMPLES}, +1, True)
    # fine-grid sensitivity on N1 frames: replica value = mean over its 19 frames
    n1 = {c: defaultdict(dict) for c in ("default", "h0.4", "h0.35")}
    for f in glob.glob(str(R4 / "n1_grid/*/*/rep*/sensitivity.csv")):
        pth = Path(f); s, m, r = pth.parts[-4], pth.parts[-3], pth.parts[-2]
        vals = defaultdict(list)
        for row in csv.DictReader(pth.open()):
            if row["config"] in n1 and row["variant"] == "0":
                vals[row["config"]].append(float(row["delta_g_mmpbsa"]))
        for c in n1:
            n1[c][s].setdefault(m, []).append(float(np.mean(vals[c])))
    rows = []
    def add(name, spec, sets):
        values, sign, start_only = spec
        for sn, samples in sets.items():
            pe = endpoints(sign * np.asarray(point(values, samples)), [pkd[s] for s in samples])
            bt = boot(values, pkd, samples, sign, start_only)
            q = np.percentile(bt, [2.5, 50, 97.5], axis=0)
            for i, e in enumerate(("spearman_rho", "kendall_tau_b", "pair_concordance")):
                rows.append(dict(method=name, set=sn, n=len(samples), endpoint=e, point=round(pe[i], 3),
                                 boot_median=round(q[1, i], 3), ci2_5=round(q[0, i], 3), ci97_5=round(q[2, i], 3),
                                 resample="starts only" if start_only else "starts then replicas"))
    for name, spec in meth.items():
        add(name, spec, SETS)
    for c in n1:
        add(f"n1frames_{c}", ({s: n1[c][s] for s in SETS_N1["no_p12q7"]}, -1, False), SETS_N1)
    (R4 / "stats").mkdir(exist_ok=True)
    with (R4 / "stats/experiment_endpoints.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    srows = [dict(sample=s, Kd_nM=float(lab[s]["value_nM"]), pKd=round(pkd[s], 3),
                  **{k: round(float(np.mean([x for m in MODELS for x in v[0][s][m]])), 3) for k, v in meth.items()})
             for s in SAMPLES]
    with (R4 / "stats/experiment_samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)
    (R4 / "stats/experiment_meta.json").write_text(json.dumps(dict(
        protocol="r4_protocol.md §8 (frozen 2026-11-08 before labels)", n_boot=N_BOOT, seed=SEED,
        pkd="9 - log10(Kd/nM)", sets=SETS, sets_n1=SETS_N1, scipy=ss.__name__), indent=2) + "\n")
    print(json.dumps([r for r in rows if r["endpoint"] == "spearman_rho"], indent=0)[:6000])


if __name__ == "__main__":
    main()
