#!/usr/bin/env python3
"""B1 minimisation sensitivity: official B1 vs a second minimisation of the same 17 starts; read-only.

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-10-17_small_molecule_benchmark/scripts/compare_b1_repeat.py

docs/P1_P2_SESSION_HANDOFF_20261101.md §6.  Official B1: reports/baseline_20261101_v2 inputs
(start_scores.csv, BRD4-5 from brd4_5_alprazolam_20261101).  Repeat: b1_minimisation_repeat_20261101
(b1_repeat.sh, PBS 22221, yayoi37 RTX 5080; setup minimisation redone, same frozen default scoring).
Per sample the difference; per P1 v2 group (protocol.json) the exact statistics of both B1 versions
against experiment (rank_hou_sampling.exact_statistics); per pair whether the two minimisations
order the pair the same way and whether B2 still reverses B1.  Writes
reports/baseline_20261101_v2/b1_repeat_{samples,groups,pairs}.csv.
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P.parents[1] / "scripts"))
from rank_hou_sampling import exact_statistics  # noqa: E402

OUT = P / "reports/baseline_20261101_v2"
REP = P / "b1_minimisation_repeat_20261101/scores/start_default"
MODEL = {"brd4": "mobley_amber", "hou": "ph74_gaff2"}


def first(path):
    return next(csv.DictReader(path.open()))


def write(name, rows):
    with (OUT / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    proto = json.loads((OUT / "protocol.json").read_text())
    src = proto["sample_sources"]
    off = {x["sample"]: (x["target"], float(x["delta_g_mmpbsa"])) for x in csv.DictReader((OUT / "start_scores.csv").open())}
    for s, v in src.items():
        off[s] = (off[s][0], float(first(P / v["package"] / f"scores/start_default/brd4/{s}/{v['model']}/frames.csv")["delta_g_mmpbsa"]))
    rep = {}
    for s, (t, _) in off.items():
        row = first(REP / t / s / (src[s]["model"] if s in src else MODEL[t]) / "frames.csv")
        if row["flag"]:
            raise RuntimeError(f"{s}: repeat start score flagged ({row['flag']})")
        rep[s] = float(row["delta_g_mmpbsa"])
    write("b1_repeat_samples.csv", [dict(sample=s, target=t, b1_official=v, b1_repeat=rep[s], repeat_minus_official=rep[s] - v)
                                    for s, (t, v) in off.items()])
    b2 = {(x["group"], x["sample_i"], x["sample_j"]): x for x in csv.DictReader((OUT / "pairs.csv").open())}
    exp = {x["sample"]: float(x["experimental_dg"]) for x in csv.DictReader((OUT / "descriptors.csv").open()) if x["experimental_dg"]}
    groups, pairs = [], []
    for g, spec in proto["groups"].items():
        if "samples" not in spec:
            continue
        ss = spec["samples"]
        y = np.array([exp[s] for s in ss])
        for name, sc in (("B1_official", {s: off[s][1] for s in ss}), ("B1_repeat", rep)):
            x = np.array([sc[s] for s in ss])
            prod = [np.sign(sc[a] - sc[b]) * np.sign(exp[a] - exp[b]) for a, b in itertools.combinations(ss, 2)]
            groups.append(dict(group=g, method=name, n=len(ss), **exact_statistics(x, y),
                               concordant_pairs=sum(v > 0 for v in prod), discordant_pairs=sum(v < 0 for v in prod)))
        for a, b in itertools.combinations(ss, 2):
            d_off, d_rep = off[a][1] - off[b][1], rep[a] - rep[b]
            row = b2[(g, a, b)]
            d_b2 = float(row["B2_md_default_diff"])
            pairs.append(dict(group=g, sample_i=a, sample_j=b, exp_diff=exp[a] - exp[b], b1_official_diff=d_off,
                              b1_repeat_diff=d_rep, b1_same_order=bool(np.sign(d_off) == np.sign(d_rep)),
                              b2_diff=d_b2, b2_reverses_b1_official=bool(np.sign(d_b2) != np.sign(d_off)),
                              b2_reverses_b1_repeat=bool(np.sign(d_b2) != np.sign(d_rep))))
    write("b1_repeat_groups.csv", groups)
    write("b1_repeat_pairs.csv", pairs)
    for r in groups:
        print(f"{r['group']:36s} {r['method']:12s} rho={r['spearman_rho']:.3f} pairs {r['concordant_pairs']}/{r['discordant_pairs']} p={r['spearman_p_exact_two_sided']:.3f}")
    for r in pairs:
        if not r["b1_same_order"] or r["b2_reverses_b1_official"] != r["b2_reverses_b1_repeat"]:
            print(r)


if __name__ == "__main__":
    main()
