#!/usr/bin/env python3
"""Compare the C-terminal start control v21mx1 with v21m1/v21m2 (read-only).

Quantities (all defined before looking at the new results, same as R3 / packet protocol §2.1):
  score     replica mean of delta_g_mmpbsa over 1-20 ns (summary.json, every 10-ps frame); the
            replica is the statistical unit.  Old: 2026-10-11 packet scores_1_20 (v21m1, v21m2);
            new: this packet scores_1_20 (v21mx1).
  structure per peptide residue, CA distance (A) in the 3EQS receptor frame (receptor CA 27-108)
            to 3EQS (B1-11), to the nearest Pro12-bearing 3LNZ copy (H/L) and to 5UMM chain B;
            median over the 20 one-ns production checkpoints of each replica.  Gln12 rotamer bins
            vs 5UMM for P12Q_L10A.  Old-model values are recomputed here with the same code.
numpy + text only.  Output: results/score_compare.csv, results/structure_compare.csv,
results/rotamer_compare.csv.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
NEW = HERE.parents[1]
ROOT = HERE.parents[3]
OLD = ROOT / "experiments/2026-10-11_public_data_execution_packet"
sys.path.insert(0, str(ROOT / "experiments/2026-11-06_ppi_structure_review/scripts"))
import stage_trace as T                                              # noqa: E402
import r3_pose_evidence as R                                         # noqa: E402

OUT = NEW / "results"
SAMPLES = ("PMI", "P12Q_L10A")
MODELS = (("v21m1", OLD), ("v21m2", OLD), ("v21mx1", NEW))


def score_rows():
    rows = []
    for s in SAMPLES:
        for m, pk in MODELS:
            for r in (1, 2, 3):
                d = json.loads((pk / "scores_1_20" / s / m / f"rep{r:02d}" / "summary.json").read_text())
                rows.append(dict(sample=s, model=m, replica=r, n_valid=d["n_valid"], n_flagged=d["n_flagged"],
                                 dG=round(d["delta_g_mmpbsa_mean"], 3),
                                 **{k: round(v, 3) for k, v in d["components_mean"].items()}))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    sc = score_rows()
    summ = []
    for s in SAMPLES:
        grp = {m: [x["dG"] for x in sc if x["sample"] == s and x["model"] == m] for m, _ in MODELS}
        old = grp["v21m1"] + grp["v21m2"]
        new = grp["v21mx1"]
        se = lambda v: st.stdev(v) / len(v) ** 0.5
        summ.append(dict(sample=s,
                         v21m1_mean=round(st.mean(grp["v21m1"]), 3), v21m2_mean=round(st.mean(grp["v21m2"]), 3),
                         old6_mean=round(st.mean(old), 3), old6_sd=round(st.stdev(old), 3), old6_se=round(se(old), 3),
                         v21mx1_mean=round(st.mean(new), 3), v21mx1_sd=round(st.stdev(new), 3), v21mx1_se=round(se(new), 3),
                         mx1_minus_old6=round(st.mean(new) - st.mean(old), 3),
                         se_of_difference=round((se(old) ** 2 + se(new) ** 2) ** 0.5, 3),
                         m1_minus_m2_old=round(st.mean(grp["v21m1"]) - st.mean(grp["v21m2"]), 3),
                         **{f"{c}_old6": round(st.mean(x[c] for x in sc if x["sample"] == s and x["model"] != "v21mx1"), 2)
                            for c in ("e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa")},
                         **{f"{c}_mx1": round(st.mean(x[c] for x in sc if x["sample"] == s and x["model"] == "v21mx1"), 2)
                            for c in ("e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa")}))
    refs = {e: [(lab, R.frame(st_)) for lab, st_ in R.read_pairs(e)] for e in ("3EQS", "3LNZ", "5UMM")}
    ref12 = {"3LNZ": [(l, x) for l, x in refs["3LNZ"] if ("B", 12) in x.res], "5UMM": [(l, x) for l, x in refs["5UMM"] if ("B", 12) in x.res]}
    srow, rot = [], []
    for s in SAMPLES:
        for m, pk in MODELS:
            prepared = T.read_pdb(pk / "prepared" / s / m / "complex_dry.pdb")
            for r in (1, 2, 3):
                rd = pk / "md" / s / m / f"rep{r:02d}"
                states = [("prepared", prepared)] + [
                    (f"{c + 1:02d}ns", T.read_state(rd / "prod" / f"chunk_{c:04d}_state.xml", prepared)) for c in range(20)]
                per = {}
                for name, x in states:
                    x = R.frame(T.image_peptide(x)[0])
                    d3 = R.per_residue(x, refs["3EQS"][0][1])
                    vals = {f"CA{i}_3eqs": d3[i]["CA"] for i in range(1, 12)}
                    for e, lst in ref12.items():
                        vals[f"CA12_{e}"] = min(R.per_residue(x, ref)[12]["CA"] for _, ref in lst)
                    if s == "P12Q_L10A":
                        d5 = R.per_residue(x, ref12["5UMM"][0][1])[12]
                        rot.append(dict(sample=s, model=m, replica=r, stage=name,
                                        chi=";".join("" if a is None else f"{a:.0f}" for a in d5["chi"]),
                                        bins=";".join(str(R.rot_bin(a)) for a in d5["chi"]),
                                        bins_ref=";".join(str(R.rot_bin(a)) for a in d5["chi_ref"]),
                                        sc_local_rmsd=round(d5.get("sc_local_rmsd", float("nan")), 2)))
                    per[name] = vals
                md = [v for k, v in per.items() if k != "prepared"]
                srow.append(dict(sample=s, model=m, replica=r,
                                 **{f"start_{k}": round(v, 2) for k, v in per["prepared"].items()},
                                 **{f"med_{k}": round(st.median(v[k] for v in md), 2) for k in md[0]}))
            print(s, m, flush=True)
    for name, rows in (("score_compare.csv", sc), ("score_summary.csv", summ), ("structure_compare.csv", srow),
                       ("rotamer_compare.csv", rot)):
        with (OUT / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
