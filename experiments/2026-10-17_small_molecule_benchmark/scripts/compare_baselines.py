#!/usr/bin/env python3
"""P1.1 B0/B1/B2 baseline comparison against experimental order; read-only, no molecular computation.

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-10-17_small_molecule_benchmark/scripts/compare_baselines.py

Groups, methods, descriptor options and statistics are fixed in reports/baseline_20261101/protocol.json
(written before this script was run).  Correlations and exact permutation p reuse
scripts/rank_hou_sampling.py (correlations, exact_statistics) so Hou B2 reproduces
reports/hou_ranking_20261030/ranking.json; the script stops if it does not.
Writes {descriptors,method_comparison,pairs}.csv and sources.json in --out (default
reports/baseline_20261101/, which holds protocol.json).  A protocol "sample_sources" entry
{sample: {package, model}} takes that sample's sdf (<package>/prepared/brd4/<sample>/<model>/ligand_in.sdf),
B1 (<package>/scores/start_default/...) and B2 replica means (<package>/scores/<grid>/.../frames.csv)
from the independent package (BRD4-5 alprazolam re-prep, docs/BRD4_5_REPREP_PLAN_20261101.md §6).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as D

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rank_hou_sampling import correlations, exact_statistics  # noqa: E402
from smallmol_records import file_identity  # noqa: E402

OUT = P / "reports/baseline_20261101"
METHODS = ["B0_size", "B1_start_default", "B2_md_default", "B2_md_h04_sensitivity"]


def csv_rows(path):
    return list(csv.DictReader((P / path).open()))


def replica_stats(path):
    by = {}
    for x in csv_rows(path):
        by.setdefault(x["sample"], []).append(float(x["dG_mmpbsa_mean"]))
    return {s: (st.mean(v), st.stdev(v) / len(v) ** 0.5, len(v)) for s, v in by.items()}


def descriptors(sources):
    inv = {x["sample"]: x for x in csv_rows("reports/baseline_20261101/inventory.csv")}
    out = {}
    for s, x in inv.items():
        src = sources.get(s)
        sdf = (P / src["package"] / f"prepared/brd4/{s}/{src['model']}/ligand_in.sdf" if src
               else P / f"sources/brd4/sdf/ligand-{s.split('-')[1]}.sdf" if x["target"] == "brd4"
               else P / f"prepared/hou/{s}/ph74_gaff2/ligand_in.sdf")
        m = Chem.MolFromMolFile(str(sdf), removeHs=False, sanitize=True)
        q = (float(next(csv.DictReader((P / src["package"] / "manifests/brd4_inputs.csv").open()))["ligand_charge"])
             if src else float(x["partial_charge_sum"]))
        out[s] = dict(sample=s, target=x["target"], n_heavy=m.GetNumHeavyAtoms(),
                      n_rotatable=D.CalcNumRotatableBonds(m, D.NumRotatableBondsOptions.Strict),
                      formal_charge=Chem.GetFormalCharge(m), partial_charge_sum=q,
                      score_size=-m.GetNumHeavyAtoms(), sdf=str(sdf.relative_to(P)))
    return out


def source_replicas(src, sample, grid):
    v = []
    for r in (1, 2, 3):
        rows = list(csv.DictReader((P / src["package"] / f"scores/{grid}/brd4/{sample}/{src['model']}/rep{r:02d}/frames.csv").open()))
        if len(rows) != 500 or any(x["flag"] for x in rows):
            raise RuntimeError(f"{sample} {grid} rep{r}: expected 500 valid frames")
        v.append(st.mean(float(x["delta_g_mmpbsa"]) for x in rows))
    return st.mean(v), st.stdev(v) / len(v) ** 0.5, len(v)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    OUT = ap.parse_args().out.resolve()
    proto = json.loads((OUT / "protocol.json").read_text())
    sources = proto.get("sample_sources", {})
    desc = descriptors(sources)
    exp = {x["sample"]: float(x["experimental_dg_kcal_mol"]) for x in csv_rows("manifests/brd4_measurements.csv")
           if x["experimental_dg_kcal_mol"]}
    exp.update({x["sample_id"]: float(x["experimental_dg_kcal_mol"])
                for x in csv_rows("manifests/hou2011_table_s1.csv") if x["sample_id"] in desc})
    b1 = {x["sample"]: float(x["delta_g_mmpbsa"]) for x in csv_rows("reports/baseline_20261101/start_scores.csv")
          if x["valid"] == "True"}
    b2 = replica_stats("reports/brd4_round1_replicas.csv")
    b2.update({x["sample"]: (float(x["score"]), float(x["replica_se"]), 6)
               for x in csv_rows("reports/hou_ranking_20261030/samples.csv")})
    h04 = replica_stats("reports/brd4_h0.4_replicas.csv")
    for s, src in sources.items():
        row = next(csv.DictReader((P / src["package"] / f"scores/start_default/brd4/{s}/{src['model']}/frames.csv").open()))
        if row["flag"]:
            raise RuntimeError(f"{s}: flagged start score")
        b1[s] = float(row["delta_g_mmpbsa"])
        b2[s], h04[s] = source_replicas(src, s, "default"), source_replicas(src, s, "h0.4")
    score = {"B0_size": {s: float(d["score_size"]) for s, d in desc.items()},
             "B1_start_default": b1,
             "B2_md_default": {s: v[0] for s, v in b2.items()},
             "B2_md_h04_sensitivity": {s: v[0] for s, v in h04.items()}}

    with (OUT / "descriptors.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(desc.values()))) + ["experimental_dg"])
        w.writeheader()
        w.writerows(dict(d, experimental_dg=exp.get(s, "")) for s, d in desc.items())

    comp, pairs = [], []
    for g, spec in proto["groups"].items():
        if "samples" not in spec:
            continue
        ss = spec["samples"]
        y = np.array([exp[s] for s in ss])
        cols = {m: score[m] for m in METHODS if all(s in score[m] for s in ss)}
        cols.update({f"raw_{k}": {s: float(desc[s][k]) for s in ss}
                     for k in ("n_heavy", "n_rotatable", "formal_charge", "partial_charge_sum")})
        for m, sc in cols.items():
            x = np.array([sc[s] for s in ss])
            row = dict(group=g, method=m, n=len(ss), samples=" ".join(ss))
            if np.ptp(x) == 0:
                row.update(note="no variance in group; statistics undefined")
            else:
                row.update(exact_statistics(x, y))
                i, j = np.triu_indices(len(ss), 1)
                prod = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
                row.update(concordant_pairs=int((prod > 0).sum()), discordant_pairs=int((prod < 0).sum()),
                           tied_pairs=int((prod == 0).sum()))
                if not m.startswith("raw_") and m != "B0_size":
                    rho, _ = correlations(x, np.array([score["B0_size"][s] for s in ss]))
                    row.update(spearman_rho_vs_score_size=float(rho))
            comp.append(row)

    groups = [(g, list(itertools.combinations(spec["samples"], 2))) for g, spec in proto["groups"].items()
              if "samples" in spec]
    groups.append(("brd4_same_source_pairs", [tuple(p) for p in proto["groups"]["brd4_same_source_pairs"]["pairs"]]))
    for g, pp in groups:
        for i, j in pp:
            dy = exp[i] - exp[j]
            row = dict(group=g, sample_i=i, sample_j=j, exp_i=exp[i], exp_j=exp[j], exp_diff=round(dy, 3))
            for m in METHODS:
                if i in score[m] and j in score[m]:
                    d = score[m][i] - score[m][j]
                    row[f"{m}_diff"] = round(d, 3)
                    row[f"{m}_agrees"] = "tie" if d == 0 else bool(np.sign(d) == np.sign(dy))
            se = math.hypot(b2[i][1], b2[j][1])
            row.update(B2_combined_replica_se=round(se, 3),
                       B2_abs_diff_over_se=round(abs(row["B2_md_default_diff"]) / se, 2),
                       B2_reverses_B1=bool(np.sign(row["B2_md_default_diff"]) != np.sign(row["B1_start_default_diff"])))
            pairs.append(row)

    for name, rows in (("method_comparison", comp), ("pairs", pairs)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    ref = json.loads((P / "reports/hou_ranking_20261030/ranking.json").read_text())["groups"]
    mine = {(r["group"], r["method"]): r for r in comp}
    for g_ref, g in (("a1_a4_same_series", "hou_a1_a4_same_series"), ("all_seven_exploratory", "hou_all_seven_exploratory"),
                     ("a5_a7_secondary", "hou_a5_a7_secondary")):
        a, b = ref[g_ref], mine[(g, "B2_md_default")]
        for k in ("spearman_rho", "kendall_tau_b", "spearman_p_exact_two_sided", "concordant_pairs", "discordant_pairs"):
            if not math.isclose(a[k], b[k], abs_tol=1e-12):
                raise SystemExit(f"Hou B2 does not reproduce ranking.json: {g} {k} {a[k]} vs {b[k]}")
    srcs = [str((OUT / "protocol.json").relative_to(P)), "reports/baseline_20261101/inventory.csv",
            "reports/baseline_20261101/start_scores.csv", "reports/brd4_round1_replicas.csv",
            "reports/brd4_h0.4_replicas.csv", "reports/hou_ranking_20261030/samples.csv",
            "manifests/brd4_measurements.csv", "manifests/hou2011_table_s1.csv"]
    srcs += [f"{src['package']}/scores/{d}/brd4/{s}/{src['model']}/{f}" for s, src in sources.items()
             for d, f in [("start_default", "frames.csv")] + [(g, f"rep{r:02d}/frames.csv")
                                                             for g in ("default", "h0.4") for r in (1, 2, 3)]]
    (OUT / "sources.json").write_text(json.dumps(dict(
        script=str(Path(__file__).relative_to(P)), inputs={s: file_identity(P / s) for s in srcs},
        sdf={s: file_identity(P / d["sdf"]) for s, d in desc.items()},
        hou_b2_reproduces_ranking_json=True), indent=1) + "\n")
    print("Hou B2 reproduces ranking.json")
    for r in comp:
        if not r["method"].startswith("raw_"):
            print(r["group"], r["method"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                            if k in ("spearman_rho", "kendall_tau_b", "spearman_p_exact_two_sided",
                                                     "concordant_pairs", "discordant_pairs", "tied_pairs",
                                                     "spearman_rho_vs_score_size", "note")})


if __name__ == "__main__":
    main()
