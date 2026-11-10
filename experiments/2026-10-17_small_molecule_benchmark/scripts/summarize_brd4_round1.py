#!/usr/bin/env python3
"""Collect BRD4 round-1 per-replica MD diagnostics and scores into reports/ (read-only over
md/ and scores/default/; no computation).  Writes reports/brd4_round1_replicas.csv and
reports/brd4_round1_samples.csv (optional argument <grid>, e.g. h0.4: reads scores/<grid>/ and
writes reports/brd4_<grid>_{replicas,samples}.csv).  Replica score = time mean over (0, 5] ns valid frames;
sample score = equal-weight mean of replica means (design §9).  No experimental statistics."""
import csv, json, statistics as st, sys
from pathlib import Path
P = Path(__file__).resolve().parents[1]
GRID = sys.argv[1] if len(sys.argv) > 1 else "default"
TAG = "brd4_round1" if GRID == "default" else f"brd4_{GRID}"
meas = {r["sample"]: r for r in csv.DictReader(open(P / "manifests/brd4_measurements.csv"))}
reps, per = [], {}
for i in range(1, 11):
    s = f"BRD4-{i}"
    for r in (1, 2, 3):
        rep = P / f"md/brd4/{s}/mobley_amber/rep{r:02d}"
        stt = json.load(open(rep / "status.json"))
        q = json.load(open(rep / "analysis/diagnostics_summary.json"))["quantiles_p5_p50_p95"]
        row = dict(sample=s, replica=r, chunks_done=stt["chunks_done"],
                   prod_s_per_ns=round(10 * st.median([v for k, v in stt["stage_wall_s"].items() if k.startswith("prod")]), 1),
                   lig_rmsd_p50_A=q["ligand_heavy_rmsd_receptor_frame_A"]["50"], lig_rmsd_p95_A=q["ligand_heavy_rmsd_receptor_frame_A"]["95"],
                   lig_com_shift_p95_A=q["ligand_com_shift_A"]["95"], rec_ca_rmsd_p50_A=q["receptor_CA_rmsd_A"]["50"],
                   min_lig_rec_heavy_p95_A=q["min_lig_rec_heavy_A"]["95"])
        sp = P / f"scores/{GRID}/brd4/{s}/mobley_amber/rep{r:02d}/summary.json"
        if sp.exists():
            sc = json.load(open(sp))
            row.update(score_status="scored", n_valid=sc["n_valid"], n_flagged=sc["n_flagged"],
                       dG_mmpbsa_mean=round(sc["delta_g_mmpbsa_mean"], 3), dG_frame_sd=round(sc["delta_g_mmpbsa_sd"], 3),
                       **{f"block_{k}": round(v["mean"], 3) for k, v in sc["block_means"].items()},
                       **{k: round(v, 3) for k, v in sc["components_mean"].items()},
                       grid="x".join(map(str, sc["sizing"]["grid"])), grid_lig="x".join(map(str, sc["sizing"]["grid_lig"])),
                       ms_per_frame=sc["ms_per_frame_this_call"]["median"] if sc["ms_per_frame_this_call"] else "")
            per.setdefault(s, []).append(sc["delta_g_mmpbsa_mean"])
        else:
            row.update(score_status="not_scored_OnlineMMPBSA_integer_charge_check_(handoff_7.4)")
        reps.append(row)
keys = list(dict.fromkeys(k for r in reps for k in r))
with open(P / f"reports/{TAG}_replicas.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(reps)
with open(P / f"reports/{TAG}_samples.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["sample", "n_replicas_scored", "dG_mmpbsa_mean_of_replicas", "replica_range",
                                       "exp_dG_kcal_mol", "assay_type", "label"])
    w.writeheader()
    for i in range(1, 11):
        s = f"BRD4-{i}"; v = per.get(s, [])
        w.writerow(dict(sample=s, n_replicas_scored=len(v), dG_mmpbsa_mean_of_replicas=round(st.mean(v), 3) if v else "",
                        replica_range=round(max(v) - min(v), 3) if v else "", exp_dG_kcal_mol=meas[s]["experimental_dg_kcal_mol"],
                        assay_type=meas[s]["assay_type"], label=meas[s]["label"]))
print(f"wrote reports/{TAG}_replicas.csv, reports/{TAG}_samples.csv")
