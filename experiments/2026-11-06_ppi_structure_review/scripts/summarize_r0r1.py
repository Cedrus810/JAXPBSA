#!/usr/bin/env python3
"""Per-model / per-replica condensation of the stage_trace.py tables (read-only CSV work).

Writes r0r1/model_summary.csv (one row per model x stage group) and
r0r1/replica_summary.csv (one row per replica: medians/max over the 20 one-ns checkpoints).
"""
import csv, re, statistics as st
from pathlib import Path
from collections import defaultdict

R = Path(__file__).resolve().parents[1] / "r0r1"
geo = [r for p in sorted(R.glob("stage_geometry_*.csv")) for r in csv.DictReader(p.open())]
res = [r for p in sorted(R.glob("stage_residue_*.csv")) for r in csv.DictReader(p.open())]
amap = [r for p in sorted(R.glob("stage_atom_map_*.csv")) for r in csv.DictReader(p.open())]
xc = [r for p in sorted(R.glob("diag_crosscheck_*.csv")) for r in csv.DictReader(p.open())]
f = lambda v: float(v) if v not in ("", None) else None

# ---- per model: build chain + minimization
rows = []
for (s, m) in sorted({(r["sample"], r["model"]) for r in geo}):
    g = {r["stage"]: r for r in geo if r["sample"] == s and r["model"] == m}
    am = [r for r in amap if r["sample"] == s and r["model"] == m]
    static = [r for r in am if r["to_stage"] != "minimized"]
    mn = g["minimized"]
    allst = [r for r in geo if r["sample"] == s and r["model"] == m]
    rows.append(dict(
        sample=s, model=m, seq_B_prepared=g["prepared"]["seq_B"],
        build_chain_max_heavy_dev_A=max(f(r["heavy_max_dev"]) for r in static),
        build_chain_removed_heavy=";".join(r["removed_heavy"] for r in static if r["removed_heavy"]),
        build_chain_resname_changes=";".join(r["resname_changes"] for r in static if r["resname_changes"]),
        npz_matches_prepared=next(r["npz_matches_prepared"] for r in am if r["to_stage"] == "prepared"),
        min_pep1_11_bb_vs3eqs=mn["pep1_11_bb_rmsd_vs3eqs"], min_pep1_11_bb_selfaligned=mn["pep1_11_bb_rmsd_selfaligned"],
        min_pep1_11_centroid_shift=mn["pep1_11_centroid_shift"], min_rec_CA_vs3eqs=mn["rec_CA_rmsd_vs3eqs"],
        min_A25_vs_stage1=mn["A25_heavy_rmsd_vs_stage1"], min_B12_vs_stage1=mn["B12_heavy_rmsd_vs_stage1"],
        all_stages_n=len(allst),
        all_stages_D_CA=sum(int(r["n_D_CA"]) for r in allst),
        all_stages_cis=";".join(sorted({r["cis_omega"] for r in allst if r["cis_omega"]})),
        all_stages_cb_chiral_flip=";".join(sorted({r["cbchiral_opposite_to_3eqs"] for r in allst if r["cbchiral_opposite_to_3eqs"]})),
        all_stages_HA_worst_deg=max(f(r["HA_worst_deg"]) for r in allst if r["HA_worst_deg"]),
        all_stages_clash_pairs=sum(int(r["n_clash_pairs"]) for r in allst),
        all_stages_max_bond_A=max(f(r["max_bond_len"]) for r in allst),
        static_bb_terms_gt4sigma=max(int(r["bb_terms_gt4sigma"]) for r in allst if "_" not in r["stage"] or r["stage"] == "3eqs_raw"),
        md_bb_terms_gt4sigma_total=sum(int(r["bb_terms_gt4sigma"]) for r in allst if re.match(r"r\d+_", r["stage"])),
        md_snapshots=sum(1 for r in allst if re.match(r"r\d+_", r["stage"])),
        pbc_shift_nonzero=sum(1 for r in allst if r["pbc_peptide_shift"] not in ("", "0,0,0")),
        diag_crosscheck_rows=sum(1 for r in xc if r["sample"] == s and r["model"] == m),
        diag_crosscheck_max_absdiff=max((max(abs(f(r["diag_pep_CA_rmsd"]) - f(r["state_pep_CA_rmsd"])),
                                             abs(f(r["diag_CA12_shift"]) - f(r["state_CA12_shift"])),
                                             abs(f(r["diag_min_rec_pep"]) - f(r["state_min_rec_pep"])))
                                         for r in xc if r["sample"] == s and r["model"] == m), default=None)))
with (R / "model_summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- per replica over production checkpoints
num = lambda lab: lab[0] + re.sub(r"\D", "", lab.split(":")[1])
rr = defaultdict(lambda: defaultdict(list))
for r in res:
    mm = re.match(r"(r\d+)_prod_", r["stage"])
    if mm and r["CA_dev"]:
        rr[(r["sample"], r["model"], mm.group(1))][num(r["residue"])].append(f(r["CA_dev"]))
gg = defaultdict(list)
for r in geo:
    mm = re.match(r"(r\d+)_prod_", r["stage"])
    if mm:
        gg[(r["sample"], r["model"], mm.group(1))].append(r)
out = []
for k in sorted(gg):
    G = gg[k]
    med = lambda c: round(st.median(f(x[c]) for x in G), 2)
    mx = lambda c: round(max(f(x[c]) for x in G), 2)
    row = dict(sample=k[0], model=k[1], replica=k[2], checkpoints=len(G),
               pep1_11_bb_vs3eqs_med=med("pep1_11_bb_rmsd_vs3eqs"), pep1_11_bb_vs3eqs_max=mx("pep1_11_bb_rmsd_vs3eqs"),
               pep1_11_bb_self_med=med("pep1_11_bb_rmsd_selfaligned"), pep1_11_bb_self_max=mx("pep1_11_bb_rmsd_selfaligned"),
               pep1_11_centroid_med=med("pep1_11_centroid_shift"),
               rec_CA_vs3eqs_med=med("rec_CA_rmsd_vs3eqs"), min_rec_pep_max=mx("min_rec_pep_heavy"))
    for p in ["A25"] + [f"B{i}" for i in range(1, 13)]:
        v = rr[k].get(p)
        row[f"CAdev_med_{p}"] = round(st.median(v), 2) if v else None
    out.append(row)
with (R / "replica_summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(rows), "models;", len(out), "replicas")
