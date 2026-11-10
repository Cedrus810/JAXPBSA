#!/usr/bin/env python3
"""Hou bounded sampling (6 replicas x 10 ns per sample): summary (read-only; run on 183).

Plan: docs/HOU_BOUNDED_SAMPLING_PLAN_20261028.md §2, §6 (window (0, 10] ns fixed before the run; halves and the
first-round 3 x (0, 5] ns for comparison; replica = time mean, sample = equal-weight replica mean).
  * scores: online rows (smallmol_records.online_rows) for every replica, except the three whose online rows have
    ligand-box-margin flags (§7.3), which use the offline rescoring with a (0, 10] ns window pilot
    (scores/window10_default/...) for all their frames; on those three the online - offline difference of the
    unflagged frames is reported;
  * every row used is checked with row_violation; the summary is not written if any used row is flagged;
  * pose per replica and half: ligand heavy-atom RMSD / COM shift to the minimised start (analysis/diagnostics.csv)
    and min ligand N .. Asp189 OD distance (Asp189 by the DACEGDSGG motif, as scripts/diag_hou_replicas.py);
  * adjacent samples in the (0, 10] order: gap, combined replica SE, and the §20.7 grid correction h0.35 - h0.4.
Writes reports/hou_sampling_{replicas,samples,pairs,replaced}_20261030.csv.
"""
import csv
import json
from pathlib import Path
import sys

import mdtraj as md
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P.parents[1] / "scripts"))
from smallmol_records import online_rows, row_violation

TAG = "20261030"
SAMPLES = [f"a{i}" for i in range(1, 8)]
REPLACED = {("a3", "01"), ("a3", "02"), ("a1", "05")}
COMP = ["e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa"]
ONE = dict(ALA="A", ARG="R", ASN="N", ASP="D", ASH="D", CYS="C", CYX="C", CYM="C", GLN="Q", GLU="E", GLH="E",
           GLY="G", HIS="H", HID="H", HIE="H", HIP="H", ILE="I", LEU="L", LYS="K", LYN="K", MET="M", PHE="F",
           PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")


def kabsch(X, Y):
    cx, cy = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1, 1, d]) @ Vt).T
    return R, cy - cx @ R.T


def asp189_distance(root, rep):
    rec = json.loads((root / "setup/setup_record.json").read_text())
    ref = np.load(root / "setup/solute_reference.npz")
    l0, l1 = rec["ligand_local_idx"]
    top = md.load_topology(str(rep / "analysis/solute_dry.pdb"))
    residues = [r for r in top.residues if r.atom(0).index < l0]
    seq = "".join(ONE.get(r.name, "X") for r in residues)
    k = seq.find("DACEGDSGG")
    if k < 0 or seq.find("DACEGDSGG", k + 1) >= 0:
        raise RuntimeError(f"{root}: Asp189 motif not found exactly once")
    od = np.array([a.index for a in residues[k].atoms if a.name in ("OD1", "OD2")])
    lig_n = np.array([i for i in range(l0, l1) if ref["element"][i] == "N"])
    tr = md.load(str(rep / "analysis/solute_dry.dcd"), top=top)
    return np.linalg.norm(tr.xyz[:, lig_n][:, :, None] - tr.xyz[:, od][:, None], axis=3).min((1, 2)) * 10


reps, replaced = [], []
for s in SAMPLES:
    root = P / f"md/hou/{s}/ph74_gaff2"
    for r in ["01", "02", "03", "04", "05", "06"]:
        rep = root / f"rep{r}"
        st = json.loads((rep / "status.json").read_text())
        on = online_rows(st)
        if (s, r) in REPLACED:
            off = list(csv.DictReader((P / f"scores/window10_default/hou/{s}/ph74_gaff2/rep{r}/frames.csv").open()))
            off_by = {int(x["step"]): x for x in off}
            pair = [(x, off_by[int(x["step"])]) for x in on
                    if int(x["step"]) in off_by and not row_violation(x, x["_margin_flag"])]
            d = np.array([float(b["delta_g_mmpbsa"]) - float(a["delta_g_mmpbsa"]) for a, b in pair])
            dpb = np.array([float(b["delta_g_pb"]) - float(a["delta_g_pb"]) for a, b in pair])
            replaced.append(dict(sample=s, replica=int(r), n_online=len(on),
                                 n_online_flagged=sum(1 for x in on if row_violation(x, x["_margin_flag"])),
                                 n_offline=len(off), n_offline_flagged=sum(1 for x in off if x["flag"]),
                                 n_paired_unflagged=len(pair), offline_minus_online_mean=round(float(d.mean()), 4),
                                 offline_minus_online_sd=round(float(d.std(ddof=1)), 4),
                                 offline_minus_online_maxabs=round(float(np.abs(d).max()), 4),
                                 d_pb_mean=round(float(dpb.mean()), 4)))
            rows = [dict(x, _bad=x["flag"]) for x in off]
        else:
            rows = [dict(x, _bad=row_violation(x, x["_margin_flag"])) for x in on]
        bad = [x for x in rows if x["_bad"]]
        if bad:
            raise RuntimeError(f"{s} rep{r}: {len(bad)} flagged rows used; summary not written")
        t = np.array([int(x["step"]) * st["dt_ps"] for x in rows])
        if len(rows) != 1000 or abs(t.max() - 10000) > 1e-6:
            raise RuntimeError(f"{s} rep{r}: {len(rows)} rows, last t {t.max()}")
        v = {k: np.array([float(x[k]) for x in rows]) for k in COMP}
        h1, h2 = t <= 5000 + 1e-6, t > 5000 + 1e-6
        dg = pd.read_csv(rep / "analysis/diagnostics.csv")
        dist = asp189_distance(root, rep)
        dg_h = [dg.time_ps <= 5000 + 1e-6, dg.time_ps > 5000 + 1e-6]
        reps.append(dict(
            sample=s, replica=int(r), source="offline_w10" if (s, r) in REPLACED else "online",
            dg_0_10=float(v["delta_g_mmpbsa"].mean()), dg_0_5=float(v["delta_g_mmpbsa"][h1].mean()),
            dg_5_10=float(v["delta_g_mmpbsa"][h2].mean()),
            **{f"{k}_0_10": float(v[k].mean()) for k in COMP[:4]},
            elec_0_10=float((v["e_coul_rl"] + v["delta_g_pb"]).mean()),
            **{f"block{b + 1}": float(v["delta_g_mmpbsa"][(t > b * 1000 + 1e-6) & (t <= (b + 1) * 1000 + 1e-6)].mean())
               for b in range(10)},
            lig_rmsd_med_h1=float(dg.ligand_heavy_rmsd_receptor_frame_A[dg_h[0]].median()),
            lig_rmsd_med_h2=float(dg.ligand_heavy_rmsd_receptor_frame_A[dg_h[1]].median()),
            com_med_h2=float(dg.ligand_com_shift_A[dg_h[1]].median()),
            asp189_med_h1=float(np.median(dist[:500])), asp189_med_h2=float(np.median(dist[500:]))))
        print(f"{s} rep{r} done", flush=True)

R = pd.DataFrame(reps)
S = []
for s, g in R.groupby("sample"):
    first = g[g.replica <= 3]
    S.append(dict(sample=s, n=len(g), dg_0_10=g.dg_0_10.mean(), se_0_10=g.dg_0_10.std(ddof=1) / np.sqrt(len(g)),
                  range_0_10=g.dg_0_10.max() - g.dg_0_10.min(),
                  dg_0_5_6rep=g.dg_0_5.mean(), dg_5_10_6rep=g.dg_5_10.mean(),
                  dg_0_5_first3=first.dg_0_5.mean(), se_0_5_first3=first.dg_0_5.std(ddof=1) / np.sqrt(3)))
S = pd.DataFrame(S).set_index("sample")
acc = pd.read_csv(P / "reports/hou_acceptance_samples_20261028.csv").set_index("sample")
order = S.sort_values("dg_0_10").index.tolist()
pairs = []
for a, b in zip(order, order[1:]):
    pairs.append(dict(stronger=a, weaker=b, gap=S.dg_0_10[b] - S.dg_0_10[a],
                      se_combined=float(np.hypot(S.se_0_10[a], S.se_0_10[b])),
                      grid_d35_40_change=float(acc.d35_40[b] - acc.d35_40[a])))
pairs = pd.DataFrame(pairs)

R.round(3).to_csv(P / f"reports/hou_sampling_replicas_{TAG}.csv", index=False)
S.round(3).reset_index().to_csv(P / f"reports/hou_sampling_samples_{TAG}.csv", index=False)
pairs.round(3).to_csv(P / f"reports/hou_sampling_pairs_{TAG}.csv", index=False)
pd.DataFrame(replaced).to_csv(P / f"reports/hou_sampling_replaced_{TAG}.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(replaced).to_string(index=False))
print(R[["sample", "replica", "source", "dg_0_10", "dg_0_5", "dg_5_10", "elec_0_10", "e_lj_rl_0_10",
         "lig_rmsd_med_h1", "lig_rmsd_med_h2", "asp189_med_h1", "asp189_med_h2"]].round(2).to_string(index=False))
print(S.round(2).to_string())
print(pairs.round(2).to_string(index=False))
