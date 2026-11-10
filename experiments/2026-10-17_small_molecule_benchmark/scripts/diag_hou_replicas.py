#!/usr/bin/env python3
"""Hou a1/a3/a4/a6: why do replicas of one sample disagree?  Read-only diagnosis of existing
products (online per-frame components, h0.4 frames.csv, analysis/solute_dry.dcd); run on 183.

Plan: docs/SMALLMOL_GRID_HANDOFF_20261026.md §5 item 2 / docs/ENGINEERING_REPAIR_HANDOFF_20261024.md §5
item 3; design docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md §7 (pose/contact/block diagnostics).
  * components per frame: online default-grid e_coul_rl, e_lj_rl, delta_g_pb, delta_g_sa, total;
    elec = e_coul_rl + delta_g_pb (the two large cancelling terms); h0.4 total and delta_g_pb;
  * pose: receptor-CA fit onto setup/solute_reference.npz (minimised start), then ligand heavy-atom
    RMSD / COM shift to that start (no symmetry correction), min ligand-N .. Asp189 OD distance
    (Asp189 found by the chymotrypsin motif DACEGDSGG), residue contacts (heavy atoms <= 4.0 A);
  * per-residue ligand-receptor Coulomb (eps 1, all pairs) and LJ (Lorentz-Berthelot) from
    setup/system.xml; their sums are checked against the online e_coul_rl / e_lj_rl;
  * pose overlap: ligand RMSD (receptor frame, no ligand fit) between every pair of frames,
    stride 5, summarised as the median over (replica, 1-ns block) x (replica, 1-ns block).
Writes reports/hou_replica_diag_{frames,blocks,residues,pose}_20261028.csv.
Scope is the 2026-10-28 state: replicas rep01-03 and production (0, 5] ns (frames 0-499).  Those
replicas were later extended to 10 ns and rep04-06 added (RESULTS §20.9); WINDOW_PS / REPS keep the
analysis on the original data (fixed 2026-11-02, P3 reproduction: the unrestricted read failed).
"""
import csv
import json
from pathlib import Path
import sys

import mdtraj as md
import numpy as np
import openmm

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P.parents[1] / "scripts"))
from smallmol_records import online_rows_by_frame

SAMPLES = ["a1", "a3", "a4", "a6"]
TAG = "20261028"
WINDOW_PS = 5000.0     # (0, 5] ns production, the 3 x 5 ns round analysed on 10-28
REPS = ("rep01", "rep02", "rep03")
BLOCK_PS, CONTACT_A, POSE_STRIDE = 1000.0, 4.0, 5
KE = 138.935456 / 4.184  # kcal/mol * nm / e^2 (OpenMM ONE_4PI_EPS0)
ONE = dict(ALA="A", ARG="R", ASN="N", ASP="D", ASH="D", CYS="C", CYX="C", CYM="C", GLN="Q", GLU="E", GLH="E",
           GLY="G", HIS="H", HID="H", HIE="H", HIP="H", ILE="I", LEU="L", LYS="K", LYN="K", MET="M", PHE="F",
           PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")


def kabsch(X, Y):
    """R, t with X @ R.T + t ~= Y (least squares)."""
    cx, cy = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1, 1, d]) @ Vt).T
    return R, cy - cx @ R.T


def block_of(t):
    return int(np.ceil(t / BLOCK_PS - 1e-9))  # (0,1] ns -> 1 ... (4,5] ns -> 5


frames_out, blocks_out, res_out, pose_out = [], [], [], []
for s in SAMPLES:
    root = P / f"md/hou/{s}/ph74_gaff2"
    rec = json.loads((root / "setup/setup_record.json").read_text())
    ref = np.load(root / "setup/solute_reference.npz")
    l0, l1 = rec["ligand_local_idx"]
    elem = ref["element"]
    lig_h = np.array([i for i in range(l0, l1) if elem[i] != "H"])
    rec_ca = ref["receptor_ca_idx"]
    ref_nm = ref["coords_A"] / 10.0
    com_ref = ref_nm[lig_h].mean(0)

    sysx = openmm.XmlSerializer.deserialize((root / "setup/system.xml").read_text())
    nb = next(f for f in sysx.getForces() if isinstance(f, openmm.NonbondedForce))
    prm = np.array([[p[0]._value if hasattr(p[0], "_value") else p[0],
                     p[1]._value if hasattr(p[1], "_value") else p[1],
                     p[2]._value if hasattr(p[2], "_value") else p[2]]
                    for p in (nb.getParticleParameters(i) for i in range(l1))])
    q, sig, eps = prm[:, 0], prm[:, 1], prm[:, 2] / 4.184
    qq = KE * q[:l0, None] * q[None, l0:l1]
    sg = 0.5 * (sig[:l0, None] + sig[None, l0:l1])
    ep = np.sqrt(eps[:l0, None] * eps[None, l0:l1])

    rep_dirs = [root / r for r in REPS]
    top = md.load_topology(str(rep_dirs[0] / "analysis/solute_dry.pdb"))
    residues = [r for r in top.residues if r.atom(0).index < l0]
    res_of_atom = np.array([a.residue.index for a in top.atoms][:l0])
    res_ids = sorted(set(res_of_atom))
    res_label = {r.index: f"{r.name}{r.resSeq}" for r in residues}
    seq = "".join(ONE.get(r.name, "X") for r in residues)
    k = seq.find("DACEGDSGG")
    if k < 0 or seq.find("DACEGDSGG", k + 1) >= 0:
        raise RuntimeError(f"{s}: Asp189 motif not found exactly once")
    asp = residues[k]
    asp_od = np.array([a.index for a in asp.atoms if a.name in ("OD1", "OD2")])
    lig_n = np.array([i for i in range(l0, l1) if elem[i] == "N"])
    rec_heavy = np.array([i for i in range(l0) if elem[i] != "H"])
    heavy_res = res_of_atom[rec_heavy]

    pose_xyz, pose_key = [], []
    for rd in rep_dirs:
        rep = int(rd.name[3:])
        online, _ = online_rows_by_frame(rd)
        h04 = {int(r["frame"]): r for r in csv.DictReader((P / f"scores/h0.4/hou/{s}/ph74_gaff2/{rd.name}/frames.csv").open())}
        times = {int(r["frame"]): float(r["time_ps"]) for r in csv.DictReader((rd / "analysis/frame_times.csv").open())}
        tr = md.load(str(rd / "analysis/solute_dry.dcd"), top=top)
        if tr.n_frames != len(times) or len(online) != tr.n_frames:
            raise RuntimeError(f"{s} {rd.name}: frame counts differ")
        keep = [f for f in range(tr.n_frames) if times[f] <= WINDOW_PS + 1e-6]
        if keep != list(range(len(keep))) or len(keep) != 500:
            raise RuntimeError(f"{s} {rd.name}: expected frames 0-499 in (0, {WINDOW_PS}] ps")
        tr = tr[:len(keep)]
        res_coul = np.zeros((tr.n_frames, len(res_ids)))
        res_lj = np.zeros_like(res_coul)
        cont = np.zeros((tr.n_frames, len(res_ids)), bool)
        rows = []
        for f in range(tr.n_frames):
            X = tr.xyz[f].astype(np.float64)
            R, t = kabsch(X[rec_ca], ref_nm[rec_ca])
            Y = X @ R.T + t
            d = np.linalg.norm(Y[:l0, None] - Y[None, l0:l1], axis=2)
            ec = qq / d
            sr6 = (sg / d) ** 6
            el = 4 * ep * (sr6 * sr6 - sr6)
            res_coul[f] = np.bincount(res_of_atom, ec.sum(1), minlength=max(res_ids) + 1)[res_ids]
            res_lj[f] = np.bincount(res_of_atom, el.sum(1), minlength=max(res_ids) + 1)[res_ids]
            dh = d[rec_heavy][:, lig_h - l0].min(1) * 10
            cont[f] = np.isin(res_ids, heavy_res[dh <= CONTACT_A])
            o, h = online[f], h04[f]
            tp = times[f]
            row = dict(sample=s, replica=rep, frame=f, time_ps=tp, block=block_of(tp),
                       e_coul_rl=float(o["e_coul_rl"]), e_lj_rl=float(o["e_lj_rl"]),
                       delta_g_pb=float(o["delta_g_pb"]), delta_g_sa=float(o["delta_g_sa"]),
                       elec=float(o["e_coul_rl"]) + float(o["delta_g_pb"]),
                       dg_default=float(o["delta_g_mmpbsa"]), dg_h04=float(h["delta_g_mmpbsa"]),
                       dpb_h04=float(h["delta_g_pb"]),
                       coul_check=float(ec.sum()), lj_check=float(el.sum()),
                       lig_rmsd_start_A=float(np.sqrt(((Y[lig_h] - ref_nm[lig_h]) ** 2).sum(1).mean()) * 10),
                       lig_com_shift_A=float(np.linalg.norm(Y[lig_h].mean(0) - com_ref) * 10),
                       ligN_asp189_A=float(np.linalg.norm(Y[lig_n][:, None] - Y[asp_od][None], axis=2).min() * 10),
                       n_contact_res=int(cont[f].sum()))
            rows.append(row)
            if f % POSE_STRIDE == 0:
                pose_xyz.append(Y[lig_h])
                pose_key.append((rep, row["block"]))
        frames_out += rows
        for b in [0, 1, 2, 3, 4, 5]:
            m = np.array([r["block"] == b or b == 0 for r in rows])
            sel = [r for r, keep in zip(rows, m) if keep]
            agg = {k: round(float(np.mean([r[k] for r in sel])), 3)
                   for k in ("e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa", "elec", "dg_default", "dg_h04",
                             "lig_rmsd_start_A", "lig_com_shift_A", "ligN_asp189_A", "n_contact_res")}
            blocks_out.append(dict(sample=s, replica=rep, block="all" if b == 0 else b, n=len(sel), **agg))
            for j, ri in enumerate(res_ids):
                fr = float(cont[m, j].mean())
                c, l = float(res_coul[m, j].mean()), float(res_lj[m, j].mean())
                res_out.append(dict(sample=s, replica=rep, block="all" if b == 0 else b, residue=res_label[ri],
                                    contact_freq=round(fr, 3), coul=round(c, 3), lj=round(l, 3)))
        print(f"{s} {rd.name}: asp189={asp.name}{asp.resSeq}  max|coul-online|="
              f"{max(abs(r['coul_check'] - r['e_coul_rl']) for r in rows):.2e}  max|lj-online|="
              f"{max(abs(r['lj_check'] - r['e_lj_rl']) for r in rows):.2e}", flush=True)

    Z = np.array(pose_xyz)
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1).mean(-1)) * 10
    keys = sorted(set(pose_key))
    idx = {kk: np.array([i for i, x in enumerate(pose_key) if x == kk]) for kk in keys}
    for a in keys:
        for b in keys:
            sub = D[np.ix_(idx[a], idx[b])]
            if a == b:
                sub = sub[~np.eye(len(idx[a]), dtype=bool)]
            pose_out.append(dict(sample=s, rep_a=a[0], block_a=a[1], rep_b=b[0], block_b=b[1],
                                 median_rmsd_A=round(float(np.median(sub)), 3), n_pairs=sub.size))

for name, data in (("frames", frames_out), ("blocks", blocks_out), ("residues", res_out), ("pose", pose_out)):
    with (P / f"reports/hou_replica_diag_{name}_{TAG}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
print("wrote", len(frames_out), "frames")
