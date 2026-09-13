#!/usr/bin/env python
"""M0 acceptance: MD quality analysis for the S4 production trajectory.

    python scripts/analyze_md.py [--dcd data/md/S4_dry.dcd]

Checks (DESIGN.md §5-M0: "RMSD 稳定"):
  - frame/atom counts, solute extent (PBC imaging sanity)
  - complex / receptor / ligand Cα RMSD (receptor-aligned; ligand RMSD after
    receptor alignment = binding-pose drift, 不做内部对齐)
  - receptor–ligand minimum heavy-atom distance (脱结合检测), 每 20 帧采样
  - temperature / density stats from S4_prod.log
Outputs: printed summary + data/md/S4_md_report.json + data/md/S4_rmsd.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    import mdtraj as md

    ap = argparse.ArgumentParser()
    ap.add_argument("--dcd", default=os.path.join(ROOT, "data", "md", "S4_dry.dcd"))
    ap.add_argument("--top", default=os.path.join(ROOT, "data", "prepared", "S4_dry.pdb"))
    ap.add_argument("--log", default=os.path.join(ROOT, "data", "md", "S4_prod.log"))
    args = ap.parse_args()

    traj = md.load(args.dcd, top=args.top)
    with open(os.path.join(ROOT, "data", "prepared", "S4_meta.json")) as fh:
        meta = json.load(fh)
    assert traj.n_atoms == meta["n_atoms"], (traj.n_atoms, meta["n_atoms"])
    print(f"trajectory: {traj.n_frames} frames x {traj.n_atoms} atoms, "
          f"dt = {traj.time[1] - traj.time[0]:.0f} ps, total {traj.time[-1] / 1000:.1f} ns")

    rec = traj.topology.select(f"chainid 0")  # S4_dry.pdb: chain C first, then F
    lig = traj.topology.select(f"chainid 1")
    assert len(rec) == meta["n_receptor"] and len(lig) == meta["n_ligand"]

    ca = traj.topology.select("name CA")
    ca_rec = np.intersect1d(ca, rec)
    ca_lig = np.intersect1d(ca, lig)

    # strip_unwrap.py 已在溶剂化体系上完成 PBC imaging, 这里不重复(干轨迹无锚分子会报错)
    ref = traj[0]
    traj.superpose(ref, frame=0, atom_indices=ca_rec)  # 受体 Cα 对齐

    rmsd_ca = md.rmsd(traj, ref, frame=0, atom_indices=ca)
    rmsd_rec = md.rmsd(traj, ref, frame=0, atom_indices=ca_rec)
    rmsd_lig = md.rmsd(traj, ref, frame=0, atom_indices=ca_lig)  # 结合姿态漂移

    def stats(x):
        tail = x[len(x) // 2:]  # 后半段
        return float(x.mean()), float(x.std()), float(tail.mean()), float(tail.std())

    m = {k: stats(v) for k, v in (("ca_all", rmsd_ca), ("rec", rmsd_rec), ("lig_pose", rmsd_lig))}
    print(f"RMSD (Å): Cα-all {m['ca_all'][0]:.2f}±{m['ca_all'][1]:.2f} | "
          f"receptor {m['rec'][0]:.2f}±{m['rec'][1]:.2f} | "
          f"ligand pose (rec-aligned) {m['lig_pose'][0]:.2f}±{m['lig_pose'][1]:.2f}")
    print(f"           后半段: receptor {m['rec'][2]:.2f}±{m['rec'][3]:.2f}, "
          f"ligand pose {m['lig_pose'][2]:.2f}±{m['lig_pose'][3]:.2f}")

    # 界面最小重原子距离 (每 20 帧)
    heavy_lig = np.array([i for i in lig if traj.topology.atom(i).element.symbol != "H"])
    heavy_rec = np.array([i for i in rec if traj.topology.atom(i).element.symbol != "H"])
    dmin = []
    for i in range(0, traj.n_frames, 20):
        xyz = traj.xyz[i] * 10.0  # Å
        d2 = ((xyz[heavy_rec][:, None, :] - xyz[heavy_lig][None, :, :]) ** 2).sum(-1)
        dmin.append(float(np.sqrt(d2.min())))
    dmin = np.array(dmin)
    n_detached = int((dmin > 5.0).sum())
    print(f"interface min heavy-atom distance: {dmin.mean():.2f}±{dmin.std():.2f} Å, "
          f"max {dmin.max():.2f} Å, frames >5 Å: {n_detached}/{len(dmin)}")

    # log 统计 (生产段)
    T, rho = [], []
    for line in open(args.log):
        if line.startswith("#") or not line.strip():
            continue
        parts = [float(v) for v in line.strip().split(",")]
        if parts[1] >= 1200:  # 生产从 ~1.1 ns 开始
            T.append(parts[3])
            rho.append(parts[4])
    print(f"log (production): T = {np.mean(T):.2f}±{np.std(T):.2f} K, "
          f"density = {np.mean(rho):.4f}±{np.std(rho):.4f} g/mL")

    report = {
        "n_frames": int(traj.n_frames),
        "n_atoms": int(traj.n_atoms),
        "length_ns": float(traj.time[-1] / 1000),
        "rmsd_ca_mean_A": m["ca_all"][0], "rmsd_ca_std_A": m["ca_all"][1],
        "rmsd_receptor_tail_A": m["rec"][2], "rmsd_lig_pose_tail_A": m["lig_pose"][2],
        "interface_min_dist_mean_A": float(dmin.mean()),
        "interface_min_dist_max_A": float(dmin.max()),
        "n_frames_detached_gt5A": n_detached,
        "temperature_K": float(np.mean(T)), "density_g_ml": float(np.mean(rho)),
    }
    with open(os.path.join(ROOT, "data", "md", "S4_md_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    t = traj.time / 1000
    ax[0].plot(t, rmsd_rec, lw=0.6, label="receptor Cα")
    ax[0].plot(t, rmsd_lig, lw=0.6, label="ligand Cα (rec-aligned)")
    ax[0].set(xlabel="time (ns)", ylabel="RMSD (Å)", title="S4 10 ns production")
    ax[0].legend(frameon=False)
    ax[1].plot(t[::20], dmin, lw=0.6)
    ax[1].axhline(5.0, color="r", ls="--", lw=0.8)
    ax[1].set(xlabel="time (ns)", ylabel="min R–L heavy dist (Å)", title="interface contact")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "data", "md", "S4_rmsd.png"), dpi=150)
    print(f"wrote S4_md_report.json + S4_rmsd.png")


if __name__ == "__main__":
    main()
