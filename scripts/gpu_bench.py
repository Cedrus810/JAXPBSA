#!/usr/bin/env python
"""大卡上的两件事(RESULTS §18.13 之后):

A. 速度: S4 / 1YCR 端到端每帧(TripletSolver, 10 帧, 去掉首帧编译), 配置
   binary 0.5(默认) / fraction 0.5 / gaussian 0.5 / gaussian 0.4。
C. 在线端到端(OnlineMMPBSA, MM + PB + SA, 默认 binary 0.5 + 配体 0.25): 网格用轨迹前 10%
   (S4 为 1 ns)作 pilot 定尺, 再在全轨迹等距 11 帧上计时 + 查 margin。
B. 高斯 + 自项修正的收敛矩阵(第 0 帧): C/R h ∈ {0.5,0.4,0.35,0.3,0.25} × 3 相位,
   L 单独 h ∈ {0.25, 0.2} × 3 相位(绕开质心归位才能变相位)。ΔG = C−R − G_L(0.2)。

    python scripts/gpu_bench.py [--part A|B|C|ABC] [--names S4,1YCR]
"""
from __future__ import annotations

import argparse
import os
import socket
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, n_frames):
    import mdtraj as md
    from openmm import unit
    from jaxpbsa.openmm_io import load_canonical
    from jaxpbsa.pb import recenter_com

    d = load_canonical(name, root=ROOT)
    q, r = np.asarray(d["charge"]), np.asarray(d["radii"])
    rec, lig = np.asarray(d["receptor_idx"]), np.asarray(d["ligand_idx"])
    m = np.array([d["system"].getParticleMass(i).value_in_unit(unit.dalton) for i in range(len(q))])
    t = md.load(os.path.join(ROOT, f"data/md/{name}_dry.dcd"),
                top=os.path.join(ROOT, f"data/prepared/{name}_complex.pdb"))
    idx = np.linspace(0, t.n_frames - 1, n_frames).astype(int)
    F = recenter_com((t.xyz[idx] * 10.0).astype(np.float64), m, np.zeros(3))
    return q, r, rec, lig, m, F


def part_a(name):
    from jaxpbsa.pb import PBParams, TripletSolver

    q, r, rec, lig, m, F = load(name, 11)
    for label, P, h in (("binary", PBParams(), 0.5), ("fraction", PBParams(surface="fraction"), 0.5),
                        ("gaussian", PBParams(surface="gaussian"), 0.5),
                        ("gaussian", PBParams(surface="gaussian"), 0.4)):
        try:
            tri = TripletSolver(F[0], m, r, rec, lig, params=P, h=h)
            ts, dg = [], []
            for f in F:
                t0 = time.perf_counter()
                o = tri(f, q)
                dg.append(float(o["delta_g_pb"]))  # float() 同步设备
                ts.append(time.perf_counter() - t0)
            print(f"[A] {name:5s} {label:8s} h={h:<4} grid {tri.grid.shape}  "
                  f"{1e3 * np.median(ts[1:]):7.1f} ms/帧 (首帧含编译 {ts[0]:5.1f} s)  "
                  f"⟨ΔG_PB⟩ {np.mean(dg[1:]):9.2f}", flush=True)
        except Exception as e:
            print(f"[A] {name} {label} h={h}: {type(e).__name__} {str(e)[:160]}", flush=True)


def part_b(name):
    from jaxpbsa.pb import PBParams, TripletSolver, make_frame_solver, make_grid, recenter_com

    q, r, rec, lig, m, F = load(name, 1)
    X = F[0]
    P = PBParams(surface="gaussian")
    cr = {}
    for h in (0.5, 0.4, 0.35, 0.3, 0.25):
        try:
            tri = TripletSolver(X, m, r, rec, lig, params=P, h=h)
            v, ts = [], []
            for s in range(3):
                sh = np.random.default_rng(50 + s).uniform(0, h, 3)
                t0 = time.perf_counter()
                o = tri._sv.pair(tri.grid.center + X + sh, q, rec)
                v.append(float(o["g_pb_complex"]) - float(o["g_pb_receptor"]))
                ts.append(time.perf_counter() - t0)
                assert bool(o["converged"])
            cr[h] = np.mean(v)
            print(f"[B] {name:5s} C−R h={h:<5} {tri.grid.shape}: {np.round(v, 1)}  均值 {np.mean(v):10.2f}"
                  f"  sd {np.std(v, ddof=1):6.2f}  {1e3 * ts[-1]:7.0f} ms/pair", flush=True)
        except Exception as e:
            print(f"[B] {name} C−R h={h}: {type(e).__name__} {str(e)[:160]}", flush=True)
    XL = recenter_com(X[lig], m[lig], np.zeros(3))
    gl = {}
    for h in (0.25, 0.2):
        g = make_grid(XL, h, padding=8.0, center=np.zeros(3))
        sv = make_frame_solver(g, r[lig], P)
        v = []
        for s in range(3):
            o = sv(XL + np.random.default_rng(80 + s).uniform(0, h, 3), q[lig], r[lig])
            v.append(float(o["g_pb"]))
            assert bool(o["converged"])
        gl[h] = np.mean(v)
        print(f"[B] {name:5s} L   h={h:<5} {g.shape}: {np.round(v, 1)}  均值 {np.mean(v):10.2f}"
              f"  sd {np.std(v, ddof=1):6.2f}", flush=True)
    for h, c in cr.items():
        print(f"[B] {name:5s} ΔG_PB(C/R h={h:<5}) = {c - gl[0.2]:9.2f}", flush=True)


def part_c(name):
    import mdtraj as md
    from jaxpbsa.online import OnlineMMPBSA
    from jaxpbsa.openmm_io import load_canonical

    d = load_canonical(name, root=ROOT)
    t = md.load(os.path.join(ROOT, f"data/md/{name}_dry.dcd"),
                top=os.path.join(ROOT, f"data/prepared/{name}_complex.pdb"))
    X = (t.xyz * 10.0).astype(np.float64)
    n = X.shape[1]
    pilot = X[: max(2, len(X) // 10)]
    t0 = time.perf_counter()
    az = OnlineMMPBSA(d["system"], d["topology"], np.arange(n), np.asarray(d["ligand_idx"]),
                      X[0], pilot_coords_A=pilot)
    init_s = time.perf_counter() - t0
    s = az.sizing
    ts, dg, mins = [], [], []
    for f in X[np.linspace(0, len(X) - 1, 11).astype(int)]:
        t0 = time.perf_counter()
        o = az(f)
        ts.append(time.perf_counter() - t0)
        dg.append(o["delta_g_mmpbsa"])
        mins.append(min(o["margin_A"] - az.margin_min, o["margin_lig_A"]))
    print(f"[C] {name:5s} online  pilot {len(pilot)} 帧 -> C/R {s['grid']} L {s['grid_lig']}  "
          f"init(含预热) {init_s:5.1f} s  {1e3 * np.median(ts):7.1f} ms/帧 (MM+PB+SA)  "
          f"⟨ΔG_MMPBSA⟩ {np.mean(dg):9.2f}  最差余量 {min(mins):+.2f} Å", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="C")
    ap.add_argument("--names", default="S4,1YCR")
    a = ap.parse_args()
    import jax
    import jaxpbsa

    jaxpbsa.enable_compilation_cache()
    print(f"host {socket.gethostname()}  devices {jax.devices()}", flush=True)
    for name in a.names.split(","):
        if "A" in a.part:
            part_a(name)
        if "B" in a.part:
            part_b(name)
        if "C" in a.part:
            part_c(name)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
