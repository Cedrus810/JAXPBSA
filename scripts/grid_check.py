#!/usr/bin/env python
"""两条网格诊断 (RESULTS §17), 换体系时重跑:

    python scripts/grid_check.py --name 1YCR

1. **摆放扫描**: canonical 构象、固定网格, 刚体平移 7 种亚格点相位。
   按相位平均 = 系统偏差, sd = 逐帧摆放噪声。回答: C−R 是否在 h=0.75 无偏,
   G_L 在 h=0.5 的偏差有多大 —— 也就是 `TripletSolver` 默认的依据在这个体系上
   是否还成立。
2. **20 帧轨迹**: 默认非对称网格 vs 旧的三者共用 h=0.5, 逐帧与 ms/帧。

外部对拍另跑 `validate_mmpbsa.py --name ... --frames 20 --stride 500`
(与这里取同一批帧)。
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIFTS = [(0, 0, 0), (.25, 0, 0), (.5, 0, 0), (0, .5, 0),
          (.25, .25, .25), (.5, .5, .5), (.33, .71, .12)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="S4")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--stride", type=int, default=500)
    a = ap.parse_args()

    import jax
    import jaxpbsa
    jaxpbsa.enable_compilation_cache()
    import mdtraj as md
    from openmm import unit
    from jaxpbsa.openmm_io import load_canonical
    from jaxpbsa.pb import (PBParams, TripletSolver, make_frame_solver, make_grid,
                            recenter_com)

    d = load_canonical(a.name, root=ROOT)
    x0, q, radii = d["positions_A"], d["charge"], d["radii"]
    rec, lig = d["receptor_idx"], d["ligand_idx"]
    m = np.array([d["system"].getParticleMass(i).value_in_unit(unit.dalton)
                  for i in range(len(q))])
    P = PBParams(swin=0.5, tol=1e-5, max_iter=8000, precond="mg")
    print(f"{a.name}: {len(q)} atoms = R {rec.size} + L {lig.size}, "
          f"q {q[rec].sum():+.0f}/{q[lig].sum():+.0f} | {jax.devices()}", flush=True)

    # ---- 1. 摆放扫描 ----
    c = recenter_com(x0, m, np.zeros(3))
    for h in (0.75, 0.5):
        g = make_grid(c[None], h, padding=20.0, center=np.zeros(3))
        sv = make_frame_solver(g, radii, P)
        v = []
        for s in SHIFTS:
            o = sv.pair(c + np.array(s) * h, q, rec)
            v.append(float(o["g_pb_complex"]) - float(o["g_pb_receptor"]))
        v = np.array(v)
        print(f"C-R  h={h:<5} {g.shape}: mean {v.mean():9.3f} sd {v.std():6.3f} "
              f"p2p {np.ptp(v):6.3f}", flush=True)
    cl = recenter_com(x0[lig], m[lig], np.zeros(3))
    for h in (0.5, 0.375, 0.25):
        g = make_grid(cl[None], h, padding=8.0, center=np.zeros(3))
        sv = make_frame_solver(g, radii[lig], P)
        v = np.array([float(sv(cl + np.array(s) * h, q[lig], radii[lig])["g_pb"])
                      for s in SHIFTS])
        print(f"G_L  h={h:<5} {g.shape}: mean {v.mean():9.3f} sd {v.std():6.3f} "
              f"p2p {np.ptp(v):6.3f}", flush=True)

    # ---- 2. 20 帧: 默认非对称 vs 共用 h=0.5 ----
    t = md.load(os.path.join(ROOT, f"data/md/{a.name}_dry.dcd"),
                top=os.path.join(ROOT, f"data/prepared/{a.name}_complex.pdb"))
    F = (t.xyz[::a.stride][:a.frames] * 10.0).astype(np.float64)
    runs = {"asym (default)": TripletSolver(F, m, radii, rec, lig, P),
            "shared h=0.5": TripletSolver(F, m, radii, rec, lig, P, h=0.5, h_lig=None)}
    res = {}
    for tag, tri in runs.items():
        jax.block_until_ready(tri(F[0], q)["delta_g_pb"])  # 编译
        v, ts, ok = [], [], True
        for f in F:
            t0 = time.perf_counter()
            o = tri(f, q)
            jax.block_until_ready(o["delta_g_pb"])
            ts.append(time.perf_counter() - t0)
            v.append(float(o["delta_g_pb"]))
            ok &= bool(o["converged"])
        res[tag] = np.array(v)
        gl = "" if tri.grid_lig is None else f" + L {tri.grid_lig.shape}"
        print(f"{tag:<15} C/R {tri.grid.shape}{gl}: <ΔG_PB> {np.mean(v):9.3f} "
              f"frame sd {np.std(v, ddof=1):6.2f} | {np.mean(ts)*1e3:6.1f} ms/frame | "
              f"converged {ok}", flush=True)
    dd = res["asym (default)"] - res["shared h=0.5"]
    print(f"asym - shared: mean {dd.mean():+.3f} sd {dd.std(ddof=1):.3f} "
          f"({len(F)} frames, stride {a.stride})")


if __name__ == "__main__":
    main()
