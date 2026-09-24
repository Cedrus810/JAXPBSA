#!/usr/bin/env python
"""ΔG_PB = G_C − G_R − G_L，跑在规范产物上 (RESULTS.md)。

    python scripts/crl.py [h] [32|64] [h_lig|shared]      # 默认 0.5 32 0.25

默认走 `TripletSolver` 的非对称网格：C/R 共用 h 的大盒，配体单独 h_lig 的紧盒
（RESULTS §15：ΔG_PB 的离散误差全在配体上）。`shared` = 旧的三者共用一张网格
（此时参考解 3 次降 2 次：u_ref_C = u_ref_R + u_ref_L，组合的是势不是能量）。
两种都先按质心归位。
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jaxpbsa
jaxpbsa.enable_compilation_cache()  # 编译 36.5 -> 5.9 s

h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
jaxpbsa.set_precision(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
h_lig = sys.argv[3] if len(sys.argv) > 3 else "0.25"
h_lig = None if h_lig == "shared" else float(h_lig)

import jax, numpy as np
from jaxpbsa.benchmark.roofline import device_info
from jaxpbsa.openmm_io import load_canonical
from openmm import unit
from jaxpbsa.pb import PBParams, TripletSolver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = load_canonical(root=ROOT)
xyz, q, radii = d["positions_A"], d["charge"], d["radii"]
rec, lig = d["receptor_idx"], d["ligand_idx"]
m = np.array([d["system"].getParticleMass(i).value_in_unit(unit.dalton)
              for i in range(len(q))])
p = PBParams(swin=0.5, tol=1e-5, max_iter=8000, precond="mg")
sv = TripletSolver(xyz, m, radii, rec, lig, p, h=h, h_lig=h_lig)
g = sv.grid

print(f"设备 {device_info()}   精度 {jaxpbsa.dtype().__name__}")
print(f"起点 {d['meta']['canonical_structure']} sha256 {d['meta']['canonical_sha256'][:16]}…")
print(f"网格 C/R {g.shape} = {int(np.prod(g.shape)):,}   h={h}")
if sv.grid_lig is not None:
    gl = sv.grid_lig
    print(f"网格 L   {gl.shape} = {int(np.prod(gl.shape)):,}   h={h_lig}")
print(f"原子 {len(q)} = receptor {rec.size} + ligand {lig.size}   "
      f"净电荷 {q.sum():+.3f} = {q[rec].sum():+.3f} + {q[lig].sum():+.3f}\n")

t0 = time.perf_counter(); o = sv(xyz, q)
jax.block_until_ready(o["delta_g_pb"]); comp = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(3):
    o = sv(xyz, q)
jax.block_until_ready(o["delta_g_pb"]); hot = (time.perf_counter() - t0) / 3

it, rr = np.asarray(o["iters"]), np.asarray(o["relres"])
print(f"{'species':<12}{'G_PB':>13}{'迭代':>7}{'真残差':>11}")
for i, nm in enumerate(("complex", "receptor", "ligand")):
    print(f"{nm:<12}{float(o[f'g_pb_{nm}']):13.3f}{it[i]:7d}{rr[i]:11.2e}")
print(f"{'ΔG_PB':<12}{float(o['delta_g_pb']):13.3f}   收敛 {bool(o['converged'])}")
print(f"\n编译 {comp:.1f}s（C/R 共用一份；非对称时配体另一份）")
print(f"热运行 {hot * 1e3:.1f} ms/帧  = {hot * 1e3 / 3:.1f} ms/species")
