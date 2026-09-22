#!/usr/bin/env python
"""ΔG_PB = G_C − G_R − G_L，跑在规范产物上 (RESULTS.md)。

    python scripts/crl.py [h] [32|64]

参考解 3 次降 2 次: 公共网格上参考方程处处 ε_in、κ̄²=0、源项与库仑边界都对电荷线性，
所以 **u_ref_C = u_ref_R + u_ref_L**（组合的是**势**）。不能把能量相加 —— 参考能量含
R–L 交叉项。溶剂方程的介电图三者不同（配体在场时受体表面被遮挡），仍是 3 次。
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jaxpbsa
jaxpbsa.enable_compilation_cache()  # 编译 36.5 -> 5.9 s

h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
jaxpbsa.set_precision(int(sys.argv[2]) if len(sys.argv) > 2 else 32)

import jax, numpy as np
from jaxpbsa.benchmark.roofline import device_info
from jaxpbsa.openmm_io import load_canonical
from jaxpbsa.pb import PBParams, make_frame_solver, make_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = load_canonical(root=ROOT)
xyz, q, radii = d["positions_A"], d["charge"], d["radii"]
rec, lig = d["receptor_idx"], d["ligand_idx"]
g = make_grid(xyz[None], h, padding=20.0)
p = PBParams(swin=0.5, tol=1e-5, max_iter=8000, precond="mg")
sv = make_frame_solver(g, radii, p)

print(f"设备 {device_info()}   精度 {jaxpbsa.dtype().__name__}")
print(f"起点 {d['meta']['canonical_structure']} sha256 {d['meta']['canonical_sha256'][:16]}…")
print(f"网格 {g.shape} = {int(np.prod(g.shape)):,}   h={h}")
print(f"原子 {len(q)} = receptor {rec.size} + ligand {lig.size}   "
      f"净电荷 {q.sum():+.3f} = {q[rec].sum():+.3f} + {q[lig].sum():+.3f}\n")

t0 = time.perf_counter(); o = sv.triplet(xyz, q, rec, lig)
jax.block_until_ready(o["delta_g_pb"]); comp = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(3):
    o = sv.triplet(xyz, q, rec, lig)
jax.block_until_ready(o["delta_g_pb"]); hot = (time.perf_counter() - t0) / 3

it, rr = np.asarray(o["iters"]), np.asarray(o["relres"])
print(f"{'species':<12}{'G_PB':>13}{'迭代':>7}{'真残差':>11}")
for i, nm in enumerate(("complex", "receptor", "ligand")):
    print(f"{nm:<12}{float(o[f'g_pb_{nm}']):13.3f}{it[i]:7d}{rr[i]:11.2e}")
print(f"{'ΔG_PB':<12}{float(o['delta_g_pb']):13.3f}   收敛 {bool(o['converged'])}")
print(f"\n编译 {comp:.1f}s（三个 species 共用一份 —— R/L 补齐到 complex 的原子数）")
print(f"热运行 {hot * 1e3:.1f} ms/帧  = {hot * 1e3 / 3:.1f} ms/species")
