#!/usr/bin/env python
"""RESULTS §18.6 ② 的验证: C/R h=0.75 + 每帧随机亚格点偏移, 能否把冻结偏差变回噪声。

    python scripts/phase_jitter_check.py --name S4
    python scripts/phase_jitter_check.py --name 1YCR

同一批帧(与 grid_check / validate_mmpbsa 相同: 每 500 帧取一, 共 20 帧)跑三种 C/R 设置,
配体一律 h=0.25:
    frozen75  h=0.75, 质心归位(当前默认)
    jitter75  h=0.75, 质心归位 + 偏移 ∈ [0,h)³, 种子 = 帧号
    frozen50  h=0.5,  质心归位(参照)
判据: ⟨jitter75 − frozen50⟩ 在 ~2 以内(配对差, 报 SE)。
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MMPBSA.py ⟨ΔG_PB⟩, 同一批 20 帧(RESULTS §17.3 / §18.4)。frozen75 应复现 778.456 / 275.311
AMBER = {"S4": 780.518, "1YCR": 292.656}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="S4")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--stride", type=int, default=500)
    ap.add_argument("--surface", default="binary", choices=("binary", "fraction"))
    a = ap.parse_args()

    import jax
    import jaxpbsa
    jaxpbsa.enable_compilation_cache()
    import mdtraj as md
    from openmm import unit
    from jaxpbsa.openmm_io import load_canonical
    from jaxpbsa.pb import PBParams, TripletSolver

    d = load_canonical(a.name, root=ROOT)
    q, radii, rec, lig = d["charge"], d["radii"], d["receptor_idx"], d["ligand_idx"]
    m = np.array([d["system"].getParticleMass(i).value_in_unit(unit.dalton)
                  for i in range(len(q))])
    P = PBParams(swin=0.5, tol=1e-5, max_iter=8000, precond="mg", surface=a.surface)
    t = md.load(os.path.join(ROOT, f"data/md/{a.name}_dry.dcd"),
                top=os.path.join(ROOT, f"data/prepared/{a.name}_complex.pdb"))
    idx = np.arange(t.n_frames)[::a.stride][:a.frames]
    F = (t.xyz[idx] * 10.0).astype(np.float64)
    print(f"{a.name}: {len(F)} frames {idx.tolist()} | surface {a.surface} | {jax.devices()}",
          flush=True)

    tri75 = TripletSolver(F, m, radii, rec, lig, P, h=0.75)
    tri50 = TripletSolver(F, m, radii, rec, lig, P, h=0.5)
    runs = {
        "frozen75": lambda i, f: tri75(f, q),
        "jitter75": lambda i, f: tri75(f, q, shift=np.random.default_rng(int(i)).uniform(0, 0.75, 3)),
        "frozen50": lambda i, f: tri50(f, q),
    }
    res = {}
    for tag, fn in runs.items():
        fn(idx[0], F[0])  # 编译
        t0 = time.perf_counter()
        o = [fn(i, f) for i, f in zip(idx, F)]
        ms = (time.perf_counter() - t0) / len(F) * 1e3
        res[tag] = np.array([float(x["delta_g_pb"]) for x in o])
        assert all(bool(x["converged"]) for x in o), f"{tag}: 有帧未收敛"
        assert min(x["margin_A"] for x in o) > 0, f"{tag}: C/R 网格丢原子"
        print(f"{tag}: <dG_PB> {res[tag].mean():8.3f}  帧间 sd {res[tag].std(ddof=1):6.2f}  "
              f"{ms:6.1f} ms/帧", flush=True)

    ref = res["frozen50"]
    for tag in ("frozen75", "jitter75"):
        dv = res[tag] - ref
        se = dv.std(ddof=1) / np.sqrt(dv.size)
        print(f"{tag} - frozen50: {dv.mean():+7.3f} ± {se:.3f} (SE)  逐帧 sd {dv.std(ddof=1):.2f}",
              flush=True)
    if a.name in AMBER and (a.frames, a.stride) == (20, 500):
        for tag, v in res.items():
            print(f"{tag} vs MMPBSA.py {AMBER[a.name]}: {v.mean() - AMBER[a.name]:+7.3f} "
                  f"({(v.mean() / AMBER[a.name] - 1) * 100:+.2f}%)", flush=True)
    dv = res["jitter75"] - ref
    print(f"判据 |<jitter75 - frozen50>| < 2: {'PASS' if abs(dv.mean()) < 2 else 'FAIL'}")
    np.savetxt(os.path.join(ROOT, f"data/md/{a.name}_phase_jitter_{a.surface}.txt"),
               np.column_stack([idx, res["frozen75"], res["jitter75"], ref]),
               header="frame frozen75 jitter75 frozen50", fmt=["%d", "%.4f", "%.4f", "%.4f"])


if __name__ == "__main__":
    main()
