#!/usr/bin/env python
"""默认 binary 路径的网格误差：C/R 网格与配体网格**分开**加密，逐帧配对误差。

    python scripts/grid_error.py --name S4 --frames 2                       # 冒烟
    python scripts/grid_error.py --name 1YCR --frames 20 --stride 500 --phases 3 --out 1ycr_grid.npz

ΔG_PB = (G_C − G_R)(C/R 网格) − G_L(配体网格)，两部分各自只依赖自己的网格：
  C/R 扫描   h ∈ --h-cr，配体固定 --h-lig-fixed；误差 = (C−R)(h) − (C−R)(最细 h)
  L   扫描   h_lig ∈ --h-lig，C/R 固定 --h-cr-fixed；误差 = −[G_L(h) − G_L(最细)]
逐帧误差的 bias / MAE / RMSE / max，加上 C/R 亚格点相位敏感性（--phases > 1 时每帧 K 个
随机平移，种子 = 帧号×1000+k；第 0 个相位不平移 = 默认行为）。参考值是列表里最细的 h，
不是连续解 —— 报的是相对最细网格的误差。

任一帧未收敛或 margin 越限直接失败（判据同在线 reporter），不进统计。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _check(r, margin_flag, tag):
    if not bool(np.asarray(r["converged"])):
        return f"{tag}: PB 未收敛 (iters {np.asarray(r['iters']).tolist()})"
    if r["margin_A"] < margin_flag:
        return f"{tag}: margin_A = {r['margin_A']:.2f} Å < {margin_flag:.2f}"
    if r["margin_lig_A"] < 0:
        return f"{tag}: margin_lig_A = {r['margin_lig_A']:.2f} Å < 0"
    return None


def run(coords, m, d, h, h_lig, shifts):
    """返回 [F, K] 的 (C−R, G_L) 和每帧耗时。"""
    from jaxpbsa.online import boundary_margin_flag
    from jaxpbsa.pb import PBParams, TripletSolver

    t0 = time.perf_counter()
    sv = TripletSolver(coords, m, d["radii"], d["receptor_idx"], d["ligand_idx"],
                       PBParams(), h=h, h_lig=h_lig)
    flag = boundary_margin_flag(sv.params)
    F, K = shifts.shape[:2]
    cr, gl, bad = np.empty((F, K)), np.empty((F, K)), []
    t_first = None
    for i in range(F):
        for k in range(K):
            r = sv(coords[i], d["charge"], shift=shifts[i, k] if k else None)
            why = _check(r, flag, f"h={h} h_lig={h_lig} 帧 {i} 相位 {k}")
            if why:
                bad.append(why)
            cr[i, k] = float(r["g_pb_complex"]) - float(r["g_pb_receptor"])
            gl[i, k] = float(r["g_pb_ligand"])
            if t_first is None:
                t_first = time.perf_counter() - t0
    if bad:
        sys.exit(f"{len(bad)} 个解不可用，拒绝出统计:\n  " + "\n  ".join(bad))
    n = F * K
    per = (time.perf_counter() - t0 - t_first) / max(n - 1, 1)
    print(f"  h={h:<5} h_lig={h_lig:<5} 首解(含编译) {t_first:6.1f} s, 稳态 {per * 1e3:7.0f} ms/解",
          flush=True)
    return cr, gl, t_first, per


def stats(err):
    e = err.ravel()
    return e.mean(), np.abs(e).mean(), np.sqrt((e ** 2).mean()), np.abs(e).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="S4")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--phases", type=int, default=1)
    ap.add_argument("--h-cr", type=float, nargs="+", default=[0.5, 0.4])
    ap.add_argument("--h-lig", type=float, nargs="+", default=[0.25, 0.2])
    ap.add_argument("--h-cr-fixed", type=float, default=0.5)
    ap.add_argument("--h-lig-fixed", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="逐帧原始值存 .npz")
    a = ap.parse_args()

    import mdtraj as md
    from openmm import unit
    from jaxpbsa.openmm_io import load_canonical

    d = load_canonical(a.name)
    t = md.load(os.path.join(ROOT, f"data/md/{a.name}_dry.dcd"),
                top=os.path.join(ROOT, f"data/prepared/{a.name}_complex.pdb"))
    coords = (t.xyz[::a.stride][:a.frames] * 10.0).astype(np.float64)
    m = np.array([d["system"].getParticleMass(i).value_in_unit(unit.dalton)
                  for i in range(len(d["charge"]))])
    F, K = len(coords), a.phases
    print(f"{a.name}: {F} 帧 × {K} 相位; C/R 扫 {a.h_cr}（L {a.h_lig_fixed}）, "
          f"L 扫 {a.h_lig}（C/R {a.h_cr_fixed}）", flush=True)

    def shifts_for(h, k=K):
        rng = [np.random.default_rng(i * 1000) for i in range(F)]
        return np.stack([np.stack([g.uniform(0, h, 3) for _ in range(k)]) for g in rng])

    res, timing = {}, {}
    for h in sorted(a.h_cr, reverse=True):
        cr, gl, tf, per = run(coords, m, d, h, a.h_lig_fixed, shifts_for(h))
        res[("cr", h)] = cr
        timing[("cr", h)] = (tf, per)
        if h == a.h_cr_fixed:  # 这一组同时就是 L 扫描里 h_lig_fixed 那一组
            res[("l", a.h_lig_fixed)] = gl
    for hl in sorted(a.h_lig, reverse=True):
        if ("l", hl) in res:
            continue
        # 平移只作用于 C/R 网格, G_L 与相位无关 -> L 扫描只算一个相位
        cr, gl, tf, per = run(coords, m, d, a.h_cr_fixed, hl, shifts_for(a.h_cr_fixed, 1))
        res[("l", hl)] = gl
        timing[("l", hl)] = (tf, per)
    print()

    def table(kind, hs, sign):
        ref_h = min(hs)
        ref = res[(kind, ref_h)].mean(axis=1)
        print(f"{'C/R' if kind == 'cr' else 'L'} 扫描对 ΔG_PB 的误差（参考 h = {ref_h}，kcal/mol，"
              f"{f'{K} 相位均值' if K > 1 else '单相位'}）")
        print(f"  {'h':>6} {'bias':>9} {'MAE':>9} {'RMSE':>9} {'max':>9} {'相位 sd':>9}")
        for h in sorted(hs, reverse=True):
            v = res[(kind, h)]
            b, mae, rmse, mx = stats(sign * (v.mean(axis=1) - ref))
            psd = float(np.mean(v.std(axis=1, ddof=1))) if v.shape[1] > 1 else float("nan")
            print(f"  {h:>6} {b:>+9.3f} {mae:>9.3f} {rmse:>9.3f} {mx:>9.3f} {psd:>9.3f}")
        print()

    table("cr", a.h_cr, +1.0)
    table("l", a.h_lig, -1.0)   # ΔG_PB 里 G_L 带负号
    if a.out:
        np.savez(a.out, frames=np.arange(F) * a.stride, phases=K,
                 **{f"{k}_{h}": v for (k, h), v in res.items()})
        print(f"逐帧原始值 -> {a.out}")


if __name__ == "__main__":
    main()
