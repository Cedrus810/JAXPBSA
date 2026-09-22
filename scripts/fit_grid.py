#!/usr/bin/env python
"""从试跑轨迹定 PB 网格尺寸 —— **换体系时每次都要重跑这一遍**。

    python scripts/fit_grid.py --traj data/md/S4_dry.dcd --top data/prepared/S4_complex.pdb

在线模式的网格全程固定（ONLINE_PLAN §0 契约 1），所以尺寸必须一次定对：
多一档 dime 白损失 27–35% 吞吐，少一档整条轨迹的 margin 低于阈值、要重跑。
本脚本把这个决定从「拍一个 padding」变成算出来的。

三个输入的泛化能力完全不同（RESULTS.md §16.9）：

    reach = r_max + probe + ion + swin   从体系自己的半径现算        自动正确
    margin_min ≈ 1.5·κ⁻¹                 从离子强度现算              自动正确
    构象涨落                              **不泛化** —— 只能从轨迹量

第三项正是单帧看不见的那一截：S4 上单帧比 10 ns 轨迹欠 **3.79 Å**，按单帧定尺跑满
10 ns 会掉到 margin +7.44（低于物理阈值 12）。所以本脚本同时打印**前缀扫描**，
让你看见自己的试跑够不够长 —— 需求是 max 统计量，单调增、不会真正收敛，
1 ns 在 S4 上够用，**这个结论没有在别的体系上验过**。

与 SA 的 `k_neighbors` 是同一个判决（RESULTS §11.2）：不自动探测，显式给 + 每帧硬验。
"""
from __future__ import annotations

import argparse

import numpy as np

from jaxpbsa.constants import debye_kappa2
from jaxpbsa.openmm_io import assign_radii
from jaxpbsa.pb import PBParams
from jaxpbsa.pb.grid import APBS_DIME


def fit_dims(need_half: np.ndarray, h: float) -> tuple[int, ...]:
    """每轴取满足半长的最小 APBS dime。"""
    out = []
    for v in need_half:
        n = int(np.ceil(2 * v / h)) + 1
        out.append(next(d for d in APBS_DIME if d >= n))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="试跑轨迹 (mdtraj 能读的任何格式)")
    ap.add_argument("--top", required=True, help="拓扑 (pdb/cif)，原子序须与轨迹一致")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--ionic", type=float, default=PBParams.ionic_strength_M)
    ap.add_argument("--radii-model", default="mbondi2")
    ap.add_argument("--margin-min", type=float, default=None,
                    help="默认由 --ionic 算 1.5·κ⁻¹")
    ap.add_argument("--safety", type=float, default=0.0,
                    help="额外余量 (Å)，给「试跑比生产短」买保险")
    a = ap.parse_args()

    import mdtraj as md

    t = md.load(a.traj, top=a.top)
    xyz = t.xyz * 10.0  # nm -> Å
    mass = np.array([at.element.mass for at in t.topology.atoms])
    com = (mass[None, :, None] * xyz).sum(1) / mass.sum()   # 与 online.recenter_com 同约定
    half = np.abs(xyz - com[:, None, :]).max(1)             # [F,3] 每帧每轴半长

    p = PBParams(ionic_strength_M=a.ionic)
    radii = assign_radii(t.topology.to_openmm(), a.radii_model)
    reach = float(np.max(radii)) + p.probe_radius + p.ion_radius + max(p.swin, 0.0)
    k2 = debye_kappa2(p.ionic_strength_M, p.eps_out, p.temperature_K)
    kinv = float(1.0 / np.sqrt(k2)) if k2 > 0 else float("inf")
    margin_min = a.margin_min if a.margin_min is not None else 1.5 * kinv

    print(f"{t.n_frames} 帧 / {t.n_atoms} 原子 | h={a.h} | I={a.ionic} M "
          f"-> κ⁻¹={kinv:.2f} Å")
    print(f"reach = r_max {np.max(radii):.2f} + probe {p.probe_radius} + ion "
          f"{p.ion_radius} + swin {p.swin} = {reach:.2f} Å")
    print(f"margin_min = {margin_min:.2f} Å"
          + ("" if a.margin_min is not None else "  (= 1.5·κ⁻¹)"))
    print(f"\n每轴半长 max|x−COM|: 帧最大 {np.round(half.max(0), 2)} | "
          f"轨迹内涨落 {np.round(half.max(0) - half.min(0), 2)}")

    need = half.max(0) + reach + margin_min + a.safety
    dims = fit_dims(need, a.h)
    got = 0.5 * (np.array(dims) - 1) * a.h
    worst = float((got - half.max(0) - reach).min())
    print(f"需要的半长 = 帧最大 + reach + margin_min"
          f"{' + safety' if a.safety else ''} = {np.round(need, 2)}")
    print(f"\n  -> 网格 {dims}  {np.prod(dims)/1e6:.2f} M 节点  "
          f"半长 {got}  全轨迹最小 margin {worst:+.2f} Å")

    # 给现行接口用的等价 padding: make_grid 按**参考帧包围盒**算 need
    ref_half_bb = 0.5 * (xyz[0].max(0) - xyz[0].min(0))
    pad = float(np.max(need - ref_half_bb))
    print(f"  -> 现行接口: OnlineMMPBSA(..., h={a.h}, padding={pad:.2f})  "
          f"(以轨迹第 0 帧为 ref_coords_A 时等价)")

    print("\n前缀扫描 —— 你的试跑够长吗（需求是 max 统计量，单调增）:")
    print(f"  {'用多少帧':>10}  {'所需半长':>8}  {'欠全轨迹':>8}  {'定出的网格':>16}  "
          f"{'全轨迹最小 margin':>16}")
    ks = [k for k in (1, 10, 100, 1000, 5000, 10000) if k < t.n_frames] + [t.n_frames]
    for k in ks:
        v = half[:k].max(0)
        d = fit_dims(v + reach + margin_min + a.safety, a.h)
        g = 0.5 * (np.array(d) - 1) * a.h
        w = float((g - half.max(0) - reach).min())
        print(f"  {k:>10}  {v.max():>8.2f}  {half.max() - v.max():>+8.2f}  "
              f"{str(d):>16}  {w:>+12.2f} Å  "
              f"{'OK' if w >= margin_min else '** < margin_min **'}")


if __name__ == "__main__":
    main()
