#!/usr/bin/env python
"""对 AmberTools MMPBSA.py 端到端对拍（T6 的参照物）。

    python scripts/validate_mmpbsa.py                 # S4 canonical 单帧
    python scripts/validate_mmpbsa.py --name 1YCR --frames 20 --stride 500
    python scripts/validate_mmpbsa.py --frames 5      # 取 MD 轨迹前 5 帧
    python scripts/validate_mmpbsa.py --keep          # 保留中间文件

**为什么是 MMPBSA.py 而不是 APBS。** APBS 是数值 FD+多重网格求解器（不是解析解），
与我们**同一个方法族**：同一个方程、同一类离散、都是格点有限差分。对上了只证明
「两边实现了同一套离散化」，不证明「答案对」。真正独立的检验只有两类：

  解析解        Born 离子 / Debye 屏蔽 —— `tests/test_pb.py` 里已有，那才是解析解。
  另一套实现栈  本脚本。Amber `pbsa`(不同离散、不同表面) + LCPO(不是 Shrake–Rupley)
                + sander 的 MM，**而且验的是整条 ΔG_MM/PBSA，不是三项里的一项**。

我们的参数本来就是 Amber 味的（mbondi2 半径、γ/β 取 INP=1），对照 Amber 才自洽。

**性能上不要拿任何一方比**：MMPBSA.py/APBS 都是单帧 CLI + 文件 I/O，不为轨迹批处理
设计。速度声明只在 RESULTS.md 的同构比较里做。

---

**能对齐的和不能对齐的，必须分开看**（这是本脚本唯一的难点）：

  能对齐   半径(radiopt=0 + 显式写 mbondi2)、h(scale=1/h)、ε_in/ε_out、离子强度、
           温度、γ/β(inp=1)、探针(prbrad)
  不能对齐 **PB 离散本身**(pbsa 的表面/边界/差分格式与我们不同)、
           **SA 的算法**(LCPO 解析近似 vs Shrake–Rupley 采样)

**判据必须跟着参照物走 —— 这是本脚本最容易用错的地方。**

  ΔE_MM     **正确性断言**。同一套力场参数、同样的 single-trajectory 假设，
            所以必须近似逐位一致。实测 ΔE_coul 5e-6、ΔE_LJ 2e-4。对不上是真 bug。
  ΔG_SA     **方法比较**。LCPO 解析近似 vs Shrake–Rupley 采样。实测 0.3% ——
            两个毫不相干的算法对到这个程度，是很强的 sanity check，但不是断言。
  ΔG_PB     **方法比较**。pbsa 与我们的离散、表面定义、能量泛函都不同。
            旧的共用 h=0.5 网格：842.6 vs 885.4（5.1%）；非对称网格默认（C/R 0.75、
            配体 0.25）20 帧：780.5 vs 778.5（**0.26%**，RESULTS §17.3）。

**不要拿 DESIGN §T6 的「ΔG_PB < 2%」卡这里。** 那条是给 APBS 写的 —— APBS 与我们
同方法族（同方程、同类离散、都是格点 FD），所以才能逐参数对齐到 2%。MMPBSA.py 不是。

下面这段「5% 不会收敛掉」的判断**已被 RESULTS §17 撤回**：两边的「收敛」都是单一摆放
下的加密，而我们欠解的是被 C/R 抵消掩盖的配体。保留原文作记录：

    pbsa   h=0.5 → 0.25   ΔG_PB 842.60 → 843.14   (0.06%)
    我们   h=0.75 → 0.5   ΔG_PB 变 5.49            (0.62%, RESULTS §0.17)

两边各自都收敛了，差的是方法本身。要把 PB 归因到具体哪一项（表面/边界/能量泛函），
用 APBS 的逐参数对齐，不要用这个脚本。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _amber_exe(name):
    """先找解释器自己的 bin/（conda 环境常常没 activate，PATH 上没有）。"""
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else shutil.which(name)


#: conda 环境即 AMBERHOME；MMPBSA.py 不设它会在 import 期就 TypeError
#: （commandlineparser 里 os.path.join(os.getenv('AMBERHOME'), ...)）。
AMBERHOME = os.environ.get("AMBERHOME") or os.path.dirname(
    os.path.dirname(_amber_exe("MMPBSA.py") or sys.executable))


def build_parms(workdir, radii, topology, system, positions_A, rec, lig):
    """OpenMM System -> 三个 prmtop。**半径显式写入，不让 pbsa 自己猜。**

    `radiopt=0` 让 pbsa 用 prmtop 里的 RADII；不写就是 parmed 的默认值，
    与我们的 mbondi2 不是同一组 —— 半径不同则表面不同，对拍当场失去意义。
    """
    import parmed

    st = parmed.openmm.load_topology(topology, system=system)
    st.coordinates = np.asarray(positions_A, float).reshape(-1, 3)

    # 被 HBonds 约束的 X–H 键在 System 里没有 HarmonicBondForce 项(907/1856),
    # parmed 给它们 bond.type = None, prmtop 写不出来。**填 k=0 而不是编一个力常数**:
    # single-trajectory 下 BOND/ANGLE/DIHED 在 ΔE_MM 里逐项抵消(没有键跨越 R–L 界面),
    # 所以这一项恒为 0, 不影响对拍; 编个 340 kcal/mol/Ų 反而是凭空捏参数。
    con = {}
    for i in range(system.getNumConstraints()):
        a1, a2, dist = system.getConstraintParameters(i)
        con[frozenset((a1, a2))] = dist.value_in_unit(parmed.unit.angstrom)
    zero = []
    for b in st.bonds:
        if b.type is None:
            req = con.get(frozenset((b.atom1.idx, b.atom2.idx)))
            bt = parmed.BondType(0.0, req if req is not None else 1.0)
            zero.append(bt)
            b.type = bt
    st.bond_types.extend(zero)
    st.bond_types.claim()
    for a, r in zip(st.atoms, np.asarray(radii, float)):
        a.solvent_radius = float(r)       # -> prmtop RADII
        a.screen = 0.85                   # GB 用，PB 路径不读，填个合法值
    st.parm_comments = getattr(st, "parm_comments", {})

    out = {}
    for name, idx in (("complex", np.arange(len(st.atoms))), ("receptor", rec),
                      ("ligand", lig)):
        sub = st[[int(i) for i in np.asarray(idx)]] if name != "complex" else st
        p = os.path.join(workdir, f"{name}.prmtop")
        amb = parmed.amber.AmberParm.from_structure(sub)
        amb.parm_data["RADIUS_SET"] = ["modified Bondi radii (mbondi2)"]
        amb.write_parm(p)
        out[name] = p
    return out


def write_traj(workdir, coords_A):
    """[B,N,3] Å -> Amber ASCII mdcrd（8 列 × %8.3f，不带盒子）。"""
    path = os.path.join(workdir, "traj.mdcrd")
    flat = np.asarray(coords_A, float).reshape(len(coords_A), -1)
    with open(path, "w") as fh:
        fh.write("jaxpbsa validation\n")
        for frame in flat:
            for i in range(0, len(frame), 10):
                fh.write("".join(f"{v:8.3f}" for v in frame[i:i + 10]) + "\n")
    return path


def write_input(workdir, h, eps_in, eps_out, istrng_M, gamma, beta, probe,
                extra=()):
    """&pb 的每一项都显式写出 —— 默认值随 AmberTools 版本变，靠默认就不是对拍。"""
    p = os.path.join(workdir, "mmpbsa.in")
    with open(p, "w") as fh:
        # MMPBSA.py 的 namelist 解析只认 '#' 注释, '!' 会被当成变量名 -> InputError。
        #
        # **每一项都必须显式写**, 因为 MMPBSA.py 14.0 的默认值和我们不是一套:
        #   exdi 默认 80.0  ≠ 我们的 78.5
        #   istrng 默认 0.0 ≠ 我们的 150 mM
        #   inp 默认 **2**  -> γ=0.0378, β=-0.5692 (我们是 INP=1 的 0.005, 0.0)
        # 这正是 DESIGN §3.10 里「论文必须写明用的是哪一档」那条的实证。
        #
        # scale = 1/h；radiopt=0 表示用 prmtop 里的 RADII(mbondi2) 而不是 pbsa 自选。
        # &pb 没有温度变量(pbsa 内部写死), 所以 T 只用于我们这边, 不传给它。
        more = "".join(f"  {kv},\n" for kv in extra)
        fh.write(f"""jaxpbsa <-> MMPBSA.py; parameters pinned explicitly, no defaults
&general
  startframe=1, verbose=2, keep_files=2,
/
&pb
  istrng={istrng_M:.4f}, indi={eps_in}, exdi={eps_out},
  scale={1.0 / h:.4f},
  radiopt=0,
  prbrad={probe},
  inp=1, cavity_surften={gamma}, cavity_offset={beta},
  fillratio=4.0,
{more}/
""")
    return p


_TERM = re.compile(r"^(EEL|VDWAALS|EPB|ENPOLAR|ECAVITY|EDISPER|DELTA TOTAL|TOTAL)\s+"
                   r"(-?\d+\.\d+)")


def parse_results(path):
    """从 FINAL_RESULTS_MMPBSA.dat 里取 DELTA 段的各项（kcal/mol）。"""
    txt = open(path).read()
    tail = txt[txt.index("Differences (Complex - Receptor - Ligand)"):] \
        if "Differences (Complex - Receptor - Ligand)" in txt else txt
    out = {}
    for line in tail.splitlines():
        m = _TERM.match(line.strip())
        if m and m.group(1) not in out:
            out[m.group(1)] = float(m.group(2))
    return out


def ours(coords_A, d, h, with_pb=True, shared=False):
    """我们这边的三项，取 frame 平均（与 MMPBSA.py 报告的 Average 对齐）。"""
    from jaxpbsa.mm.cross import mm_cross
    from jaxpbsa.openmm_io.system import extract_nonbonded
    from jaxpbsa.pb import PBParams, TripletSolver
    from jaxpbsa.sa import delta_g_sa

    c = np.asarray(coords_A, float)
    mm = mm_cross(c, extract_nonbonded(d["system"]), d["ligand_idx"],
                  d["receptor_idx"])
    dg_sa, areas = delta_g_sa(c, d["radii"], d["receptor_idx"], d["ligand_idx"])
    out = {"EEL": float(np.mean(mm["e_coul_rl"])),
           "VDWAALS": float(np.mean(mm["e_lj_rl"])),
           "ENPOLAR": float(np.mean(np.atleast_1d(dg_sa))),
           "_dsasa": float(np.mean(np.atleast_1d(areas["delta_sasa"])))}
    if with_pb:
        from openmm import unit
        m = np.array([d["system"].getParticleMass(i).value_in_unit(unit.dalton)
                      for i in range(len(d["charge"]))])
        # 默认非对称网格(C/R h, 配体 h=0.25 紧盒) + 质心归位; --shared 退回旧法
        sv = TripletSolver(c, m, d["radii"], d["receptor_idx"], d["ligand_idx"],
                           PBParams(swin=0.5, tol=1e-5, max_iter=8000,
                                    precond="mg"),
                           h=h, h_lig=None if shared else 0.25)
        per = [float(sv(f, d["charge"])["delta_g_pb"]) for f in c]
        out["EPB"] = float(np.mean(per))
        out["_epb_frames"] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="S4", help="体系: data/prepared/{name}_meta.json")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--h", type=float, default=0.5,
                    help="我们的 C/R 网格间距(配体固定 0.25 紧盒)")
    ap.add_argument("--pbsa-h", type=float, default=0.5,
                    help="pbsa 的网格间距(0.5 → 0.25 只动 0.06%%, 已收敛)")
    ap.add_argument("--shared", action="store_true",
                    help="旧法: C/R/L 共用一张 h 网格")
    ap.add_argument("--stride", type=int, default=1,
                    help="--frames>1 时每隔多少帧取一帧")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--pbopt", action="append", default=[], metavar="k=v",
                    help="额外写进 &pb 的旋钮，可重复（如 --pbopt ipb=1 --pbopt nfocus=1）")
    ap.add_argument("--no-pb", action="store_true",
                    help="跳过我们这边的 PB（只对 MM/SA，秒级）")
    a = ap.parse_args()

    from jaxpbsa.openmm_io import load_canonical
    from jaxpbsa.pb.energy import PBParams
    from jaxpbsa.sa import BETA_INP1, GAMMA_INP1

    d = load_canonical(a.name)
    pb = PBParams()
    coords = np.asarray(d["positions_A"], float)
    coords = coords[None] if coords.ndim == 2 else coords
    if a.frames > 1:
        import mdtraj as md
        t = md.load(os.path.join(ROOT, f"data/md/{a.name}_dry.dcd"),
                    top=os.path.join(ROOT, f"data/prepared/{a.name}_complex.pdb"))
        coords = (t.xyz[::a.stride][:a.frames] * 10.0).astype(np.float64)

    rec, lig = np.asarray(d["receptor_idx"]), np.asarray(d["ligand_idx"])
    if not (np.all(np.diff(rec) == 1) and np.all(np.diff(lig) == 1)):
        sys.exit("MMPBSA.py 要求 receptor/ligand 各自原子连续；当前索引不连续。")

    wd = a.workdir or tempfile.mkdtemp(prefix="jaxpbsa_mmpbsa_")
    os.makedirs(wd, exist_ok=True)
    print(f"工作目录 {wd}\nAMBERHOME {AMBERHOME}\n{len(coords)} 帧, 我们 h={a.h}"
          f"{' 共用网格' if a.shared else ' / 配体 0.25'}, pbsa h={a.pbsa_h}\n")

    parms = build_parms(wd, d["radii"], d["topology"], d["system"],
                        coords[0], rec, lig)
    traj = write_traj(wd, coords)
    # **istrng 单位是 M**, MMPBSA.py 内部再 ×1000 写成 pbsa 的 mM。
    # 传 150.0 会变成 150 M —— 比 0.15 M 高 1000 倍, 而且不报错。
    inp = write_input(wd, a.pbsa_h, pb.eps_in, pb.eps_out, pb.ionic_strength_M,
                      GAMMA_INP1, BETA_INP1, pb.probe_radius, a.pbopt)

    env = dict(os.environ, AMBERHOME=AMBERHOME)
    cmd = [_amber_exe("MMPBSA.py"), "-O", "-i", inp, "-o", "FINAL.dat",
           "-cp", parms["complex"], "-rp", parms["receptor"],
           "-lp", parms["ligand"], "-y", traj]
    r = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True)
    res_path = os.path.join(wd, "FINAL.dat")
    if r.returncode != 0 or not os.path.exists(res_path):
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        sys.exit(f"MMPBSA.py 失败 (rc={r.returncode})")

    mdin = os.path.join(wd, "_MMPBSA_pb.mdin")
    if os.path.exists(mdin):
        got = dict(l.split("=", 1) for l in open(mdin) if "=" in l)
        got = {k.strip(): v.strip() for k, v in got.items()}
        print("pbsa 实际吃到:", ", ".join(
            f"{k}={got[k]}" for k in ("istrng", "epsin", "epsout", "space",
                                      "dprob", "inp", "nfocus", "bcopt",
                                      "eneopt", "solvopt", "accept") if k in got))
        # **MMPBSA.py 14.0 只透传固定一批 &pb 变量**, 其余(ipb/nfocus/solvopt...)
        # namelist 解析器收下了但从不写进 mdin —— 静默丢弃, 不报错。
        dropped = [kv for kv in a.pbopt
                   if got.get(kv.split("=")[0].strip()) != kv.split("=")[1].strip()]
        if dropped:
            print(f"!! 这些 --pbopt 被 MMPBSA.py 丢弃了(没进 mdin): {dropped}")
    print()
    ref = parse_results(res_path)
    us = ours(coords, d, a.h, with_pb=not a.no_pb, shared=a.shared)
    rows = [("ΔE_coul", "EEL", "EEL", "同一套力场参数 —— **必须**近似逐位一致"),
            ("ΔE_LJ", "VDWAALS", "VDWAALS", "同上"),
            ("ΔG_PB", "EPB", "EPB", "方法比较，**不是断言**：离散/表面/泛函都不同"),
            ("ΔG_SA", "ENPOLAR", "ENPOLAR", "方法比较：LCPO 解析 ≠ Shrake–Rupley 采样")]
    print(f"{'项':<10}{'MMPBSA.py':>12}{'jaxpbsa':>12}{'Δ':>10}{'rel':>10}   判据")
    tot = [0.0, 0.0]
    for name, k, mine, note in rows:
        x, y = ref.get(k), us.get(mine)
        if x is None or y is None:  # 一侧缺（--no-pb）时仍要把另一侧印出来
            fx = "—" if x is None else f"{x:.3f}"
            fy = "—" if y is None else f"{y:.3f}"
            print(f"{name:<10}{fx:>12}{fy:>12}{'—':>10}{'—':>10}   {note}")
            continue
        tot[0] += x; tot[1] += y
        print(f"{name:<10}{x:>12.3f}{y:>12.3f}{y - x:>+10.3f}"
              f"{abs(y - x) / max(abs(x), 1e-12):>10.1e}   {note}")
    print(f"{'─' * 76}")
    print(f"{'ΔG_MM/PBSA':<10}{tot[0]:>12.3f}{tot[1]:>12.3f}{tot[1] - tot[0]:>+10.3f}"
          f"{abs(tot[1] - tot[0]) / max(abs(tot[0]), 1e-12):>10.1e}   端到端（差值由 PB 主导）")
    print("\n判据：只有 ΔE_MM 是正确性断言；ΔG_PB/ΔG_SA 是方法比较。见本脚本头注释。")
    print(f"\n我们的 ΔSASA = {us['_dsasa']:.2f} Ų; "
          f"MMPBSA.py 反推 {ref.get('ENPOLAR', float('nan')) / 0.005:.2f} Ų "
          f"(γ=0.005)")
    print(f"\n完整报告: {res_path}")
    if not a.keep and not a.workdir:
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()
