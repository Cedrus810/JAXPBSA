#!/usr/bin/env python
"""S6 (ONLINE_PLAN): MD 与在线 PBSA **同卡争用**实测。

    python scripts/online_overhead.py                       # 全部配置
    python scripts/online_overhead.py --steps 20000         # 短冒烟
    python scripts/online_overhead.py --config 5000         # 单个配置
    python scripts/online_overhead.py --order jax           # 先预热 JAX 再建 OpenMM Context

配置 = reporter 间隔: baseline(无 reporter) / 5000 / 10000 / 25000 步
(plan §21.5 约束 4 的表期望 18% / 9% / 3.6%, 前提 t_step=247 µs @ 2080 Ti)。

**每个配置独立子进程**(benchmark.py 陷阱 1: 同进程顺序跑多个配置, 编译产物和
显存单调累积, 结论会错)。

**必须 `XLA_PYTHON_CLIENT_PREALLOCATE=false`** —— JAX 默认预占 75% 显存,
OpenMM 的 CUDA Context 会起不来或 OOM。本脚本在 import jax 前设好; 也试
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` 这类限额(两顺序都测, 见 --order)。

读法: 干净串行时实测应**对上**公式 `overhead ≈ T_pbsa / (N·t_step)`; 明显更差
= 瓶颈在显存/上下文切换, 那时才轮到 plan §21 的双 GPU async worker —— 在这个
数出来之前不做 async。

**口径注意**: 本脚本是 **2 fs 无 HMR**(与 run_s4_md.py 一致), 而 plan §21.5
表里的 t_step = 247 µs 是 **4 fs + HMR** —— overhead **百分比**可比(公式两边
同除 t_step), 但「每次查询覆盖多少 ps」差 2 倍。写结论时两个 t_step 不许混在
同一个数里。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 run_s4_md.py 一致(同成本结构才有可比的 t_step)
T_K, DT_PS, FRICTION = 298.15, 0.002, 1.0
INTERVALS = (5000, 10000, 25000)


def _worker(cfg: str, steps: int, order: str) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import numpy as np
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    pdb = app.PDBFile(os.path.join(ROOT, "data", "md", "S4_solvated.pdb"))
    state0 = mm.XmlSerializer.deserialize(
        open(os.path.join(ROOT, "data", "md", "S4_prod_final.xml")).read())
    ff = app.ForceField("amber/ff14SB.xml", "amber/phosaa14SB.xml",
                        "amber/tip3p_standard.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds)
    pos_A = np.asarray(state0.getPositions(asNumpy=True).value_in_unit(
        unit.angstrom), dtype=np.float64)

    # 溶质 = 溶剂化前的干体系(Modeller.addSolvent 把水/离子追加在尾部)
    import json
    meta = json.load(open(os.path.join(ROOT, "data", "prepared", "S4_meta.json")))
    n_dry = int(meta["n_atoms"])
    solute_idx = np.arange(n_dry)
    lig_local = np.asarray(meta["ligand_idx"])  # 干体系里本就是 0 基局部索引

    analyzer = reporter = None
    if order == "jax" and cfg != "baseline":
        import jaxpbsa
        jaxpbsa.enable_compilation_cache()
        from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
        analyzer = OnlineMMPBSA(system, pdb.topology, solute_idx, lig_local,
                                 pos_A[solute_idx])
        # margin_min=12: S4 的 1.5κ⁻¹(RESULTS §15.8) —— 只查丢原子(0)不够,
        # 边界贴脸的帧数值上不丢原子, ΔG_PB 的边界误差已超标
        reporter = PBSAReporter(analyzer, int(cfg), solute_idx, margin_min=12.0)

    integrator = mm.LangevinMiddleIntegrator(
        T_K, FRICTION / unit.picosecond, DT_PS * unit.picoseconds)
    sim = app.Simulation(pdb.topology, system, integrator,
                         mm.Platform.getPlatformByName("CUDA"))
    sim.context.setState(state0)

    if order != "jax" and cfg != "baseline":
        import jaxpbsa
        jaxpbsa.enable_compilation_cache()
        from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
        analyzer = OnlineMMPBSA(system, pdb.topology, solute_idx, lig_local,
                                 pos_A[solute_idx])
        # margin_min=12: S4 的 1.5κ⁻¹(RESULTS §15.8) —— 只查丢原子(0)不够,
        # 边界贴脸的帧数值上不丢原子, ΔG_PB 的边界误差已超标
        reporter = PBSAReporter(analyzer, int(cfg), solute_idx, margin_min=12.0)
    if reporter is not None:
        sim.reporters.append(reporter)

    t0 = time.perf_counter()
    sim.step(steps)
    wall = time.perf_counter() - t0
    ns_day = steps * DT_PS / 1000.0 / wall * 86400.0
    pbsa_s = reporter.total_pbsa_s if reporter is not None else 0.0
    n_rep = reporter.n_reports if reporter is not None else 0
    per_frame = (pbsa_s / n_rep * 1e3) if n_rep else 0.0
    warmup = analyzer.warmup_time_s if analyzer is not None else 0.0
    print(f"RESULT {cfg} steps={steps} order={order} wall={wall:.2f}s "
          f"ns_day={ns_day:.1f} t_step_us={wall / steps * 1e6:.1f} "
          f"pbsa_total={pbsa_s:.2f}s reports={n_rep} "
          f"pbsa_ms_per_report={per_frame:.1f} analyzer_warmup={warmup:.1f}s",
          flush=True)
    if reporter is not None:
        reporter.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--config", choices=["baseline", *[str(i) for i in INTERVALS]])
    ap.add_argument("--order", choices=["openmm", "jax"], default="openmm")
    ap.add_argument("--worker", type=str, default=None)  # 内部: 子进程模式
    args = ap.parse_args()

    if args.worker:
        _worker(args.worker, args.steps, args.order)
        return

    cfgs = [args.config] if args.config else ["baseline", *map(str, INTERVALS)]
    for cfg in cfgs:
        for order in (["openmm", "jax"] if cfg != "baseline" else ["openmm"]):
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 "--worker", cfg, "--steps", str(args.steps), "--order", order],
                cwd=ROOT)
            if r.returncode != 0:
                print(f"config {cfg}/{order} 失败 (exit {r.returncode})", file=sys.stderr)


if __name__ == "__main__":
    main()
