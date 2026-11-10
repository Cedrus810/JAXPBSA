#!/usr/bin/env python
"""S6 (ONLINE_PLAN): MD 与在线 PBSA **同卡争用**实测。

    python scripts/online_overhead.py                       # 全部配置
    python scripts/online_overhead.py --steps 20000         # 短冒烟
    python scripts/online_overhead.py --config 5000         # 单个配置
    python scripts/online_overhead.py --order jax           # 先预热 JAX 再建 OpenMM Context
    python scripts/online_overhead.py --mode both           # 同步 vs 常驻后台 worker
    python scripts/online_overhead.py --mode both --mps-percentage 20 --mps-priority 1
    python scripts/online_overhead.py --hou-run-id a1_ph74_gaff2_rep01 --config 5000 --grid h0.4 ...

`--hou-run-id`(docs/HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md §3): 小分子包里一个 Hou 副本的体系,
与生产入口 run_smallmol_md.py 对齐 —— setup/system.xml + MonteCarloBarostat(1 atm, 298.15 K)、CUDA mixed
精度、从该副本 npt_end.xml 出发、pilot = npt_solute_A.npy、ref = pilot[-1]。不给时 S4 路径不变。
`--grid h0.4` 只改 h/h_lig(0.4/0.2, 同 score_smallmol_replica.py 的 GRIDS)。`--cache-dir` 指定 XLA
编译缓存目录(冷启动 / 命中缓存分开测)。

配置 = reporter 间隔: baseline(无 reporter) / 5000 / 10000 / 25000 步
(plan §21.5 约束 4 的表期望 18% / 9% / 3.6%, 前提 t_step=247 µs @ 2080 Ti)。

**每个配置独立子进程**(benchmark.py 陷阱 1: 同进程顺序跑多个配置, 编译产物和
显存单调累积, 结论会错)。

**必须 `XLA_PYTHON_CLIENT_PREALLOCATE=false`** —— JAX 默认预占 75% 显存,
OpenMM 的 CUDA Context 会起不来或 OOM。本脚本在 import jax 前设好; 也试
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` 这类限额(两顺序都测, 见 --order)。

同步口径: 实测应对上 `overhead ≈ T_pbsa / (N·t_step)`。异步模式分析器只在子进程
构造, 两进程共用当前可见的 GPU; 报告 MD 吞吐、结束后的排空耗时和结果延迟。
MPS 必须在启动脚本前启用(同一次 daemon 下比 baseline/sync/async), 线程限额与
低优先级只给 PBSA worker。串行旧实测不能证明后台并发是否有收益。

**口径注意**: 本脚本是 **2 fs 无 HMR**(与 run_s4_md.py 一致), 而 plan §21.5
表里的 t_step = 247 µs 是 **4 fs + HMR** —— overhead **百分比**可比(公式两边
同除 t_step), 但「每次查询覆盖多少 ps」差 2 倍。写结论时两个 t_step 不许混在
同一个数里。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 run_s4_md.py 一致(同成本结构才有可比的 t_step)
T_K, DT_PS, FRICTION = 298.15, 0.002, 1.0
INTERVALS = (5000, 10000, 25000)
GRIDS = {"default": {}, "h0.4": dict(h=0.4, h_lig=0.2)}
PACKAGE = os.path.join(ROOT, "experiments", "2026-10-17_small_molecule_benchmark")


def _hou_system(run_id: str):
    """Hou replica as run by run_smallmol_md.py: (system, topology, state0, solute_idx, lig_local,
    pilot_A, platform_properties).  Read-only."""
    import csv
    import json
    import numpy as np
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    run = next(r for r in csv.DictReader(open(os.path.join(PACKAGE, "manifests", "runs.csv")))
               if r["run_id"] == run_id)
    inp = next(r for r in csv.DictReader(open(os.path.join(PACKAGE, "manifests", "hou_inputs.csv")))
               if r["sample"] == run["sample_id"])
    rep = os.path.join(PACKAGE, run["output_directory"])
    setup = os.path.join(os.path.dirname(rep), "setup")
    srec = json.load(open(os.path.join(setup, "setup_record.json")))
    system = mm.XmlSerializer.deserialize(open(os.path.join(setup, "system.xml")).read())
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.atmosphere, T_K * unit.kelvin))
    state0 = mm.XmlSerializer.deserialize(open(os.path.join(rep, "npt_end.xml")).read())
    top = app.AmberPrmtopFile(inp["prmtop"]).topology
    pilot = np.load(os.path.join(rep, "npt_solute_A.npy")).astype(np.float64)
    n_sol = int(srec["n_solute_atoms"])
    return (system, top, state0, np.arange(n_sol), np.arange(*srec["ligand_local_idx"]), pilot,
            {"CudaPrecision": "mixed"})


def _verify_mps_clients(expected_pids):
    """Verify benchmark processes in MPS v2's client list, not just its daemon.

    Run after the OpenMM Context and (if used) PBSA worker are initialized.
    PID membership is process-level evidence, not per-Context GPU utilization.
    """
    control = shutil.which("nvidia-cuda-mps-control")
    if control is None:
        raise RuntimeError("Cannot verify MPS clients: nvidia-cuda-mps-control is unavailable")

    def query(command):
        try:
            result = subprocess.run(
                [control], input=command + "\n", text=True,
                capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Cannot verify MPS clients: {command}") from exc
        tokens = result.stdout.split()
        if result.returncode or any(not token.isdecimal() for token in tokens):
            raise RuntimeError(
                f"MPS query {command!r} failed: {result.stdout} {result.stderr}")
        return {int(token) for token in tokens}

    servers = query("get_server_list")
    clients = set()
    for server in sorted(servers):
        clients.update(query(f"get_client_list {server}"))
    missing = set(expected_pids) - clients
    if missing:
        raise RuntimeError(
            f"MPS client verification failed: PIDs {sorted(missing)} missing from "
            f"client list {sorted(clients)}; start MPS before creating CUDA contexts "
            "and use the same CUDA_MPS_PIPE_DIRECTORY and PID namespace")
    print(f"MPS_VERIFIED servers={sorted(servers)} clients={sorted(expected_pids)}", flush=True)


def _pilot():
    """网格定尺用的试跑轨迹: S4 干轨迹前 1 ns(1000 帧)。RESULTS §16.9: 1 ns 前缀定出的
    193³ 在全 10 ns 上最小 margin +15.4; 0.1 ns 不够。"""
    import mdtraj as md
    import numpy as np
    t = md.load(os.path.join(ROOT, "data", "md", "S4_dry.dcd"),
                top=os.path.join(ROOT, "data", "prepared", "S4_complex.pdb"))
    return (t.xyz[:1000] * 10.0).astype(np.float64)


def _worker(cfg: str, steps: int, order: str, mode: str,
            mps_percentage: int | None, mps_priority: int | None,
            hou_run_id: str | None = None, grid: str = "default", cache_dir: str | None = None) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import numpy as np
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    if hou_run_id:
        system, topology, state0, solute_idx, lig_local, pilot, props = _hou_system(hou_run_id)
        ref_A = pilot[-1]
    else:
        pdb = app.PDBFile(os.path.join(ROOT, "data", "md", "S4_solvated.pdb"))
        topology = pdb.topology
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
        pilot, ref_A, props = None, pos_A[solute_idx], {}
    if cfg != "baseline" and pilot is None:
        pilot = _pilot()
    analyzer_kwargs = dict(pilot_coords_A=pilot, **GRIDS[grid])

    def make_reporter():
        if mode == "async":
            from jaxpbsa.online_async import AsyncPBSAReporter
            return AsyncPBSAReporter.from_system(
                system, topology, solute_idx, lig_local, ref_A,
                interval_steps=int(cfg), analyzer_kwargs=analyzer_kwargs,
                mps_thread_percentage=mps_percentage, mps_priority=mps_priority,
                max_pending=2, cache_dir=cache_dir)
        import jaxpbsa
        jaxpbsa.enable_compilation_cache(cache_dir)
        from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
        analyzer = OnlineMMPBSA(system, topology, solute_idx, lig_local,
                                 ref_A, **analyzer_kwargs)
        return PBSAReporter(analyzer, int(cfg), solute_idx)

    reporter = None
    try:
        if order == "jax" and cfg != "baseline":
            reporter = make_reporter()
        integrator = mm.LangevinMiddleIntegrator(
            T_K, FRICTION / unit.picosecond, DT_PS * unit.picoseconds)
        sim = app.Simulation(topology, system, integrator,
                             mm.Platform.getPlatformByName("CUDA"), props)
        sim.context.setState(state0)
        if order != "jax" and cfg != "baseline":
            reporter = make_reporter()
        if reporter is not None:
            sim.reporters.append(reporter)
        if mps_percentage is not None or mps_priority is not None:
            # Force OpenMM GPU initialization before checking client membership,
            # outside the MD timer. Check baseline/sync too for a fair comparison.
            sim.context.getState(getEnergy=True)
            expected_pids = {os.getpid()}
            if mode == "async" and reporter is not None:
                expected_pids.add(reporter._process.pid)
            _verify_mps_clients(expected_pids)
        t0 = time.perf_counter()
        sim.step(steps)
        wall = time.perf_counter() - t0
    finally:
        # Drain outside the MD timer, but report it separately. Do not make an
        # async benchmark appear faster by abandoning its last accepted frames.
        drain_start = time.perf_counter()
        if reporter is not None:
            if mode == "async":
                reporter.close(timeout=reporter.close_timeout)
            else:
                reporter.close()
        drain_s = time.perf_counter() - drain_start
    ns_day = steps * DT_PS / 1000.0 / wall * 86400.0
    pbsa_s = reporter.total_pbsa_s if reporter is not None else 0.0
    n_rep = reporter.n_reports if reporter is not None else 0
    per_frame = (pbsa_s / n_rep * 1e3) if n_rep else 0.0
    warmup = (getattr(reporter, "warmup_time_s", None) if reporter is not None else 0.0)
    if warmup is None:
        warmup = reporter.analyzer.warmup_time_s
    print(f"RESULT {cfg} steps={steps} order={order} mode={mode} wall={wall:.2f}s "
          f"ns_day={ns_day:.1f} t_step_us={wall / steps * 1e6:.1f} "
          f"pbsa_total={pbsa_s:.2f}s reports={n_rep} "
          f"pbsa_ms_per_report={per_frame:.1f} analyzer_warmup={warmup:.1f}s "
          f"drain_s={drain_s:.3f} total_ns_day={steps * DT_PS / 1000 / (wall + drain_s) * 86400:.1f} "
          f"max_latency_s={getattr(reporter, 'max_latency_s', 0.0):.3f} "
          f"submit_s={getattr(reporter, 'total_submit_s', 0.0):.3f} "
          f"worker_mps_percentage={mps_percentage if mode == 'async' and reporter else None} "
          f"worker_mps_priority={mps_priority if mode == 'async' and reporter else None} "
          f"system={hou_run_id or 'S4'} grid={grid} expected={steps // int(cfg) if cfg != 'baseline' else 0} "
          f"submitted={getattr(reporter, 'n_submitted', n_rep)} skipped={getattr(reporter, 'n_skipped', 0)} "
          f"max_pending={getattr(reporter, 'max_pending', None)}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--config", choices=["baseline", *[str(i) for i in INTERVALS]])
    ap.add_argument("--order", choices=["openmm", "jax"], default="openmm")
    ap.add_argument("--mode", choices=["sync", "async", "both"], default="sync")
    ap.add_argument("--mps-percentage", type=int, choices=range(1, 101), metavar="1..100")
    ap.add_argument("--mps-priority", type=int, choices=[0, 1])
    ap.add_argument("--hou-run-id", default=None, help="Hou replica of the small-molecule package (default: S4)")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="default")
    ap.add_argument("--cache-dir", default=None, help="XLA compilation cache directory")
    ap.add_argument("--worker", type=str, default=None)  # 内部: 子进程模式
    args = ap.parse_args()
    if args.mps_percentage is not None and args.mps_priority is None:
        args.mps_priority = 1

    if args.worker:
        _worker(args.worker, args.steps, args.order, args.mode,
                args.mps_percentage, args.mps_priority, args.hou_run_id, args.grid, args.cache_dir)
        return

    cfgs = [args.config] if args.config else ["baseline", *map(str, INTERVALS)]
    modes = ["sync", "async"] if args.mode == "both" else [args.mode]
    failed = False
    for mode in modes:
        for cfg in cfgs:
            for order in (["openmm", "jax"] if cfg != "baseline" else ["openmm"]):
                command = [sys.executable, os.path.abspath(__file__),
                           "--worker", cfg, "--steps", str(args.steps),
                           "--order", order, "--mode", mode]
                if args.mps_percentage is not None:
                    command += ["--mps-percentage", str(args.mps_percentage)]
                if args.mps_priority is not None:
                    command += ["--mps-priority", str(args.mps_priority)]
                if args.hou_run_id:
                    command += ["--hou-run-id", args.hou_run_id]
                command += ["--grid", args.grid]
                if args.cache_dir:
                    command += ["--cache-dir", args.cache_dir]
                r = subprocess.run(command, cwd=ROOT)
                if r.returncode != 0:
                    failed = True
                    print(f"config {mode}/{cfg}/{order} 失败 (exit {r.returncode})", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
