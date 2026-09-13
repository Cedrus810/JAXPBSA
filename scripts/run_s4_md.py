#!/usr/bin/env python
"""M0: solvated MD for S4 (Src SH2 + phosphotyrosyl peptide). Run on GPU:

    python scripts/run_s4_md.py --device CUDA --length-ns 10

Pipeline: solvate (TIP3P, 0.15 M NaCl, 1.2 nm padding) -> minimise -> NVT 100 ps
-> NPT 1 ns -> production N ns (DCD every 1 ps -> 10,000 frames for 10 ns,
matching the M9 benchmark matrix). Thermostat/barostat at 298.15 K to match the
PB temperature (DESIGN.md §1). Use --smoke for a CPU end-to-end sanity run.

Follow with scripts/strip_unwrap.py to produce the dry, unwrapped DCD that
JAXPBSA consumes.
"""
from __future__ import annotations

import argparse
import os
import time

import openmm as mm
from openmm import unit
import openmm.app as app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF_XML = ("amber/ff14SB.xml", "amber/phosaa14SB.xml", "amber/tip3p_standard.xml")
T = 298.15 * unit.kelvin
DT_PS = 0.002


def pick_platform(name: str) -> mm.Platform:
    if name != "auto":
        return mm.Platform.getPlatformByName(name)
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except Exception:
        return mm.Platform.getPlatformByName("CPU")


def run_stage(simulation: app.Simulation, n_steps: int, label: str, report_every: int):
    t0 = time.time()
    done = 0
    while done < n_steps:
        chunk = min(report_every, n_steps - done)
        simulation.step(chunk)
        done += chunk
        rate = done / (time.time() - t0)
        print(
            f"  [{label}] {done}/{n_steps} steps "
            f"({rate:.0f} steps/s, eta {(n_steps - done) / rate / 60:.1f} min)",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=os.path.join(ROOT, "data", "prepared", "S4_complex.pdb"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "md"))
    ap.add_argument("--device", default="auto", choices=["auto", "CUDA", "CPU", "OpenCL"])
    ap.add_argument("--length-ns", type=float, default=10.0)
    ap.add_argument("--nvt-ps", type=float, default=100.0)
    ap.add_argument("--npt-ps", type=float, default=1000.0)
    ap.add_argument("--dcd-interval-ps", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="CPU 快速端到端自检")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.smoke:  # 缩到分钟级: 溶剂化+短最小化+千步动力学
        args.nvt_ps, args.npt_ps, args.length_ns, args.dcd_interval_ps = 1.0, 1.0, 0.002, 0.1

    pdb = app.PDBFile(args.pdb)
    ff = app.ForceField(*FF_XML)
    print(f"[1/5] solvating {pdb.topology.getNumAtoms()} solute atoms "
          f"(TIP3P, 1.2 nm padding, 0.15 M NaCl)...", flush=True)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        ff, model="tip3p", padding=1.2 * unit.nanometer,
        ionicStrength=0.15 * unit.molar, neutralize=True,
    )
    solvated_pdb = os.path.join(args.out, "S4_solvated.pdb")
    with open(solvated_pdb, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)
    print(f"[1/5] solvated: {modeller.topology.getNumAtoms()} atoms -> {solvated_pdb}", flush=True)

    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds,
        rigidWater=True, removeCMMotion=True,
    )
    integrator = mm.LangevinMiddleIntegrator(T, 1.0 / unit.picosecond, DT_PS * unit.picoseconds)
    platform = pick_platform(args.device)
    props = {"CudaPrecision": "mixed"} if platform.getName() == "CUDA" else {}
    simulation = app.Simulation(modeller.topology, system, integrator, platform, props)
    simulation.context.setPositions(modeller.positions)

    print(f"[2/5] platform={platform.getName()}, minimising...", flush=True)
    simulation.minimizeEnergy(10.0 * unit.kilojoule_per_mole / unit.nanometer,
                              maxIterations=100 if args.smoke else 0)
    e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"      minimised, E = {e.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol", flush=True)

    print("[3/5] NVT...", flush=True)
    run_stage(simulation, int(args.nvt_ps / DT_PS), "NVT", 250 if args.smoke else 50_000)

    system.addForce(mm.MonteCarloBarostat(1 * unit.atmosphere, T))
    simulation.context.reinitialize(preserveState=True)
    print("[4/5] NPT...", flush=True)
    run_stage(simulation, int(args.npt_ps / DT_PS), "NPT", 250 if args.smoke else 50_000)

    dcd = app.DCDReporter(os.path.join(args.out, "S4_prod.dcd"), int(args.dcd_interval_ps / DT_PS))
    log_fh = open(os.path.join(args.out, "S4_prod.log"), "w")
    simulation.reporters = [
        dcd,
        app.StateDataReporter(
            log_fh, 50_000, step=True, time=True, potentialEnergy=True,
            temperature=True, density=True, speed=True,
        ),
        app.CheckpointReporter(os.path.join(args.out, "S4_prod.chk"), 250_000),
    ]
    n_prod = int(args.length_ns * 1000 / DT_PS)
    print(f"[5/5] production {args.length_ns} ns ({n_prod} steps, "
          f"DCD every {args.dcd_interval_ps} ps -> {n_prod * DT_PS / args.dcd_interval_ps / 1000 * 1000:.0f} frames)",
          flush=True)
    run_stage(simulation, n_prod, "prod", 250 if args.smoke else 50_000)
    simulation.reporters = []
    log_fh.close()

    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(os.path.join(args.out, "S4_prod_final.xml"), "w") as fh:
        fh.write(mm.XmlSerializer.serialize(state))
    print(f"done: {args.out}/S4_prod.dcd | .log | .chk | S4_prod_final.xml", flush=True)


if __name__ == "__main__":
    main()
