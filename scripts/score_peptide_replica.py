#!/usr/bin/env python3
"""Per-frame MM/PBSA of one packet replica with the frozen JAXPBSA analyzer (P4 scoring entry).

    python scripts/score_peptide_replica.py --run-id PMI_v21m1_rep01 --dry-run
    python scripts/score_peptide_replica.py --run-id PMI_v21m1_rep01                    # window 1-10 ns, every frame
    python scripts/score_peptide_replica.py --run-id PMI_v21m1_rep01 --stride 5

Scoring plan: docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §4 (2026-10-15 proposal, user
approved).  Reads <replica>/analysis/solute_dry.dcd (strip_peptide_replica.py: solute
only, molecules made whole, Å) and the prepared solute (parameters.npz, system_vacuum.xml,
complex_dry.cif).  Scores with `jaxpbsa.online.OnlineMMPBSA` exactly as constructed by the
online path -- every method/default parameter left at the frozen main 3737870 values
(h 0.5 / ligand box h 0.25, PBParams(precond="auto"), mbondi2, INP=1 gamma/beta, SA k/points);
the grid is sized from the scored frames themselves (pilot = the window).  Each frame gets
the reporter's own validity rule (PBSAReporter._violation: margin vs margin_flag 0.1/kappa,
PB convergence, ligand box, SA); flagged frames are written with the reason, never dropped.

Writes <packet>/scores/<sample>/<model>/<repNN>/:
    frames.csv      one row per scored frame: frame, time_ps, all analyzer fields, flag, wall_ms
    summary.json    inputs, window/stride, sizing, warm-up time, ms/frame, device, counts,
                    mean/SD of delta_g_mmpbsa over valid frames, 3-ns block means from the
                    window start (1-4, 4-7, 7-10 ns; a shorter last block is labelled by its
                    own range), autocorrelation of delta_g_mmpbsa (diagnostic)
Window [start, end): frame k is labelled time_ps = 10 k (frame k is written at the end of
production step 5000 (k+1), i.e. 10 ps later); the end is exclusive so that 1-10 and 10-20 ns
share no frame.  Resumable: frames already in frames.csv are skipped (same window/stride required).
No acceptance threshold is applied here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
FRAME_PS = 10.0                       # strip writes every saved frame; run_peptide_md --dcd-ps 10
BLOCK_NS = 3.0                        # protocol §2.1
FIELDS = ["e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand", "delta_g_pb",
          "sasa_complex", "sasa_receptor", "sasa_ligand", "delta_g_sa", "delta_g_mmpbsa",
          "solver_iters", "converged", "margin_A", "margin_lig_A", "sa_ok"]


def run_row(packet: Path, run_id: str) -> dict:
    for r in csv.DictReader((packet / "proposed_runs.csv").open()):
        if r["run_id"] == run_id:
            return r
    raise SystemExit(f"run_id {run_id} not in proposed_runs.csv")


def blocks_ns(start, end):
    """Consecutive BLOCK_NS blocks from the window start; the last one may be shorter."""
    out, b0 = [], start
    while b0 < end - 1e-9:
        out.append((b0, min(b0 + BLOCK_NS, end)))
        b0 += BLOCK_NS
    return out


def autocorr(x, max_lag):
    """Normalised autocorrelation of x for lags 0..max_lag (diagnostic only)."""
    import numpy as np
    x = np.asarray(x, float) - np.mean(x)
    v = float(x @ x) / len(x)
    if v == 0:
        return [1.0]
    return [float((x[: len(x) - k] @ x[k:]) / (len(x) - k) / v) for k in range(min(max_lag, len(x) - 1) + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=PACKET)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--start-ns", type=float, default=1.0, help="window start (production time)")
    ap.add_argument("--end-ns", type=float, default=10.0, help="window end (production time)")
    ap.add_argument("--stride", type=int, default=1, help="score every n-th saved frame (10 ps each)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <packet>/scores/<sample>/<model>/<rep>)")
    ap.add_argument("--dry-run", action="store_true", help="print inputs/outputs; no JAX/OpenMM import")
    a = ap.parse_args()

    row = run_row(a.packet, a.run_id)
    rep = Path(row["output_directory"])
    sample, model = rep.parent.parent.name, rep.parent.name
    prep = a.packet / "prepared" / sample / model
    dcd, dpdb = rep / "analysis" / "solute_dry.dcd", rep / "analysis" / "solute_dry.pdb"
    out = a.out or a.packet / "scores" / sample / model / rep.name
    status = json.loads((rep / "status.json").read_text()) if (rep / "status.json").exists() else {}
    plan = dict(run_id=a.run_id, sample=sample, model=model, replica=rep.name, prepared=str(prep),
                dcd=str(dcd), dcd_exists=dcd.exists(), replica_complete=status.get("complete"),
                window_ns=[a.start_ns, a.end_ns], stride_frames=a.stride, frame_ps=FRAME_PS, out=str(out))
    if a.dry_run:
        print(json.dumps(plan)); return
    if not dcd.exists() or not status.get("complete"):
        raise SystemExit(f"{a.run_id}: replica not complete or solute_dry.dcd missing ({rep})")

    import numpy as np
    import mdtraj as md
    import openmm as mm
    from openmm import app
    sys.path.insert(0, str(ROOT))
    import jaxpbsa
    from jaxpbsa.online import OnlineMMPBSA, PBSAReporter

    ref = np.load(prep / "parameters.npz")
    n = len(ref["atom_name"])
    cif = app.PDBxFile(str(prep / "complex_dry.cif"))
    if [x.name for x in cif.topology.atoms()] != list(ref["atom_name"]):
        raise RuntimeError("complex_dry.cif atom order differs from parameters.npz")
    system = mm.XmlSerializer.deserialize((prep / "system_vacuum.xml").read_text())
    if system.getNumParticles() != n:
        raise RuntimeError(f"system_vacuum.xml has {system.getNumParticles()} particles, prepared solute {n}")
    traj = md.load_dcd(str(dcd), top=str(dpdb))
    if [x.name for x in traj.topology.atoms] != list(ref["atom_name"]):
        raise RuntimeError("solute_dry atom order differs from parameters.npz")
    xyz = traj.xyz.astype(np.float64)            # mdtraj reads the DCD in nm
    xyz *= 10.0
    t_ps = np.arange(traj.n_frames) * FRAME_PS
    sel = [k for k in range(traj.n_frames)
           if a.start_ns * 1000 - 1e-6 <= t_ps[k] < a.end_ns * 1000 - 1e-6 and k % a.stride == 0]
    if not sel:
        raise SystemExit("no frames in the window")

    out.mkdir(parents=True, exist_ok=True)
    fcsv = out / "frames.csv"
    done = set()
    if fcsv.exists():
        done = {int(r["frame"]) for r in csv.DictReader(fcsv.open())}
        prev = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else None
        if prev and (prev["window_ns"] != [a.start_ns, a.end_ns] or prev["stride_frames"] != a.stride):
            raise SystemExit(f"{out} was scored with another window/stride; use a new output or remove it")

    jaxpbsa.enable_compilation_cache()
    t0 = time.perf_counter()
    az = OnlineMMPBSA(system, cif.topology, np.arange(n), ref["ligand_idx"], xyz[sel[0]],
                      pilot_coords_A=xyz[sel])
    t_init = time.perf_counter() - t0
    judge = PBSAReporter(az, interval_steps=1, solute_idx=np.arange(n))   # only for its frame rule
    import jax
    dev = str(jax.devices()[0])

    new = not fcsv.exists()
    walls = []
    with fcsv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "time_ps"] + FIELDS + ["flag", "wall_ms"])
        if new:
            w.writeheader()
        for k in sel:
            if k in done:
                continue
            t1 = time.perf_counter()
            r = az(xyz[k])
            ms = (time.perf_counter() - t1) * 1000
            walls.append(ms)
            w.writerow(dict(frame=k, time_ps=t_ps[k], **{f: (json.dumps(r[f]) if f == "solver_iters" else r[f])
                                                          for f in FIELDS},
                            flag=judge._violation(r) or "", wall_ms=round(ms, 2)))
            fh.flush()

    rows = list(csv.DictReader(fcsv.open()))
    rows.sort(key=lambda x: int(x["frame"]))
    valid = [x for x in rows if not x["flag"]]
    g = np.array([float(x["delta_g_mmpbsa"]) for x in valid])
    tv = np.array([float(x["time_ps"]) for x in valid])
    blocks = {f"{b0:g}-{b1:g}ns": (float(g[(tv >= b0 * 1000) & (tv < b1 * 1000)].mean())
                                    if ((tv >= b0 * 1000) & (tv < b1 * 1000)).any() else None)
              for b0, b1 in blocks_ns(a.start_ns, a.end_ns)}
    acf = autocorr(g, min(len(g) // 2, 450)) if len(g) > 10 else []
    lag_e = next((k for k, v in enumerate(acf) if v < np.exp(-1)), None)
    # integrated autocorrelation time, Sokal automatic window (M >= 5 tau): the 1/e lag only
    # sees the fast part (PMI_v21m1_rep01: 10 ps vs tau 173 ps, slow tail ~0 only by ~1 ns)
    tau, win = None, None
    for M in range(1, len(acf)):
        tau, win = 1.0 + 2.0 * sum(acf[1:M + 1]), M
        if M >= 5 * tau:
            break
    summary = dict(
        **plan, n_frames_scored=len(rows), n_valid=len(valid), n_flagged=len(rows) - len(valid),
        flags=sorted({x["flag"] for x in rows if x["flag"]})[:10],
        delta_g_mmpbsa_mean=float(g.mean()) if len(g) else None,
        delta_g_mmpbsa_sd=float(g.std(ddof=1)) if len(g) > 1 else None,
        components_mean={f: float(np.mean([float(x[f]) for x in valid])) for f in
                         ("e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa")} if valid else None,
        block_means=blocks,
        autocorr_lag_ps_below_1_over_e=(lag_e * a.stride * FRAME_PS if lag_e is not None else None),
        autocorr_first_lags=[round(v, 3) for v in acf[:21]],
        autocorr_integrated_tau_ps=(round(tau * a.stride * FRAME_PS, 1) if tau else None),
        autocorr_sokal_window_ps=(win * a.stride * FRAME_PS if win else None),
        n_eff_estimate=(round(len(g) / tau, 1) if tau else None),
        autocorr_note="diagnostic of delta_g_mmpbsa over valid frames at the scored stride; sets the "
                      "frame-spacing rule (design §4.1, 2026-10-15), not a statistical weight",
        sizing={k: (list(v) if isinstance(v, tuple) else v) for k, v in az.sizing.items()},
        analyzer_defaults="OnlineMMPBSA defaults (frozen main 3737870); only pilot_coords_A given",
        init_incl_warmup_s=round(t_init, 2), warmup_s=round(az.warmup_time_s, 2),
        ms_per_frame_this_call=dict(median=round(float(np.median(walls)), 2), n=len(walls)) if walls else None,
        device=dev, host=platform.node(), jax=jax.__version__, openmm=mm.__version__,
        units="kcal/mol (analyzer output); Å for margins and areas",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("run_id", "n_frames_scored", "n_valid", "n_flagged",
                                              "delta_g_mmpbsa_mean", "block_means",
                                              "autocorr_integrated_tau_ps", "ms_per_frame_this_call",
                                              "warmup_s", "device")}, default=float), flush=True)


if __name__ == "__main__":
    main()
