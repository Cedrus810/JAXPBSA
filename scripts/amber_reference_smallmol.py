#!/usr/bin/env python3
"""Same-frame Amber MM/PBSA and MM/GBSA for small-molecule replicas (CPU only).

    python scripts/amber_reference_smallmol.py --run-id BRD4-3_mobley_amber_rep01 --frames 124,249,374
    python scripts/amber_reference_smallmol.py --run-id ... --frames 124 --dry-run

Plan: docs/BRD4_HOU_EXECUTION_HANDOFF.md §4.6, docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md §8.
Follows scripts/validate_mmpbsa.py (what can / cannot be aligned with MMPBSA.py, explicit &pb
values, istrng in M, inp=1); differences:

  * topologies come from the replica's own upstream prmtop through ante-MMPBSA.py (strip
    solvent/ions and box, split at the ligand residue), not an OpenMM -> ParmEd round trip;
    so bonded terms and the GB SCREEN values are the real tleap ones;
  * RADII in all three prmtops are then overwritten with jaxpbsa.assign_radii(mbondi2), the
    radii our PB uses (Amber's own mbondi2 differs for some halogens); atoms whose radius
    changed are counted in the record.  radiopt=0 makes pbsa read them;
  * coordinates are the same float32 frames our scorer used (analysis/solute_dry.dcd),
    written to a frame-subset DCD (no box) for cpptraj -- no %8.3f mdcrd rounding;
  * one MMPBSA.py run does &gb (igb=5) and &pb; per-frame terms come from -eo.

Judgement follows validate_mmpbsa.py: DELTA EEL / VDWAALS vs our e_coul_rl / e_lj_rl is a
correctness check (same parameters, no R-L bonds); PB and SA are method comparisons (different
discretisation / surface / LCPO), GB is a common endpoint baseline.  Nothing is accepted or
rejected here; all numbers are written.

Writes <out>/: complex/receptor/ligand.prmtop, frames.dcd, mmpbsa.in, MMPBSA.py outputs
(FINAL_RESULTS.dat, per_frame.csv, _MMPBSA_* kept), record.json, comparison.csv (per frame:
ours from scores frames.csv vs Amber GB/PB terms).  Re-running with the same --out redoes it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/2026-10-17_small_molecule_benchmark"
GB = dict(igb=5, saltcon=0.150, surften=0.0072, surfoff=0.0)  # MMPBSA.py LCPO nonpolar, written explicitly


def amber_exe(name: str) -> str:
    cand = Path(sys.executable).parent / name
    return str(cand) if cand.exists() else (shutil.which(name) or name)


def rows_by(path: Path, key: str) -> dict:
    return {r[key]: r for r in csv.DictReader(path.open())}


def parse_eo(path: Path) -> dict:
    """MMPBSA.py -eo CSV -> {"GB"|"PB": {frame_no: {term: value}}} from the DELTA blocks
    (frame_no as printed; MMPBSA.py 14 numbers them from 0 -- callers match by order)."""
    out, method, in_delta, header = {}, None, False, None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        up = line.upper()
        if up.startswith("GENERALIZED BORN"):
            method, in_delta, header = "GB", False, None; continue
        if up.startswith("POISSON BOLTZMANN"):
            method, in_delta, header = "PB", False, None; continue
        if up.startswith("DELTA ENERGY TERMS") or up.startswith("DELTA ENERGY"):
            in_delta, header = True, None; continue
        if up.endswith("ENERGY TERMS"):
            in_delta = False; continue
        if not in_delta or method is None:
            continue
        if not line:
            in_delta = False; continue
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            header = cells; continue
        try:
            fno = int(float(cells[0]))
        except ValueError:
            continue
        out.setdefault(method, {})[fno] = {h: float(v) for h, v in zip(header[1:], cells[1:]) if v != ""}
    return out


def comparison_rows(frames, used, ours, gb_rows, pb_rows):
    """Pair energy rows and time metadata by the same zero-based ordinal."""
    if not (len(frames) == len(used) == len(gb_rows) == len(pb_rows)):
        raise ValueError("reference frame/metadata/energy counts differ")
    rows = []
    for j, k in enumerate(frames):
        o = ours.get(k, {})
        g, p = gb_rows[j], pb_rows[j]
        f = lambda d, key: d.get(key, "")
        rows.append(dict(frame=k, step=used[j]["step"], time_ps=used[j]["time_ps"],
                         ours_e_coul=o.get("e_coul_rl", ""), amber_EEL=f(p, "EEL"),
                         ours_e_lj=o.get("e_lj_rl", ""), amber_VDW=f(p, "VDWAALS"),
                         ours_dG_pb=o.get("delta_g_pb", ""), amber_EPB=f(p, "EPB"),
                         ours_dG_sa=o.get("delta_g_sa", ""), amber_ENPOLAR=f(p, "ENPOLAR"),
                         amber_EDISPER=f(p, "EDISPER"),
                         ours_dG_mmpbsa=o.get("delta_g_mmpbsa", ""), amber_PB_TOTAL=f(p, "DELTA TOTAL"),
                         amber_EGB=f(g, "EGB"), amber_ESURF=f(g, "ESURF"),
                         amber_GB_TOTAL=f(g, "DELTA TOTAL"), ours_flag=o.get("flag", "")))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--inputs", type=Path, default=PACKAGE / "manifests/brd4_inputs.csv")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--frames", required=True, help="comma-separated 0-based frame indices of solute_dry.dcd")
    ap.add_argument("--scores", type=Path, default=None,
                    help="our frames.csv dir (default <package>/scores/default/<target>/<sample>/<model>/<rep>)")
    ap.add_argument("--ours", choices=("scores", "online"), default="scores",
                    help="our per-frame values: offline --scores frames.csv, or the replica's committed online CSV (Hou)")
    ap.add_argument("--pbsa-h", type=float, default=0.5, help="pbsa grid spacing (scale = 1/h)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <package>/reference/amber_mmpbsa/<frame-set>/<target>/<sample>/<model>/<rep>")
    ap.add_argument("--frame-set", default="q3", help="name of the frame set (output path component)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    run = rows_by(a.package / "manifests/runs.csv", "run_id")[a.run_id]
    inp = rows_by(a.inputs, "sample")[run["sample_id"]]
    rep = a.package / run["output_directory"]
    setup = rep.parent / "setup"
    frames = [int(x) for x in a.frames.split(",")]
    scores = a.scores or a.package / "scores/default" / run["target"] / run["sample_id"] / run["model_id"] / rep.name
    out = a.out or (a.package / "reference/amber_mmpbsa" / a.frame_set / run["target"] / run["sample_id"]
                    / run["model_id"] / rep.name)
    plan = dict(run_id=a.run_id, prmtop=inp["prmtop"], frames=frames, dcd=str(rep / "analysis/solute_dry.dcd"),
                scores=str(scores / "frames.csv"), out=str(out), pbsa_h=a.pbsa_h, gb=GB,
                ligand_residue_mask=f":{inp['ligand_residue_1based']}")
    if a.dry_run:
        print(json.dumps(plan)); return

    import numpy as np
    import mdtraj as md
    import parmed
    from openmm import app
    sys.path.insert(0, str(ROOT))
    from jaxpbsa.openmm_io.radii import assign_radii
    from jaxpbsa.pb.energy import PBParams
    from jaxpbsa.sa import BETA_INP1, GAMMA_INP1

    t0 = time.monotonic()
    srec = json.loads((setup / "setup_record.json").read_text())
    n = srec["n_solute_atoms"]
    lig0, lig1 = srec["ligand_local_idx"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1. topologies: ante-MMPBSA.py on the upstream prmtop, then our radii
    prm = {k: out / f"{k}.prmtop" for k in ("complex", "receptor", "ligand")}
    cmd = [sys.executable, amber_exe("ante-MMPBSA.py"), "-p", inp["prmtop"], "-c", str(prm["complex"]),
           "-r", str(prm["receptor"]), "-l", str(prm["ligand"]), "-s", ":WAT,HOH,Na+,Cl-",
           "-n", plan["ligand_residue_mask"], "--radii=mbondi2"]
    # ante-MMPBSA.py / MMPBSA.py need AMBERHOME and find parmed/cpptraj/pbsa on PATH; when the env's
    # python is called directly (no `activate`, e.g. ssh to 183) neither is set
    amberhome = os.environ.get("AMBERHOME") or str(Path(amber_exe("MMPBSA.py")).parent.parent)
    env = dict(os.environ, AMBERHOME=amberhome, OMP_NUM_THREADS="1",
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True)
    (out / "ante_mmpbsa.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"ante-MMPBSA.py failed rc={r.returncode}; see {out / 'ante_mmpbsa.log'}")
    top = app.AmberPrmtopFile(inp["prmtop"]).topology
    radii = np.asarray(assign_radii(top, "mbondi2", atom_indices=np.arange(n)))[:n]
    changed, natoms = {}, {}
    for k, sl in (("complex", slice(0, n)), ("receptor", slice(0, lig0)), ("ligand", slice(lig0, lig1))):
        p = parmed.amber.AmberParm(str(prm[k]))
        want = radii[sl]
        natoms[k] = len(p.atoms)
        if len(p.atoms) != len(want):
            raise RuntimeError(f"{k}.prmtop has {len(p.atoms)} atoms, expected {len(want)}")
        old = np.array(p.parm_data["RADII"], float)
        changed[k] = int((np.abs(old - want) > 1e-6).sum())
        p.parm_data["RADII"] = [float(x) for x in want]
        p.parm_data["RADIUS_SET"] = ["jaxpbsa assign_radii mbondi2 (written by amber_reference_smallmol.py)"]
        p.write_parm(str(prm[k]))
        # order must line up with the trajectory we score; compare element + residue index per atom
        # (not names: OpenMM's prmtop reader renames N-terminal H1 -> H and HIE -> HIS)
        if k == "complex":
            ref_ids = [(x.element.atomic_number, x.residue.index) for x in list(top.atoms())[:n]]
            if [(x.atomic_number, x.residue.idx) for x in p.atoms] != ref_ids:
                raise RuntimeError("stripped complex atom order differs from the prmtop solute")
    charges = {k: float(np.sum(parmed.amber.AmberParm(str(prm[k])).parm_data["CHARGE"])) for k in prm}  # ParmEd: already e

    # 2. the exact frames we scored
    tr = md.load_dcd(str(rep / "analysis/solute_dry.dcd"), top=md.load_topology(inp["prmtop"]).subset(range(n)))
    times = list(csv.DictReader((rep / "analysis/frame_times.csv").open()))
    sub = tr[frames]
    with md.formats.DCDTrajectoryFile(str(out / "frames.dcd"), "w") as fh:
        fh.write(sub.xyz * 10.0)
    used = [times[k] for k in frames]

    # 3. MMPBSA.py, every &gb / &pb value written explicitly (validate_mmpbsa.py notes)
    pb = PBParams()
    (out / "mmpbsa.in").write_text(f"""same-frame reference for {a.run_id}; parameters pinned explicitly
&general
  startframe=1, verbose=2, keep_files=2,
/
&gb
  igb={GB['igb']}, saltcon={GB['saltcon']:.4f}, surften={GB['surften']}, surfoff={GB['surfoff']},
/
&pb
  istrng={pb.ionic_strength_M:.4f}, indi={pb.eps_in}, exdi={pb.eps_out},
  scale={1.0 / a.pbsa_h:.4f},
  radiopt=0,
  prbrad={pb.probe_radius},
  inp=1, cavity_surften={GAMMA_INP1}, cavity_offset={BETA_INP1},
  fillratio=4.0,
/
""")
    t1 = time.monotonic()
    # note: MMPBSA.py converts the trajectory with cpptraj to ASCII mdcrd (%8.3f) for sander/pbsa,
    # so Amber sees coordinates rounded to 0.001 A; EEL/VDW differences of ~1e-3 kcal/mol come from that
    cmd = [sys.executable, amber_exe("MMPBSA.py"), "-O", "-i", "mmpbsa.in", "-o", "FINAL_RESULTS.dat",
           "-eo", "per_frame.csv", "-cp", str(prm["complex"]), "-rp", str(prm["receptor"]),
           "-lp", str(prm["ligand"]), "-y", "frames.dcd"]
    r = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True)
    (out / "mmpbsa_stdout.log").write_text(r.stdout + r.stderr)
    t_amber = time.monotonic() - t1
    if r.returncode != 0 or not (out / "per_frame.csv").exists():
        raise SystemExit(f"MMPBSA.py failed rc={r.returncode}; see {out / 'mmpbsa_stdout.log'}")
    mdin = {}
    if (out / "_MMPBSA_pb.mdin").exists():
        for ln in (out / "_MMPBSA_pb.mdin").read_text().splitlines():
            if "=" in ln:
                for kv in ln.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1); mdin[k.strip()] = v.strip()

    # 4. per-frame comparison with our scores of the same frames
    terms = parse_eo(out / "per_frame.csv")
    if a.ours == "online":
        from smallmol_records import online_rows_by_frame
        ours, ours_src = online_rows_by_frame(rep)
    else:
        ours = {int(x["frame"]): x for x in csv.DictReader((scores / "frames.csv").open())} \
            if (scores / "frames.csv").exists() else {}
        ours_src = str(scores / "frames.csv")
    gb_rows = [v for _, v in sorted(terms.get("GB", {}).items())]
    pb_rows = [v for _, v in sorted(terms.get("PB", {}).items())]
    if len(gb_rows) != len(frames) or len(pb_rows) != len(frames):
        raise RuntimeError(f"-eo has {len(gb_rows)} GB / {len(pb_rows)} PB delta rows for {len(frames)} frames")
    rows = comparison_rows(frames, used, ours, gb_rows, pb_rows)
    with (out / "comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rec = dict(run_id=a.run_id, frames=used, prmtop_source=inp["prmtop"], atoms=natoms,
               radii_overwritten_vs_ante_mbondi2=changed, prmtop_charge_e=charges, gb=GB,
               pb=dict(istrng_M=pb.ionic_strength_M, indi=pb.eps_in, exdi=pb.eps_out, pbsa_h=a.pbsa_h, prbrad=pb.probe_radius,
                       inp=1, cavity_surften=GAMMA_INP1, cavity_offset=BETA_INP1, fillratio=4.0),
               pbsa_mdin_as_read=mdin, eo_methods_parsed={k: len(v) for k, v in terms.items()},
               ours_scores=ours_src, ours_found=sum(1 for x in frames if x in ours),
               amber_wall_s=round(t_amber, 1), total_wall_s=round(time.monotonic() - t0, 1),
               host=platform.node(), amberhome=amberhome,
               note="EEL/VDW: correctness check; PB/SA: method comparison; GB: baseline. No thresholds applied.")
    (out / "record.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("run_id", "eo_methods_parsed", "ours_found", "radii_overwritten_vs_ante_mbondi2",
                                          "amber_wall_s")}), flush=True)


if __name__ == "__main__":
    main()
