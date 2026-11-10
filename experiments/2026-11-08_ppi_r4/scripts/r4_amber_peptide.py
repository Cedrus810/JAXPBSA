#!/usr/bin/env python3
"""R4 N2/N3: same-frame Amber MM/PBSA and MM/GBSA for one peptide replica (CPU only).

    python experiments/2026-11-08_ppi_r4/scripts/r4_amber_peptide.py --run-id PMI_v21m1_rep01
    python ... --run-id PMI_v21m1_rep01 --frames 300 --out <smoke dir>

Protocol: experiments/2026-11-08_ppi_r4/r4_protocol.md §2 (N2, N3).  Follows
scripts/amber_reference_smallmol.py (MMPBSA.py with &gb igb=5 and &pb in one run, every value
written explicitly, per-frame -eo, the exact float32 frames we scored written as a frame-subset
DCD) with one difference: the peptide systems have no tleap prmtop, so the three prmtops are
built from the prepared system_vacuum.xml + complex_dry.cif with ParmEd, as
scripts/validate_mmpbsa.py does (constraint bonds get their length and k = 0; single-trajectory
MM/PBSA cancels bonded terms).  RADII = jaxpbsa assign_radii(mbondi2) (the radii our PB uses;
radiopt=0 makes pbsa read them, igb=5 is parameterised for mbondi2); SCREEN = tleap's per-element
defaults (H .85, C .72, N .79, O .85, S .96), which GB needs and the OpenMM system lacks.

Judgement as validate_mmpbsa.py: DELTA EEL / VDWAALS vs our e_coul_rl / e_lj_rl is a correctness
check; PB and SA are method comparisons; GB is a baseline.  No thresholds.  Writes
<out>/{complex,receptor,ligand}.prmtop, frames.dcd, mmpbsa.in, MMPBSA.py outputs, comparison.csv,
record.json.  Re-running with the same --out redoes it (removes the directory first).
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

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
R4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from amber_reference_smallmol import GB, amber_exe, comparison_rows, parse_eo   # noqa: E402

SCREEN = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "S": 0.96, "P": 0.86, "F": 0.88}
DEFAULT_FRAMES = "300,700,1100,1500,1900"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=PACKET)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--pbsa-h", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    row = next(r for r in csv.DictReader((a.packet / "proposed_runs.csv").open()) if r["run_id"] == a.run_id)
    rep = Path(row["output_directory"])
    sample, model = rep.parent.parent.name, rep.parent.name
    prep = a.packet / "prepared" / sample / model
    scores = a.packet / "scores_1_20" / sample / model / rep.name / "frames.csv"
    frames = [int(x) for x in a.frames.split(",") if x]
    out = (a.out or R4 / "n2n3_amber" / sample / model / rep.name).resolve()   # MMPBSA.py runs with cwd=out
    plan = dict(run_id=a.run_id, frames=frames, prepared=str(prep), scores=str(scores), out=str(out),
                pbsa_h=a.pbsa_h, gb=GB)
    if a.dry_run:
        print(json.dumps(plan)); return

    import numpy as np
    import mdtraj as md
    import openmm as mm
    import parmed
    from openmm import app
    sys.path.insert(0, str(ROOT))
    from jaxpbsa.openmm_io.radii import assign_radii
    from jaxpbsa.pb.energy import PBParams
    from jaxpbsa.sa import BETA_INP1, GAMMA_INP1

    t0 = time.monotonic()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ref = np.load(prep / "parameters.npz")
    n = len(ref["atom_name"])
    rec, lig = np.asarray(ref["receptor_idx"]), np.asarray(ref["ligand_idx"])
    if not (np.all(np.diff(rec) == 1) and np.all(np.diff(lig) == 1) and rec[-1] + 1 == lig[0] and lig[-1] == n - 1):
        raise RuntimeError("receptor/ligand are not two contiguous blocks (MMPBSA.py needs that)")
    cif = app.PDBxFile(str(prep / "complex_dry.cif"))
    system = mm.XmlSerializer.deserialize((prep / "system_vacuum.xml").read_text())

    # 1. prmtops from the OpenMM system (scripts/validate_mmpbsa.py build_parms), our radii, tleap SCREEN
    st = parmed.openmm.load_topology(cif.topology, system=system)
    con = {}
    for i in range(system.getNumConstraints()):
        a1, a2, dist = system.getConstraintParameters(i)
        con[frozenset((a1, a2))] = dist.value_in_unit(mm.unit.angstrom)
    zero = []
    for b in st.bonds:
        if b.type is None:
            bt = parmed.BondType(0.0, con.get(frozenset((b.atom1.idx, b.atom2.idx)), 1.0))
            zero.append(bt); b.type = bt
    st.bond_types.extend(zero); st.bond_types.claim()
    radii = np.asarray(assign_radii(cif.topology, "mbondi2"), float)
    if not np.allclose(radii, ref["radii_A"], atol=1e-6):
        raise RuntimeError("assign_radii(mbondi2) differs from parameters.npz radii_A")
    for atom, r in zip(st.atoms, radii):
        atom.solvent_radius = float(r)
        atom.screen = SCREEN.get(atom.element_name[0] if atom.element_name else "C", 0.8)
    prm, charges, natoms = {}, {}, {}
    for name, idx in (("complex", np.arange(n)), ("receptor", rec), ("ligand", lig)):
        sub = st if name == "complex" else st[[int(i) for i in idx]]
        amb = parmed.amber.AmberParm.from_structure(sub)
        amb.parm_data["RADIUS_SET"] = ["jaxpbsa assign_radii mbondi2 (r4_amber_peptide.py)"]
        p = out / f"{name}.prmtop"
        amb.write_parm(str(p))
        prm[name], natoms[name] = p, len(amb.atoms)
        charges[name] = round(float(np.sum(amb.parm_data["CHARGE"])), 4)

    # 2. the exact frames we scored (float32 Å), frame-subset DCD without box
    tr = md.load_dcd(str(rep / "analysis/solute_dry.dcd"), top=str(rep / "analysis/solute_dry.pdb"))
    if [x.name for x in tr.topology.atoms] != list(ref["atom_name"]):
        raise RuntimeError("solute_dry atom order differs from parameters.npz")
    with md.formats.DCDTrajectoryFile(str(out / "frames.dcd"), "w") as fh:
        fh.write(tr.xyz[frames] * 10.0)
    used = [dict(step=5000 * (k + 1), time_ps=10.0 * k) for k in frames]

    # 3. MMPBSA.py, all values explicit
    pb = PBParams()
    (out / "mmpbsa.in").write_text(f"""R4 same-frame reference for {a.run_id}; parameters pinned explicitly
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
    amberhome = os.environ.get("AMBERHOME") or str(Path(amber_exe("MMPBSA.py")).parent.parent)
    env = dict(os.environ, AMBERHOME=amberhome, OMP_NUM_THREADS="1",
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    t1 = time.monotonic()
    cmd = [sys.executable, amber_exe("MMPBSA.py"), "-O", "-i", "mmpbsa.in", "-o", "FINAL_RESULTS.dat",
           "-eo", "per_frame.csv", "-cp", str(prm["complex"]), "-rp", str(prm["receptor"]),
           "-lp", str(prm["ligand"]), "-y", "frames.dcd"]
    r = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True)
    (out / "mmpbsa_stdout.log").write_text(r.stdout + r.stderr)
    t_amber = time.monotonic() - t1
    if r.returncode != 0 or not (out / "per_frame.csv").exists():
        raise SystemExit(f"MMPBSA.py failed rc={r.returncode}; see {out / 'mmpbsa_stdout.log'}")

    # 4. per-frame comparison with our scores_1_20 rows of the same frames
    terms = parse_eo(out / "per_frame.csv")
    ours = {int(x["frame"]): x for x in csv.DictReader(scores.open())}
    gb_rows = [v for _, v in sorted(terms.get("GB", {}).items())]
    pb_rows = [v for _, v in sorted(terms.get("PB", {}).items())]
    if len(gb_rows) != len(frames) or len(pb_rows) != len(frames):
        raise RuntimeError(f"-eo has {len(gb_rows)} GB / {len(pb_rows)} PB delta rows for {len(frames)} frames")
    rows = comparison_rows(frames, used, ours, gb_rows, pb_rows)
    with (out / "comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    recd = dict(run_id=a.run_id, frames=frames, prepared=str(prep), atoms=natoms, prmtop_charge_e=charges,
                prmtop_source="ParmEd from system_vacuum.xml + complex_dry.cif", screen=SCREEN, gb=GB,
                pb=dict(istrng_M=pb.ionic_strength_M, indi=pb.eps_in, exdi=pb.eps_out, pbsa_h=a.pbsa_h,
                        prbrad=pb.probe_radius, inp=1, cavity_surften=GAMMA_INP1, cavity_offset=BETA_INP1,
                        fillratio=4.0),
                eo_methods_parsed={k: len(v) for k, v in terms.items()}, ours_scores=str(scores),
                ours_found=sum(1 for x in frames if x in ours), amber_wall_s=round(t_amber, 1),
                total_wall_s=round(time.monotonic() - t0, 1), host=platform.node(), amberhome=amberhome,
                note="EEL/VDW: correctness check; PB/SA: method comparison; GB: baseline. No thresholds applied.")
    (out / "record.json").write_text(json.dumps(recd, indent=2) + "\n")
    print(json.dumps({k: recd[k] for k in ("run_id", "eo_methods_parsed", "ours_found", "prmtop_charge_e",
                                           "amber_wall_s")}), flush=True)


if __name__ == "__main__":
    main()
