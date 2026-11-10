#!/usr/bin/env python3
"""R4 N4: baselines without MD for the peptide models (start score on GPU; the rest read-only).

    python experiments/2026-11-08_ppi_r4/scripts/r4_start_baselines.py --models PMI:v21m1 P12Q_L10A:v21m1
    python ... --all

Protocol: experiments/2026-11-08_ppi_r4/r4_protocol.md §2 (N4).  Per model:
  (1) start score: the solute of setup/minimized_state.xml (the state MD starts from), scored
      once with OnlineMMPBSA defaults, the grid sized from that structure alone
      (fluctuation_allowance = 0, as scripts/score_smallmol_start.py); not re-minimised
  (2) start interface descriptors: delta SASA (analyzer areas: complex - receptor - ligand) and
      the number of receptor-peptide heavy-atom pairs closer than 4.0 A (the contact definition
      of protocol §1.2)
  (3) MD delta SASA: per replica mean of sasa_complex - sasa_receptor - sasa_ligand over the
      existing scores_1_20 frames.csv (no new computation)
Writes <out>/start_baselines.csv (one row per model) and <out>/md_dsasa.csv (one row per
replica); record.json.  Resumable: models already in start_baselines.csv are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
R4 = Path(__file__).resolve().parents[1]
FIELDS = ["e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa", "sasa_complex", "sasa_receptor",
          "sasa_ligand", "converged", "margin_A", "margin_lig_A", "sa_ok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=PACKET)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--models", nargs="+", help="sample:model")
    g.add_argument("--all", action="store_true", help="8 samples x v21m1, v21m2")
    ap.add_argument("--out", type=Path, default=R4 / "n4_baselines")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.all:
        names = [r["peptide_name"] for r in csv.DictReader((a.packet / "sample_structure_audit.csv").open())]
        models = [(s, m) for s in names for m in ("v21m1", "v21m2")]
    else:
        models = [tuple(x.split(":")) for x in a.models]
    if a.dry_run:
        print(json.dumps(dict(models=models, out=str(a.out)))); return

    import numpy as np
    import openmm as mm
    from openmm import app
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments/2026-11-06_ppi_structure_review/scripts"))
    import jaxpbsa
    import stage_trace as T
    from jaxpbsa.online import OnlineMMPBSA, PBSAReporter
    import jax

    a.out.mkdir(parents=True, exist_ok=True)
    fcsv = a.out / "start_baselines.csv"
    done = {(r["sample"], r["model"]) for r in csv.DictReader(fcsv.open())} if fcsv.exists() else set()
    jaxpbsa.enable_compilation_cache()
    new = not fcsv.exists()
    walls = {}
    with fcsv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample", "model", "source"] + FIELDS +
                           ["delta_sasa", "contacts_4A_heavy", "peptide_net_charge", "flag", "init_s", "wall_ms"])
        if new:
            w.writeheader()
        for s, m in models:
            if (s, m) in done:
                continue
            prep = a.packet / "prepared" / s / m
            ref = np.load(prep / "parameters.npz")
            n = len(ref["atom_name"])
            cif = app.PDBxFile(str(prep / "complex_dry.cif"))
            system = mm.XmlSerializer.deserialize((prep / "system_vacuum.xml").read_text())
            pdb = T.read_pdb(prep / "complex_dry.pdb")
            src = a.packet / "md" / s / m / "setup/minimized_state.xml"
            st = T.image_peptide(T.read_state(src, pdb))[0]
            x = st.xyz
            t0 = time.perf_counter()
            az = OnlineMMPBSA(system, cif.topology, np.arange(n), ref["ligand_idx"], x, fluctuation_allowance=0.0)
            t_init = time.perf_counter() - t0
            judge = PBSAReporter(az, interval_steps=1, solute_idx=np.arange(n))
            t1 = time.perf_counter()
            r = az(x)
            ms = (time.perf_counter() - t1) * 1000
            heavy = np.array([not str(nm).startswith("H") for nm in ref["atom_name"]])
            A = np.zeros(n, bool); A[ref["receptor_idx"]] = True
            D = np.linalg.norm(x[heavy & ~A][:, None] - x[heavy & A][None], axis=2)
            w.writerow(dict(sample=s, model=m, source=str(src), **{f: r[f] for f in FIELDS},
                            delta_sasa=r["sasa_complex"] - r["sasa_receptor"] - r["sasa_ligand"],
                            contacts_4A_heavy=int((D < 4.0).sum()),
                            peptide_net_charge=round(float(ref["charge"][ref["ligand_idx"]].sum()), 4),
                            flag=judge._violation(r) or "", init_s=round(t_init, 2), wall_ms=round(ms, 2)))
            fh.flush()
            walls[f"{s}:{m}"] = dict(init_s=round(t_init, 2), ms=round(ms, 2))
            print(json.dumps({"model": f"{s}:{m}", "dG": r["delta_g_mmpbsa"], "init_s": round(t_init, 2)}), flush=True)
            del az, judge
    rows = []
    for s, m in models:
        for rp in ("rep01", "rep02", "rep03"):
            fr = list(csv.DictReader((a.packet / "scores_1_20" / s / m / rp / "frames.csv").open()))
            v = [float(x["sasa_complex"]) - float(x["sasa_receptor"]) - float(x["sasa_ligand"]) for x in fr if not x["flag"]]
            rows.append(dict(sample=s, model=m, replica=rp, n_frames=len(v), delta_sasa_mean=round(float(np.mean(v)), 3)))
    with (a.out / "md_dsasa.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rec = json.loads((a.out / "record.json").read_text()) if (a.out / "record.json").exists() else {}
    rec.setdefault("calls", []).append(dict(models=[f"{s}:{m}" for s, m in models], new=walls, host=platform.node(),
                                            device=str(jax.devices()[0]), jax=jax.__version__,
                                            note="start = minimized_state.xml solute; grid from the start alone"))
    (a.out / "record.json").write_text(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    main()
