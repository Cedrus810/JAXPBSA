#!/usr/bin/env python3
"""Collect the P2 structure QC table (plan §4 P2 deliverable `structure_qc.csv`).

    cd /home/ruigengji/JAXPBSA
    $PY experiments/2026-10-11_public_data_execution_packet/scripts/collect_structure_qc.py

Reads only files the prep / setup / run / strip entries already wrote; no
simulation.  One row per replica in proposed_runs.csv whose status.json says
complete; sample- and setup-level columns repeat on each replica row.

Columns, by origin:
  prep_record.json      sequence, mutations, rebuilt residues, chirality repairs,
                        atom counts, net charges, His states, termini, closest
                        non-bonded heavy contact of rebuilt/mutated residues
                        (before any minimisation)
  setup_record.json     solvent/ion counts, box, minimisation energies, stereo
                        class changes, largest stereo dihedral shift and the term
                        it belongs to (recomputed here from complex_dry.pdb vs
                        minimized_state.xml)
  status.json           seeds, NVT/NPT wall time, production ns and s/ns
  diagnostics.csv       p50/p95 over all frames, and p50 over the first and second
                        half of production (time-block view; 1 frame = 10 ps here)
  diagnostics_summary   mean T and density from the chunk logs

Report only: no acceptance thresholds or pass/fail flags (plan §5.1–5.2).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                       # noqa: E402
import peptide_geometry as geo           # noqa: E402

METRICS = ("receptor_CA_rmsd_A", "peptide_CA_rmsd_receptor_frame_A", "peptide_CA12_shift_A",
           "min_rec_pep_heavy_A", "nearest_waterO_A_HOH1_A", "nearest_waterO_A_HOH185_A",
           "nearest_waterO_A_HOH186_A", "nearest_waterO_B_HOH74_A")
BLOCK_METRICS = METRICS[:4]


def state_positions_nm(path: Path) -> np.ndarray:
    pos = ET.parse(path).getroot().find("Positions")
    return np.array([[float(p.get(k)) for k in "xyz"] for p in pos])


def largest_setup_shift(prep_dir: Path, setup_dir: Path, n_sol: int) -> tuple[float, str]:
    res, xyz_prep = geo.residues_from_pdb(prep_dir / "complex_dry.pdb")
    terms = geo.stereo_terms(res)
    before = geo.measure(terms, xyz_prep)
    after = geo.measure(terms, state_positions_nm(setup_dir / "minimized_state.xml")[:n_sol] * 10.0)
    shift = [(abs((a["deg"] - b["deg"] + 180.0) % 360.0 - 180.0), b) for b, a in zip(before, after)]
    d, b = max(shift, key=lambda x: x[0])
    a = next(x for x in after if x["label"] == b["label"] and x["kind"] == b["kind"])
    return round(d, 1), f"{b['label']} {b['kind']} {b['deg']:.1f}->{a['deg']:.1f}"


def main():
    rows = []
    setup_cache = {}
    for run in csv.DictReader((PACKET / "proposed_runs.csv").open()):
        out = Path(run["output_directory"])
        st_path = out / "status.json"
        if not st_path.exists() or not json.loads(st_path.read_text()).get("complete"):
            continue
        st = json.loads(st_path.read_text())
        name, model = st["peptide_name"], st["model_id"]
        prep_dir = PACKET / "prepared" / name / model
        setup_dir = PACKET / "md" / name / model / "setup"
        prep = json.loads((prep_dir / "prep_record.json").read_text())
        setup = json.loads((setup_dir / "setup_record.json").read_text())
        mini = setup["minimisation"]
        if (name, model) not in setup_cache:
            setup_cache[(name, model)] = largest_setup_shift(prep_dir, setup_dir, prep["n_atoms"])
        shift_deg, shift_term = setup_cache[(name, model)]
        c0 = prep["closest_nonbonded_heavy_contacts_of_rebuilt_or_mutated"][0]

        diag = list(csv.DictReader((out / "analysis" / "diagnostics.csv").open()))
        summ = json.loads((out / "analysis" / "diagnostics_summary.json").read_text())
        val = {k: np.array([float(r[k]) for r in diag]) for k in METRICS}
        half = len(diag) // 2
        prod = [v for k, v in st["stage_wall_s"].items() if k.startswith("prod_chunk")]
        prod_ns = st["chunks_done"] * st["chunk_steps"] * st["dt_ps"] / 1000.0

        row = dict(
            run_id=st["run_id"], peptide_name=name, model_id=model, replica=st["replica"],
            sequence=prep["sequence"], mutations=";".join(prep["mutations"]),
            rebuilt=";".join(f"{r['chain']}:{r['built']}{r['resnum']}" for r in prep["rebuilt_residues"]),
            prep_seed=prep["prep_seed"],
            chirality_repair=";".join(f"{c['residue']} {c['chirality_before_deg']}->{c['chirality_after_deg']}"
                                      for c in prep.get("chirality_repair", [])) or "none",
            prep_stereo_violations=prep["stereo_check"]["violations"],
            n_solute_atoms=prep["n_atoms"], n_receptor_atoms=prep["n_receptor_atoms"],
            n_peptide_atoms=prep["n_peptide_atoms"],
            net_charge_complex=prep["net_charge_complex"], net_charge_receptor=prep["net_charge_receptor"],
            net_charge_peptide=prep["net_charge_peptide"],
            his_states=";".join(f"{k}:{'+'.join(v)}" for k, v in prep["his_states"].items()),
            termini=";".join(f"{c}:{t['n_residue']}-NH3+/{t['c_residue']}-COO-" for c, t in prep["termini"].items()
                             if t["has_OXT"] and len(t["n_H"]) == 3),
            prep_closest_contact_A=c0["d_A"], prep_closest_contact_pair=f"{c0['a']}--{c0['b']}",
            setup_n_atoms=setup["n_atoms"], setup_n_water=setup["added_residues"].get("HOH"),
            setup_n_na=setup["added_residues"].get("NA"), setup_n_cl=setup["added_residues"].get("CL"),
            setup_box_nm=round(setup["box_vectors_nm"][0][0], 4),
            setup_E_before_kJ_mol=f"{mini['E_before_kJ_mol']:.4g}",
            setup_E_after_kJ_mol=round(mini["E_after_kJ_mol"], 1),
            setup_stereo_class_changes=mini["stereo_class_changes"],
            setup_max_stereo_shift_deg=shift_deg, setup_max_stereo_shift_term=shift_term,
            seed_velocity=st["seeds"]["velocity"],
            nvt_wall_s=st["stage_wall_s"].get("nvt"), npt_wall_s=st["stage_wall_s"].get("npt"),
            prod_ns=round(prod_ns, 3), prod_s_per_ns_median=round(float(np.median(prod)), 1),
            prod_s_per_ns_max=round(float(max(prod)), 1),
            frames=len(diag), stereo_change_frames=sum(int(r["stereo_class_changes"]) > 0 for r in diag),
            T_mean_K=summ["log_means"].get("T_K"), density_mean=summ["log_means"].get("density"),
        )
        for k in METRICS:
            short = k.replace("_A", "").replace("nearest_waterO_", "w_")
            row[f"{short}_p50"] = round(float(np.percentile(val[k], 50)), 3)
            row[f"{short}_p95"] = round(float(np.percentile(val[k], 95)), 3)
        for k in BLOCK_METRICS:
            short = k.replace("_A", "")
            row[f"{short}_p50_first_half"] = round(float(np.median(val[k][:half])), 3)
            row[f"{short}_p50_second_half"] = round(float(np.median(val[k][half:])), 3)
        rows.append(row)

    if not rows:
        raise SystemExit("no complete replicas")
    path = PACKET / "structure_qc.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
