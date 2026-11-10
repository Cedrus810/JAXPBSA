#!/usr/bin/env python3
"""P1.0 static inventory for the BRD4 / Hou baseline comparison; read-only, no molecular computation.

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-10-17_small_molecule_benchmark/scripts/build_baseline_inventory.py

Plan docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §4.1.  One row per candidate (BRD4-1..10, Hou a1..a7):
identity and source, R/L ranges, labels, the minimised start already written by `run_smallmol_md.py
setup`, descriptor (bond-order) sources, existing B2 scores, and per baseline what is missing.
Reads manifests/, md/*/setup/, prepared/hou/*/ph74_gaff2/, reports/ and scores/; writes
reports/baseline_20261101/inventory.csv.  Geometry columns are plain distances on
setup/solute_reference.npz (the minimised solute), not scores.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
OUT = P / "reports/baseline_20261101/inventory.csv"
BRD4_COMMIT = "5233e0a1529b3c3ba9adf8c245de2b658e5fbf40"
DEFAULT_GRID = "C/R h 0.5 A, L h 0.25 A, mbondi2, INP1 SA, PB tol 1e-5, float32"


def rows(path, key):
    return {x[key]: x for x in csv.DictReader((P / path).open())}


def start_geometry(target, sample, model):
    z = np.load(P / f"md/{target}/{sample}/{model}/setup/solute_reference.npz")
    x, el, lig = z["coords_A"], z["element"], z["ligand_local_idx"]
    rec = np.setdiff1d(np.arange(len(x)), lig)
    lh, rh = lig[el[lig] != "H"], rec[el[rec] != "H"]
    d = np.sqrt(((x[lh][:, None, :] - x[rh][None, :, :]) ** 2).sum(-1))
    ca = x[z["receptor_ca_idx"]]
    breaks = int((np.linalg.norm(np.diff(ca, axis=0), axis=1) > 4.2).sum())
    return dict(n_lig_heavy=len(lh), min_lr=float(d.min()), n_rec_heavy_4A=int((d < 4.0).any(0).sum()),
                ca_breaks=breaks)


def replica_se(path, sample, col="dG_mmpbsa_mean"):
    v = [float(x[col]) for x in csv.DictReader((P / path).open()) if x["sample"] == sample]
    return len(v), st.stdev(v) / len(v) ** 0.5


def main():
    brd4_in, brd4_lab = rows("manifests/brd4_inputs.csv", "sample"), rows("manifests/brd4_measurements.csv", "sample")
    hou_in, hou_lab = rows("manifests/hou_inputs.csv", "sample"), rows("manifests/hou2011_table_s1.csv", "sample_id")
    b_def, b_h04 = rows("reports/brd4_round1_samples.csv", "sample"), rows("reports/brd4_h0.4_samples.csv", "sample")
    h_def = rows("reports/hou_ranking_20261030/samples.csv", "sample")
    h_h04 = rows("reports/hou_h04_vs_default_20261026.csv", "sample")
    out = []

    for i in range(1, 11):
        s = f"BRD4-{i}"
        inp, lab = brd4_in[s], brd4_lab[s]
        setup = P / f"md/brd4/{s}/mobley_amber/setup"
        rec = json.loads((setup / "setup_record.json").read_text())
        g = start_geometry("brd4", s, "mobley_amber")
        n_def, se_def = replica_se("reports/brd4_round1_replicas.csv", s)
        n_h04, se_h04 = replica_se("reports/brd4_h0.4_replicas.csv", s)
        out.append(dict(
            sample=s, target="brd4", model_id="mobley_amber", ligand_resname=inp["ligand_residue"],
            identity_source=f"MobleyLab/benchmarksets BRD4 ligand ID {i} (upstream README table)",
            source_files=f"sources/brd4/amber/{s}.prmtop,.crds", source_version=f"commit {BRD4_COMMIT}",
            force_field="ff14SB / GAFF / AM1-BCC (upstream, unchanged)",
            receptor_atoms_0based=f"0-{int(inp['ligand_atom_start_0based']) - 1}",
            ligand_atoms_0based=f"{inp['ligand_atom_start_0based']}-{int(inp['ligand_atom_end_exclusive']) - 1}",
            n_lig_atoms=inp["n_ligand_atoms"], n_lig_heavy=g["n_lig_heavy"],
            formal_charge="0 (sources/brd4/sdf, no M CHG; matches upstream README)",
            partial_charge_sum=f"{float(inp['ligand_charge']):.6f}",
            exp_label=lab["label"], exp_dG=lab["experimental_dg_kcal_mol"], exp_sem=lab["sem_kcal_mol"],
            assay=lab["assay_type"], exp_source=lab["source_doi"], label_verification=lab["verification"],
            start_origin="docking (AutoDock Vina, upstream)" if i == 1 else "co-crystal pose (upstream)",
            start_structure=f"md/brd4/{s}/mobley_amber/setup/minimized_state.xml; solute_reference.npz",
            start_minimisation=f"unrestrained, tol 10 kJ/mol/nm; ligand heavy max shift "
                               f"{rec['minimisation']['ligand_heavy_shift_A']['max']:.2f} A",
            start_min_lig_rec_heavy_A=f"{g['min_lr']:.2f}", start_rec_heavy_within_4A=g["n_rec_heavy_4A"],
            start_ca_breaks=g["ca_breaks"],
            bond_order_source=f"sources/brd4/sdf/ligand-{i}.sdf (commit {BRD4_COMMIT[:7]}); atoms/order/bonds equal "
                              "prmtop (reports/baseline_20261101/brd4_ligand_file_check.csv)",
            B0_status="all inputs present; not computed",
            B1_status="scored: scores/start_default (score_smallmol_start.py, RESULTS §21.1)",
            B2_default=f"{float(b_def[s]['dG_mmpbsa_mean_of_replicas']):.3f} (n={n_def}, SE {se_def:.3f})",
            B2_default_design="3 x (0,5] ns, 500 frames/replica, offline, pilot = own window; "
                              "no protocol.json (pre-binding run)",
            B2_h04=f"{float(b_h04[s]['dG_mmpbsa_mean_of_replicas']):.3f} (n={n_h04}, SE {se_h04:.3f})",
            B2_h04_design="3 x (0,5] ns, same frames as default, pilot = own window",
            scoring_params=DEFAULT_GRID, amber_q3_reference="reports/reference_q3_samples.csv (9 frames)",
            gaps="none for B0/B1/B2 inputs",
            next_action="B0 from sdf; B1 in scores/start_default (RESULTS §21.1)",
        ))

    for i in range(1, 8):
        s = f"a{i}"
        inp, lab = hou_in[s], hou_lab[s]
        prep = P / f"prepared/hou/{s}/ph74_gaff2"
        pr = json.loads((prep / "prep_record.json").read_text())
        lp = pr["ligand_prep"]
        rec = json.loads((P / f"md/hou/{s}/ph74_gaff2/setup/setup_record.json").read_text())
        g = start_geometry("hou", s, "ph74_gaff2")
        stereo = "; stereo C2 corrected to CCD (C10 moved 2.47 A)" if lp.get("stereo_corrected_to_ccd") else ""
        out.append(dict(
            sample=s, target="hou", model_id="ph74_gaff2", ligand_resname=inp["ligand_residue"],
            identity_source=f"CCD {lp['ccd']} (PDB {lab['pdb_id']}); Table S1 row {s}",
            source_files=f"sources/hou2011/pdb/{lab['pdb_id']}; sources/hou2011/ccd/{lp['ccd']}.cif; "
                         f"prepared/hou/{s}/ph74_gaff2/",
            source_version="CCD retrieved 2026-10-23; prepare_hou_thrombin.py",
            force_field="ff14SB / GAFF2 / AM1-BCC (project re-prep, pH 7.4)",
            receptor_atoms_0based=f"0-{int(inp['ligand_atom_start_0based']) - 1}",
            ligand_atoms_0based=f"{inp['ligand_atom_start_0based']}-{int(inp['ligand_atom_end_exclusive']) - 1}",
            n_lig_atoms=inp["n_ligand_atoms"], n_lig_heavy=g["n_lig_heavy"],
            formal_charge=f"{lp['formal_charge']} (prep_record + ligand_in.sdf M CHG)",
            partial_charge_sum=f"{float(inp['ligand_charge']):.6f}",
            exp_label="quantitative (Table S1 dG)", exp_dG=lab["experimental_dg_kcal_mol"], exp_sem="",
            assay="not stated in Table S1", exp_source=f"{lab['source_doi']} {lab['source_location']}",
            label_verification="Table S1 transcription; underlying assay/conditions not rechecked",
            start_origin=("co-crystal pose + MODELLER loop B184-190" if i <= 4 else "co-crystal pose") + stereo,
            start_structure=f"md/hou/{s}/ph74_gaff2/setup/minimized_state.xml; solute_reference.npz",
            start_minimisation=f"unrestrained, tol 10 kJ/mol/nm; ligand heavy max shift "
                               f"{rec['minimisation']['ligand_heavy_shift_A']['max']:.2f} A",
            start_min_lig_rec_heavy_A=f"{g['min_lr']:.2f}", start_rec_heavy_within_4A=g["n_rec_heavy_4A"],
            start_ca_breaks=g["ca_breaks"],
            bond_order_source=f"prepared/hou/{s}/ph74_gaff2/ligand_in.sdf (CCD bond orders + pH 7.4 edits)",
            B0_status="all inputs present; not computed",
            B1_status="scored: scores/start_default (score_smallmol_start.py, RESULTS §21.1)",
            B2_default=f"{float(h_def[s]['score']):.3f} (n=6, SE {float(h_def[s]['replica_se']):.3f})",
            B2_default_design=f"6 x (0,10] ns, 1000 frames/replica, online (pilot = NPT 1 ns); "
                              f"offline-recovered replicas {h_def[s]['offline_recovered_replicas']} "
                              "(pilot = full window)",
            B2_h04=f"{float(h_h04[s]['h04']):.3f} (3 x (0,5] ns only)",
            B2_h04_design="3 x (0,5] ns; NOT the 6 x 10 ns budget; sensitivity only",
            scoring_params=DEFAULT_GRID, amber_q3_reference="reports/reference_q3_hou_samples.csv (9 frames)",
            gaps="none for B0/B1/B2 inputs",
            next_action="B0 from ligand_in.sdf; B1 in scores/start_default (RESULTS §21.1)",
        ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
