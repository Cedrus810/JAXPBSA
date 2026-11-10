#!/usr/bin/env python3
"""Check the upstream BRD4 ligand sdf/mol2 against the prmtop ligand actually simulated; read-only.

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-10-17_small_molecule_benchmark/scripts/check_brd4_ligand_files.py

Plan docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §4.1 (descriptor bond orders must come from the
fixed upstream files, not from PDB distances).  Files: sources/brd4/{sdf,mol2}/ligand-<i>.sdf/.mol2
(commit 5233e0a, sources/retrieval_brd4_ligands.json).  Per sample: atom count, element order,
bond graph (sdf vs prmtop), atom names and partial charges (mol2 vs prmtop), internal geometry
(max |pair-distance difference|; the crds frame is translated), formal charge from the sdf.
Writes reports/baseline_20261101/brd4_ligand_file_check.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import parmed
from rdkit import Chem

P = Path(__file__).resolve().parents[1]


def main():
    rows = {r["sample"]: r for r in csv.DictReader((P / "manifests/brd4_inputs.csv").open())}
    out = []
    for i in range(1, 11):
        s = f"BRD4-{i}"
        r = rows[s]
        a0, a1 = int(r["ligand_atom_start_0based"]), int(r["ligand_atom_end_exclusive"])
        st = parmed.load_file(r["prmtop"], r["coordinates"])
        lat = st.atoms[a0:a1]
        px = np.array([[a.xx, a.xy, a.xz] for a in lat])
        pq = np.array([a.charge for a in lat])
        pb = {tuple(sorted((b.atom1.idx - a0, b.atom2.idx - a0))) for b in st.bonds
              if a0 <= b.atom1.idx < a1 and a0 <= b.atom2.idx < a1}
        m = Chem.MolFromMolFile(str(P / f"sources/brd4/sdf/ligand-{i}.sdf"), removeHs=False, sanitize=True)
        sx = m.GetConformer().GetPositions()
        sb = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in m.GetBonds()}
        block = (P / f"sources/brd4/mol2/ligand-{i}.mol2").read_text().split("@<TRIPOS>ATOM")[1]
        m2 = [x.split() for x in block.split("@<TRIPOS>")[0].strip().splitlines()]
        same_n = len(lat) == m.GetNumAtoms() == len(m2)
        dist = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)
        out.append(dict(
            sample=s, n_prmtop=len(lat), n_sdf=m.GetNumAtoms(), n_mol2=len(m2),
            element_order_equal=same_n and [a.element_name.capitalize() for a in lat]
                                == [a.GetSymbol().capitalize() for a in m.GetAtoms()],
            bond_graph_equal=pb == sb,
            mol2_names_equal_prmtop=same_n and [a.name for a in lat] == [x[1] for x in m2],
            max_abs_charge_diff_mol2_prmtop=f"{np.abs(np.array([float(x[8]) for x in m2]) - pq).max():.2e}"
                                            if same_n else "",
            max_pair_distance_diff_A=f"{np.abs(dist(sx) - dist(px)).max():.4f}" if same_n else "",
            sdf_formal_charge=Chem.GetFormalCharge(m), n_heavy=m.GetNumHeavyAtoms(),
            smiles=Chem.MolToSmiles(Chem.RemoveHs(m)),
        ))
    dst = P / "reports/baseline_20261101/brd4_ligand_file_check.csv"
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for x in out:
        print(x)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
