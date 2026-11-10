#!/usr/bin/env python3
"""Rebuild BRD4-5 with the measured ligand (alprazolam) in place of the upstream 5,6-dihydro form.

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-10-17_small_molecule_benchmark/scripts/prepare_brd4_5_alprazolam.py \
        --out experiments/2026-10-17_small_molecule_benchmark/brd4_5_alprazolam_20261101

Plan docs/BRD4_5_REPREP_PLAN_20261101.md §3-§4.  Only the ligand changes: the upstream ligand
(sources/brd4/sdf/ligand-5.sdf for bond orders, BRD4-5.crds for coordinates, mol2 for atom names)
loses H8 (on NAL) and H9 (on CAO) and NAL-CAO becomes a double bond; the result must have
alprazolam's InChIKey.  antechamber AM1-BCC / GAFF (as upstream) + parmchk2, ligand prmtop by
tleap, then ParmEd splices protein (residues 1-121) + new ligand + upstream water/ions/box in the
original order.  Every non-ligand atom's charge, LJ, bonded terms and coordinates are compared with
the upstream prmtop and recorded (no threshold; the InChIKey check is the only stop).  Writes <out>/prepared/brd4/BRD4-5/alprazolam_gaff_fix/{complex_solv.prmtop,
complex_solv.rst7, prep_record.json, ...} and <out>/manifests/{brd4_inputs.csv,runs.csv}.
Resumable: a finished prep_record.json is reported and nothing is redone.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
MODEL_ID = "alprazolam_gaff_fix"
SAMPLE, RESNAME = "BRD4-5", "08H"
ALPRAZOLAM_INCHIKEY = "VREFGVBLTWBCJP-UHFFFAOYSA-N"   # PubChem CID 2118; CCD 08H formula C17H13ClN4
DROP_H = ("H8", "H9")          # upstream H on NAL and on CAO (mol2 atoms 31, 32)
DOUBLE = ("NAL", "CAO")
ENV_BIN = Path(sys.executable).parent


def run(cmd, cwd: Path, log: Path):
    env = dict(os.environ, AMBERHOME=str(ENV_BIN.parent), OMP_NUM_THREADS="1",
               PATH=str(ENV_BIN) + os.pathsep + os.environ.get("PATH", ""))
    t = time.monotonic()
    with log.open("w") as fh:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd} -> rc {p.returncode}; see {log}")
    return round(time.monotonic() - t, 2)


def mol2_names(path: Path):
    block = path.read_text().split("@<TRIPOS>ATOM")[1].split("@<TRIPOS>")[0]
    return [x.split()[1] for x in block.strip().splitlines()]


def build_ligand(up, lig0, lig1, work: Path):
    """RDKit molecule of alprazolam on the upstream pose; returns (mol, kept upstream names)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    names = mol2_names(PACKAGE / "sources/brd4/mol2/ligand-5.mol2")
    if names != [a.name for a in up.atoms[lig0:lig1]]:
        raise RuntimeError("upstream mol2 names differ from the prmtop ligand")
    m = Chem.RWMol(Chem.MolFromMolFile(str(PACKAGE / "sources/brd4/sdf/ligand-5.sdf"), removeHs=False, sanitize=False))
    conf = m.GetConformer()
    for k, a in enumerate(up.atoms[lig0:lig1]):          # upstream crds frame (sdf is a translated copy)
        conf.SetAtomPosition(k, Point3D(a.xx, a.xy, a.xz))
    i, j = names.index(DOUBLE[0]), names.index(DOUBLE[1])
    drop = sorted((names.index(x) for x in DROP_H), reverse=True)
    for h in drop:
        nbr = [n.GetIdx() for n in m.GetAtomWithIdx(h).GetNeighbors()]
        if m.GetAtomWithIdx(h).GetSymbol() != "H" or len(nbr) != 1 or nbr[0] not in (i, j):
            raise RuntimeError(f"atom {names[h]} is not an H on {DOUBLE}")
    m.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)
    for h in drop:
        m.RemoveAtom(h)
    kept = [n for k, n in enumerate(names) if k not in drop]
    mol = m.GetMol()
    for a in mol.GetAtoms():
        a.SetNoImplicit(True)
    Chem.SanitizeMol(mol)
    key = Chem.MolToInchiKey(mol)
    if key != ALPRAZOLAM_INCHIKEY:
        raise RuntimeError(f"edited ligand InChIKey {key} != alprazolam {ALPRAZOLAM_INCHIKEY}; nothing written")
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    (work / "ligand_in.sdf").write_text(Chem.MolToMolBlock(mol))
    return mol, kept, dict(inchikey=key, formula=CalcMolFormula(mol), removed=list(DROP_H),
                           double_bond=list(DOUBLE), formal_charge=Chem.GetFormalCharge(mol))


def parametrize(mol, kept, work: Path):
    t_ac = run(["antechamber", "-i", "ligand_in.sdf", "-fi", "mdl", "-o", "ligand_gaff.mol2", "-fo", "mol2",
                "-c", "bcc", "-nc", "0", "-at", "gaff", "-rn", RESNAME, "-pf", "y"], work, work / "antechamber.log")
    t_pc = run(["parmchk2", "-i", "ligand_gaff.mol2", "-f", "mol2", "-o", "ligand.frcmod", "-s", "gaff"],
               work, work / "parmchk2.log")
    lines = (work / "ligand_gaff.mol2").read_text().splitlines()
    i0 = lines.index("@<TRIPOS>ATOM") + 1
    i1 = next(k for k in range(i0, len(lines)) if lines[k].startswith("@<TRIPOS>"))
    if i1 - i0 != len(kept):
        raise RuntimeError(f"antechamber mol2 has {i1 - i0} atoms, expected {len(kept)}")
    xyz = mol.GetConformer().GetPositions()
    shift, q = [], []
    for k in range(i0, i1):            # upstream names and the exact input pose back in
        f = lines[k].split()
        shift.append(float(np.linalg.norm(np.array([float(v) for v in f[2:5]]) - xyz[k - i0])))
        q.append(float(f[8]))
        f[1] = kept[k - i0]
        f[2:5] = [f"{v:.4f}" for v in xyz[k - i0]]
        lines[k] = "%7s %-8s %10s %10s %10s %-6s %4s %-8s %10s" % tuple(f[:9])
    b0 = lines.index("@<TRIPOS>BOND") + 1
    bonds = {}
    for k in range(b0, len(lines)):
        if lines[k].startswith("@<TRIPOS>"):
            break
        f = lines[k].split()
        bonds[tuple(sorted((kept[int(f[1]) - 1], kept[int(f[2]) - 1])))] = f[3]
    if bonds.get(tuple(sorted(DOUBLE))) != "2":
        raise RuntimeError(f"antechamber bond {DOUBLE} type {bonds.get(tuple(sorted(DOUBLE)))}, expected 2")
    (work / "ligand.mol2").write_text("\n".join(lines) + "\n")
    version = sorted(x.name.rsplit(".json", 1)[0] for x in (ENV_BIN.parent / "conda-meta").glob("ambertools-*.json"))
    (work / "tleap_ligand.in").write_text("\n".join([
        "source leaprc.gaff", "loadamberparams ligand.frcmod", "LIG = loadmol2 ligand.mol2", "check LIG",
        "saveamberparm LIG ligand.prmtop ligand.rst7", "quit"]) + "\n")
    t_tl = run(["tleap", "-f", "tleap_ligand.in"], work, work / "tleap_ligand.log")
    log = (work / "tleap_ligand.log").read_text()
    if "FATAL" in log or "Errors = 0" not in log or not (work / "ligand.prmtop").exists():
        raise RuntimeError("tleap ligand reported errors; see tleap_ligand.log")
    return dict(am1bcc_charge_sum=round(sum(q), 6), mol2_vs_input_max_shift_A=round(max(shift), 4),
                parmchk2_attn_lines=(work / "ligand.frcmod").read_text().count("ATTN"),
                ambertools_conda_package=version, wall_s=dict(antechamber=t_ac, parmchk2=t_pc, tleap=t_tl))


def terms(st, offset_map):
    """Per-atom and bonded parameters keyed by upstream-index tuples (offset_map: new idx -> key)."""
    out = {}
    for name, lst, n in (("bond", st.bonds, 2), ("angle", st.angles, 3), ("dihedral", st.dihedrals, 4)):
        for t in lst:
            at = [t.atom1, t.atom2] + ([t.atom3] if n >= 3 else []) + ([t.atom4] if n == 4 else [])
            if any(a.idx not in offset_map for a in at):
                continue
            key = tuple(offset_map[a.idx] for a in at)
            key = min(key, key[::-1])
            ty = t.type
            val = ((ty.k, ty.req) if name == "bond" else (ty.k, ty.theteq) if name == "angle"
                   else (ty.phi_k, ty.per, ty.phase, ty.scee, ty.scnb, getattr(t, "improper", False), getattr(t, "ignore_end", False)))
            out.setdefault((name, key), []).append(tuple(round(float(v), 6) for v in val))
    return {k: sorted(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="independent package directory")
    a = ap.parse_args()
    out = a.out.resolve()
    work = out / "prepared/brd4" / SAMPLE / MODEL_ID
    rec_path = work / "prep_record.json"
    if rec_path.exists():
        print(json.dumps({"skip_prep_done": str(rec_path)})); return
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    import parmed
    up_row = next(r for r in csv.DictReader((PACKAGE / "manifests/brd4_inputs.csv").open()) if r["sample"] == SAMPLE)
    lig0, lig1 = int(up_row["ligand_atom_start_0based"]), int(up_row["ligand_atom_end_exclusive"])
    up = parmed.load_file(up_row["prmtop"], up_row["coordinates"])
    if up.residues[121].name != up_row["ligand_residue"] or up.residues[121].atoms[0].idx != lig0:
        raise RuntimeError("upstream ligand residue/index differs from manifests/brd4_inputs.csv")

    mol, kept, ident = build_ligand(up, lig0, lig1, work)
    lig_rec = parametrize(mol, kept, work)
    lig = parmed.load_file(str(work / "ligand.prmtop"), str(work / "ligand.rst7"))
    if [x.name for x in lig.atoms] != kept or lig.residues[0].name != RESNAME:
        raise RuntimeError("ligand prmtop atom names/residue differ from the edited ligand")

    n_up = len(up.atoms)
    # plain Structures: copying a sliced AmberParm re-validates its stale LJ tables (ParmEd 4.3.1)
    base = up.copy(parmed.Structure)
    prot = base[f"@1-{lig0}"]
    solv = base[f"@{lig1 + 1}-{n_up}"]
    new = prot + lig.copy(parmed.Structure) + solv
    new.box = up.box
    new.coordinates = np.concatenate([up.coordinates[:lig0], lig.coordinates, up.coordinates[lig1:]])
    new = parmed.amber.AmberParm.from_structure(new)
    new.save(str(work / "complex_solv.prmtop"), overwrite=True)
    new.save(str(work / "complex_solv.rst7"), format="rst7", overwrite=True)

    chk = parmed.load_file(str(work / "complex_solv.prmtop"), str(work / "complex_solv.rst7"))
    n_lig = len(kept)
    new_to_up = {k: k for k in range(lig0)}
    new_to_up.update({lig0 + n_lig + k: lig1 + k for k in range(n_up - lig1)})
    diffs = []
    for k, u in new_to_up.items():
        x, y = chk.atoms[k], up.atoms[u]
        if (x.name, x.residue.name, round(x.charge, 6), x.type, round(x.rmin, 6), round(x.epsilon, 6)) != \
           (y.name, y.residue.name, round(y.charge, 6), y.type, round(y.rmin, 6), round(y.epsilon, 6)):
            diffs.append(k)
        if len(diffs) > 20:
            break
    up_map = {k: k for k in range(n_up) if not lig0 <= k < lig1}
    t_up, t_new = terms(up, up_map), terms(chk, new_to_up)
    bonded_equal = t_up == t_new
    coord_shift = float(np.abs(chk.coordinates[list(new_to_up)] - up.coordinates[list(new_to_up.values())]).max())
    q = np.array([x.charge for x in chk.atoms])
    n_sol = lig0 + n_lig
    rec = dict(
        sample=SAMPLE, model_id=MODEL_ID, plan="docs/BRD4_5_REPREP_PLAN_20261101.md", upstream=up_row,
        ligand=dict(**ident, **lig_rec, n_atoms=n_lig, atom_names=kept),
        complex=dict(n_atoms=len(chk.atoms), ligand_atoms_0based=[lig0, n_sol], ligand_residue_1based=122,
                     ligand_residue=chk.residues[121].name, solute_charge=round(float(q[:n_sol].sum()), 6),
                     total_charge=round(float(q.sum()), 6), box=[float(v) for v in chk.box]),
        upstream_comparison=dict(non_ligand_atoms=len(new_to_up), per_atom_mismatches=diffs,
                                 bonded_terms_equal=bonded_equal, bonded_term_keys=len(t_up),
                                 max_coordinate_difference_A=round(coord_shift, 6)),
        tools=dict(python=sys.executable, parmed=parmed.__version__, host=os.uname().nodename),
        wall_s=round(time.monotonic() - t0, 1))
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    row = dict(sample=SAMPLE, ligand_residue=RESNAME, ligand_residue_1based=122, ligand_atom_start_0based=lig0,
               ligand_atom_end_exclusive=n_sol, n_ligand_atoms=n_lig, n_solute_atoms=n_sol, n_system_atoms=len(chk.atoms),
               ligand_charge=float(q[lig0:n_sol].sum()), n_waters=up_row["n_waters"],
               prmtop=str(work / "complex_solv.prmtop"), coordinates=str(work / "complex_solv.rst7"),
               status="alprazolam_reprep_not_molecularly_validated")
    with (out / "manifests/brd4_inputs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    old = [r for r in csv.DictReader((PACKAGE / "manifests/runs.csv").open()) if r["sample_id"] == SAMPLE]
    with (out / "manifests/runs.csv").open("w", newline="") as fh:   # same seeds as the original BRD4-5 replicas
        w = csv.DictWriter(fh, fieldnames=list(old[0]))
        w.writeheader()
        for r in old:
            rep = f"rep{int(r['replica']):02d}"
            w.writerow(dict(r, run_id=f"{SAMPLE}_{MODEL_ID}_{rep}", model_id=MODEL_ID,
                            output_directory=f"md/brd4/{SAMPLE}/{MODEL_ID}/{rep}"))
    rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(dict(prep=str(rec_path), inchikey=ident["inchikey"], formula=ident["formula"],
                          am1bcc_charge_sum=lig_rec["am1bcc_charge_sum"], n_atoms=len(chk.atoms),
                          solute_charge=rec["complex"]["solute_charge"], bonded_terms_equal=bonded_equal,
                          per_atom_mismatches=len(diffs), max_coordinate_difference_A=round(coord_shift, 6),
                          attn=lig_rec["parmchk2_attn_lines"], wall_s=rec["wall_s"])))


if __name__ == "__main__":
    main()
