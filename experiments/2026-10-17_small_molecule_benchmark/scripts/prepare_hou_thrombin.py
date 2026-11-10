#!/usr/bin/env python3
"""Hou 2011 alpha-thrombin a1-a7 -> solvated Amber inputs (model id ph74_gaff2), CPU only.

Plan: docs/BRD4_HOU_EXECUTION_HANDOFF.md §3.1 (selection), §4.2/§4.4 (prep, solvent),
§7.1 (pH 7.4 ligand states), §7.2 (ligand identity = CCD), §7.10 (MODELLER loop, items 1-5).
Output is the same kind of input as BRD4 (prmtop + rst7 + a manifests/hou_inputs.csv row), so
scripts/run_smallmol_md.py and the online scorer take it unchanged:

    # on 183 (AmberTools in openmm_dev; the script sets AMBERHOME/PATH itself)
    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python scripts/prepare_hou_thrombin.py --sample a5
    ... --sample all --jobs 7          # entries in parallel, one sqm thread each

Per entry, written to prepared/hou/<id>/ph74_gaff2/:
  1. receptor heavy atoms: a1-a4 from modeller/receptor_heavy.pdb (loop B184-190 built on 181,
     MODELLER's OXT dropped); a5-a7 from the raw PDB chains L,H.  PDBFixer adds missing heavy
     atoms only (no missing residues, no terminal OXT).
  2. His states from reduce -NOFLIP -HIS -NUClear (same call as prep v2.1 on 183): HD1 -> HID,
     HE2 -> HIE, both -> HIP.  All H are then discarded and tleap rebuilds them.
  3. truncated chain ends get ACE / NME caps (light chain both ends, heavy chain C end); the
     heavy chain N end (Ile16 = B37 / H16) stays a charged N terminus.  Cap heavy atoms are
     placed from ideal internal coordinates (trans peptide); minimisation relaxes them.
  4. SSBOND records -> CYX + explicit S-S bonds in tleap (SG-SG distance recorded).
  5. ligand: CCD bond orders and atom names, heavy-atom coordinates from the PDB, pH 7.4 state
     (tertiary aliphatic amines +1, aryl amidines +1, carboxylic acids -1), RDKit H positions,
     antechamber AM1-BCC / GAFF2, parmchk2.  Net charge is checked against §7.1.
  6. crystal waters (all chains) and Na+ are kept, except waters whose O lies < 2.6 A from a
     heavy atom that was built rather than observed (loop, PDBFixer atoms, caps).  2.6 A is the
     lower end of observed O...N/O hydrogen-bond distances (2.6-3.2 A); anything closer is a
     clash with a modelled atom, not a contact.  Removed waters are listed in the record.
  7. tleap: ff14SB / GAFF2 / TIP3P (JC ions), rectangular box 12 A, neutralise + 0.15 M NaCl
     (n_salt = round(0.15 * N_wat / 55.5)), two passes (first one only counts water / charge).
Order in the prmtop: receptor, ligand, crystal water/Na+, solvent, ions -> ligand ends the solute.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
MODEL_ID = "ph74_gaff2"
SAMPLES = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
PDB_ID = dict(a1="1D3D", a2="1D3P", a3="1D3Q", a4="1D3T", a5="1DWB", a6="1DWC", a7="1DWD")
# chain roles; heavy N terminus (Ile16) is the real one, the other three ends are truncations
SERIES = {s: dict(light="A", heavy="B") for s in ("a1", "a2", "a3", "a4")}
SERIES.update({s: dict(light="L", heavy="H") for s in ("a5", "a6", "a7")})
EXPECTED_LIGAND_CHARGE = dict(a1=2, a2=2, a3=2, a4=2, a5=1, a6=0, a7=1)  # handoff §7.1
WATER_CLASH_A = 2.6
BOX_A, SALT_M = 12.0, 0.15
ENV_BIN = Path(sys.executable).parent


# ------------------------------------------------------------------ small helpers
def run(cmd, cwd: Path, log: Path, ok=(0,)):
    env = dict(os.environ, AMBERHOME=str(ENV_BIN.parent), OMP_NUM_THREADS="1",
               PATH=str(ENV_BIN) + os.pathsep + os.environ.get("PATH", ""))
    t = time.monotonic()
    with log.open("w") as fh:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                           shell=isinstance(cmd, str))
    if p.returncode not in ok:
        raise RuntimeError(f"{cmd} -> rc {p.returncode}; see {log}")
    return round(time.monotonic() - t, 2)


def pdb_atoms(path: Path, records=("ATOM", "HETATM")):
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith(records):
            continue
        if line[16] not in (" ", "A"):          # first alternate location only
            continue
        out.append(dict(record=line[:6].strip(), name=line[12:16].strip(), res=line[17:20].strip(),
                        chain=line[21], resseq=int(line[22:26]), icode=line[26].strip(),
                        xyz=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                        element=line[76:78].strip() or line[12:14].strip().lstrip("0123456789")[:1]))
    return out


def place(a, b, c, bond, angle_deg, torsion_deg):
    """NeRF: position d with |cd| = bond, angle(b,c,d), torsion(a,b,c,d)."""
    ang, tor = math.radians(angle_deg), math.radians(torsion_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * math.cos(ang), bond * math.sin(ang) * math.cos(tor), bond * math.sin(ang) * math.sin(tor)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


# ------------------------------------------------------------------ receptor
def receptor_heavy(sample: str, raw: Path, work: Path) -> tuple[list, dict]:
    """Observed/MODELLER heavy atoms + PDBFixer missing heavy atoms, as a residue list."""
    from openmm import app
    from pdbfixer import PDBFixer
    ser = SERIES[sample]
    if sample in ("a1", "a2", "a3", "a4"):
        src = PACKAGE / f"prepared/hou/{sample}/modeller/receptor_heavy.pdb"
        atoms = [x for x in pdb_atoms(src, ("ATOM",)) if x["name"] != "OXT"]
        built_res = {(ser["heavy"], r, "") for r in range(184, 191)}   # §7.10 loop
    else:
        src = raw
        atoms = [x for x in pdb_atoms(raw, ("ATOM",)) if x["chain"] in (ser["light"], ser["heavy"])]
        built_res = set()
    observed = {(x["chain"], x["resseq"], x["icode"], x["name"]) for x in atoms
                if (x["chain"], x["resseq"], x["icode"]) not in built_res}
    tmp = work / "receptor_in.pdb"
    with tmp.open("w") as fh:
        for i, x in enumerate(atoms, 1):
            fh.write("ATOM  %5d %-4s %3s %s%4d%1s   %8.3f%8.3f%8.3f  1.00  0.00          %2s\n" % (
                i % 100000, x["name"] if len(x["name"]) == 4 else " " + x["name"], x["res"], x["chain"],
                x["resseq"], x["icode"] or " ", *x["xyz"], x["element"]))
            if i == len(atoms) or atoms[i]["chain"] != x["chain"]:
                fh.write("TER\n")
        fh.write("END\n")
    fixer = PDBFixer(filename=str(tmp))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        raise RuntimeError(f"{sample}: nonstandard receptor residues {fixer.nonstandardResidues}")
    fixer.findMissingAtoms()
    fixer.missingTerminals = {}
    added = {f"{r.chain.id}{r.id}{r.insertionCode.strip()}:{r.name}": [a.name for a in v]
             for r, v in fixer.missingAtoms.items()}
    fixer.addMissingAtoms()
    pos = np.array([[v.x, v.y, v.z] for v in fixer.positions]) * 10.0
    residues = []
    for r in fixer.topology.residues():
        key = (r.chain.id, int(r.id), r.insertionCode.strip())
        ats = []
        for a in r.atoms():
            if a.element is None or a.element.symbol == "H":
                continue
            is_built = key in built_res or (key[0], key[1], key[2], a.name) not in observed
            ats.append(dict(name=a.name, element=a.element.symbol, xyz=pos[a.index], built=is_built))
        residues.append(dict(chain=key[0], resseq=key[1], icode=key[2], res=r.name, atoms=ats))
    return residues, dict(source=str(src), pdbfixer_added_heavy=added,
                          n_built_heavy=sum(a["built"] for r in residues for a in r["atoms"]))


def his_states(residues, ligand_atoms, work: Path) -> tuple[dict, float]:
    path = work / "reduce_in.pdb"
    lines, i = [], 0
    for r in residues:
        for a in r["atoms"]:
            i += 1
            lines.append("ATOM  %5d %-4s %3s %s%4d%1s   %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (
                i, a["name"] if len(a["name"]) == 4 else " " + a["name"], r["res"], r["chain"], r["resseq"],
                r["icode"] or " ", *a["xyz"], a["element"]))
    for a in ligand_atoms:
        i += 1
        lines.append("HETATM%5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (
            i, a["name"] if len(a["name"]) == 4 else " " + a["name"], a["res"], a["chain"], a["resseq"],
            *a["xyz"], a["element"]))
    path.write_text("\n".join(lines) + "\nEND\n")
    # reduce exits 1 when it modified the model (it always adds H here); >1 is an error
    wall = run(f"reduce -NOFLIP -HIS -NUClear {path.name} > reduce_out.pdb", work, work / "reduce.log", ok=(0, 1))
    hs = {}
    for x in pdb_atoms(work / "reduce_out.pdb", ("ATOM",)):
        if x["res"] == "HIS":
            hs.setdefault((x["chain"], x["resseq"], x["icode"]), set()).add(x["name"])
    states = {}
    for key, names in hs.items():
        d1, e2 = "HD1" in names, "HE2" in names
        states[key] = "HIP" if d1 and e2 else "HID" if d1 else "HIE"
    return states, wall


def add_caps(residues, sample) -> list:
    ser = SERIES[sample]
    out = []
    by_chain = {}
    for r in residues:
        by_chain.setdefault(r["chain"], []).append(r)
    for ch, rs in by_chain.items():
        first, last = rs[0], rs[-1]
        xyz = lambda r, n: next(a["xyz"] for a in r["atoms"] if a["name"] == n)
        if ch == ser["light"]:
            N, CA, C = xyz(first, "N"), xyz(first, "CA"), xyz(first, "C")
            c_ace = place(C, CA, N, 1.335, 121.9, -65.0)
            out.append(dict(chain=ch, resseq=first["resseq"], icode="cap", res="ACE", atoms=[
                dict(name="CH3", element="C", xyz=place(CA, N, c_ace, 1.522, 116.6, 180.0), built=True),
                dict(name="C", element="C", xyz=c_ace, built=True),
                dict(name="O", element="O", xyz=place(CA, N, c_ace, 1.229, 122.5, 0.0), built=True)]))
        out.extend(rs)
        N, CA, C, O = (xyz(last, n) for n in ("N", "CA", "C", "O"))
        n_nme = place(N, CA, C, 1.335, 116.2, dihedral(N, CA, C, O) + 180.0)
        out.append(dict(chain=ch, resseq=last["resseq"], icode="cap", res="NME", atoms=[
            dict(name="N", element="N", xyz=n_nme, built=True),
            # ff14SB NME names its methyl carbon C (ACE's is CH3)
            dict(name="C", element="C", xyz=place(CA, C, n_nme, 1.449, 121.9, 180.0), built=True)]))
    return out


# ------------------------------------------------------------------ ligand
def read_ccd(path: Path):
    """atoms [(id, element, charge, stereo)], bonds [(a, b, order)] from a CCD mmCIF."""
    rows = {"_chem_comp_atom.": [], "_chem_comp_bond.": []}
    cols = {"_chem_comp_atom.": [], "_chem_comp_bond.": []}
    cur = None
    import shlex
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("loop_") or s.startswith("#"):     # some CCD files close with "#   #"
            cur = None
            continue
        hit = next((k for k in cols if s.startswith(k)), None)
        if hit:
            cur = hit
            cols[hit].append(s[len(hit):].split()[0])
            continue
        if cur and s and not s.startswith("_"):
            rows[cur].append(dict(zip(cols[cur], shlex.split(s))))
    atoms = [(r["atom_id"], r["type_symbol"], int(r["charge"]), r["pdbx_stereo_config"])
             for r in rows["_chem_comp_atom."]]
    bonds = [(r["atom_id_1"], r["atom_id_2"], r["value_order"]) for r in rows["_chem_comp_bond."]]
    return atoms, bonds


def build_ligand(sample, code, lig_atoms, work: Path) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem  # noqa: F401  (registers AddHs coordinate generation)
    atoms, bonds = read_ccd(PACKAGE / f"sources/hou2011/ccd/{code}.cif")
    rw = Chem.RWMol()
    idx = {}
    for name, el, q, _ in atoms:
        a = Chem.Atom(el.capitalize() if len(el) > 1 else el)
        a.SetFormalCharge(q)
        a.SetNoImplicit(True)
        a.SetProp("_Name", name)
        idx[name] = rw.AddAtom(a)
    order = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}
    for a, b, o in bonds:
        rw.AddBond(idx[a], idx[b], order[o])
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    n_h_ccd = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "H")
    n_h = {a.GetProp("_Name"): sum(1 for b in a.GetNeighbors() if b.GetSymbol() == "H")
           for a in m.GetAtoms() if a.GetSymbol() != "H"}
    m = Chem.RemoveHs(m)
    for a in m.GetAtoms():       # H counts exactly as in the CCD definition, never inferred
        a.SetNumExplicitHs(n_h[a.GetProp("_Name")])
        a.SetNoImplicit(True)
    Chem.SanitizeMol(m)
    heavy = {a.GetProp("_Name"): a.GetIdx() for a in m.GetAtoms()}
    pdb = {a["name"]: a["xyz"] for a in lig_atoms}
    missing = sorted(set(heavy) - set(pdb))
    extra = sorted(set(pdb) - set(heavy))
    if missing or extra:
        raise RuntimeError(f"{sample} {code}: PDB vs CCD heavy atoms differ: missing {missing}, extra {extra}")
    conf = Chem.Conformer(m.GetNumAtoms())
    for name, i in heavy.items():
        conf.SetAtomPosition(i, pdb[name].tolist())
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)

    # pH 7.4 microstate (handoff §7.1)
    edits = []
    rules = [("tertiary_aliphatic_amine", "[NX3;H0;+0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S]);!$(N-a);!$(N-C=C)]([CX4])([CX4])[CX4]", 0, +1),
             ("aryl_amidine_imine", "[NX2;H1;+0]=[CX3](-[NX3;H2])-a", 0, +1),
             ("carboxylic_acid_OH", "[OX2;H1;+0]-[CX3](=O)-[#6]", 0, -1)]
    for label, smarts, k, dq in rules:
        for match in m.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            a = m.GetAtomWithIdx(match[k])
            nh = a.GetTotalNumHs()
            a.SetFormalCharge(a.GetFormalCharge() + dq)
            a.SetNumExplicitHs(nh + dq)
            a.SetNoImplicit(True)
            edits.append(dict(rule=label, atom=a.GetProp("_Name"), dq=dq))
    Chem.SanitizeMol(m)
    charge = Chem.GetFormalCharge(m)
    if charge != EXPECTED_LIGAND_CHARGE[sample]:
        raise RuntimeError(f"{sample} {code}: pH 7.4 state has charge {charge}, handoff §7.1 says "
                           f"{EXPECTED_LIGAND_CHARGE[sample]}; edits {edits}")
    Chem.AssignStereochemistryFrom3D(m)
    ccd_stereo = {n: s for n, _, _, s in atoms if s in ("R", "S")}
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(m)
    pose_stereo = {a.GetProp("_Name"): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    # §7.2: stereochemistry follows the CCD.  A deposited pose with the other configuration at a
    # centre is corrected by mirroring that centre's terminal heavy substituent (e.g. a methyl)
    # through the plane of the centre and its two other heavy neighbours; nothing else moves.
    stereo_fix = []
    for name in sorted(n for n in ccd_stereo if pose_stereo.get(n) != ccd_stereo[n]):
        c = m.GetAtomWithIdx(heavy[name])
        nbr = [b.GetIdx() for b in c.GetNeighbors()]
        term = [i for i in nbr if m.GetAtomWithIdx(i).GetDegree() == 1]
        if len(nbr) != 3 or c.GetTotalNumHs() != 1 or len(term) != 1:
            raise RuntimeError(f"{sample} {code}: pose stereo at {name} is {pose_stereo.get(name)}, CCD "
                               f"{ccd_stereo[name]}; no single terminal substituent to mirror, fix by hand")
        pos = m.GetConformer().GetPositions()
        t, (o1, o2) = term[0], [i for i in nbr if i != term[0]]
        nrm = np.cross(pos[o1] - pos[c.GetIdx()], pos[o2] - pos[c.GetIdx()])
        nrm /= np.linalg.norm(nrm)
        v = pos[t] - pos[c.GetIdx()]
        new = pos[c.GetIdx()] + v - 2 * np.dot(v, nrm) * nrm
        m.GetConformer().SetAtomPosition(t, new.tolist())
        stereo_fix.append(dict(centre=name, moved=m.GetAtomWithIdx(t).GetProp("_Name"),
                               shift_A=round(float(np.linalg.norm(new - pos[t])), 3)))
    if stereo_fix:
        Chem.AssignStereochemistryFrom3D(m, replaceExistingTags=True)
        rdCIPLabeler.AssignCIPLabels(m)
        fixed = {a.GetProp("_Name"): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
        still = sorted(n for n in ccd_stereo if fixed.get(n) != ccd_stereo[n])
        if still:
            raise RuntimeError(f"{sample} {code}: stereo still differs from CCD at {still} after mirroring")
    mh = Chem.AddHs(m, addCoords=True)
    for a in mh.GetAtoms():
        if not a.HasProp("_Name"):
            a.SetProp("_Name", "H")
    sdf = work / "ligand_in.sdf"
    names = [a.GetProp("_Name") for a in mh.GetAtoms()]
    xyz = mh.GetConformer().GetPositions()
    text = Chem.MolToMolBlock(mh)
    # resume: AM1-BCC is the slow step; reuse it only when the input molecule is the same text
    reuse = (sdf.exists() and sdf.read_text() == text and (work / "ligand_gaff2.mol2").exists()
             and (work / "ligand.frcmod").exists())
    if reuse:
        t_ac = t_pc = "reused"
    else:
        sdf.write_text(text)
        t_ac = run(["antechamber", "-i", sdf.name, "-fi", "mdl", "-o", "ligand_gaff2.mol2", "-fo", "mol2",
                    "-c", "bcc", "-nc", str(charge), "-at", "gaff2", "-rn", code, "-pf", "y"],
                   work, work / "antechamber.log")
        t_pc = run(["parmchk2", "-i", "ligand_gaff2.mol2", "-f", "mol2", "-o", "ligand.frcmod", "-s", "gaff2"],
                   work, work / "parmchk2.log")
    # put the crystal-pose coordinates back (antechamber keeps input order; sqm optimises a copy)
    lines = (work / "ligand_gaff2.mol2").read_text().splitlines()
    i0 = lines.index("@<TRIPOS>ATOM") + 1
    i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith("@<TRIPOS>"))
    if i1 - i0 != len(names):
        raise RuntimeError(f"{sample}: mol2 has {i1 - i0} atoms, sdf {len(names)}")
    shift, q_mol2 = [], []
    for k in range(i0, i1):
        f = lines[k].split()
        old = np.array([float(v) for v in f[2:5]])
        new = xyz[k - i0]
        shift.append(float(np.linalg.norm(old - new)))
        q_mol2.append(float(f[8]))
        f[2:5] = [f"{v:.4f}" for v in new]
        lines[k] = "%7s %-8s %10s %10s %10s %-6s %4s %-8s %10s" % tuple(f[:9])
    (work / "ligand.mol2").write_text("\n".join(lines) + "\n")
    attn = (work / "ligand.frcmod").read_text().count("ATTN")
    return dict(ccd=code, n_heavy=len(heavy), n_atoms=len(names), n_h_ccd_definition=n_h_ccd,
                formal_charge=charge, protonation_edits=edits, ccd_cip=ccd_stereo, pose_cip=pose_stereo,
                cip_mismatch=sorted(n for n in ccd_stereo if pose_stereo.get(n) != ccd_stereo[n]),
                stereo_corrected_to_ccd=stereo_fix,
                am1bcc_charge_sum=round(sum(q_mol2), 6), mol2_vs_input_max_shift_A=round(max(shift), 4),
                parmchk2_attn_lines=attn, source_atom_order=names,
                wall_s=dict(antechamber=t_ac, parmchk2=t_pc))


# ------------------------------------------------------------------ assembly
def write_pdb(path: Path, residues, ter_between_chains=True):
    lines, serial, rnum, prev_chain = [], 0, 0, None
    numbering = {}
    for r in residues:
        if prev_chain is not None and ter_between_chains and r["chain"] != prev_chain:
            lines.append("TER")
        prev_chain = r["chain"]
        rnum += 1
        numbering[(r["chain"], r["resseq"], r["icode"])] = rnum
        for a in r["atoms"]:
            serial += 1
            nm = a["name"] if len(a["name"]) == 4 else " " + a["name"]
            lines.append("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (
                serial % 100000, nm, r["res"], "A", rnum % 10000, *a["xyz"], a["element"]))
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n")
    return numbering


def tleap(work: Path, tag: str, n_rec_res: int, ss_pairs, ions: tuple[int, int] | None) -> float:
    lines = ["source leaprc.protein.ff14SB", "source leaprc.gaff2", "source leaprc.water.tip3p",
             "loadamberparams ligand.frcmod", "LIG = loadmol2 ligand.mol2", "REC = loadpdb receptor_amber.pdb"]
    lines += [f"bond REC.{i}.SG REC.{j}.SG" for i, j in ss_pairs]
    lines += ["XTAL = loadpdb xtal_water_ions.pdb" if (work / "xtal_water_ions.pdb").exists() else "",
              "COMPLEX = combine {REC LIG" + (" XTAL}" if (work / "xtal_water_ions.pdb").exists() else "}"),
              "check COMPLEX", "charge COMPLEX", f"solvatebox COMPLEX TIP3PBOX {BOX_A}"]
    if ions:
        n_na, n_cl = ions
        lines.append(f"addionsrand COMPLEX Na+ {n_na} Cl- {n_cl}")
    lines += ["charge COMPLEX", f"saveamberparm COMPLEX {tag}.prmtop {tag}.rst7", "quit"]
    (work / f"tleap_{tag}.in").write_text("\n".join(x for x in lines if x) + "\n")
    for f in (f"{tag}.prmtop", f"{tag}.rst7"):
        if (work / f).exists():
            (work / f).unlink()
    wall = run(["tleap", "-f", f"tleap_{tag}.in"], work, work / f"tleap_{tag}.log")
    log = (work / f"tleap_{tag}.log").read_text()
    if "FATAL" in log or "Errors = 0" not in log or not (work / f"{tag}.prmtop").exists():
        raise RuntimeError(f"tleap {tag} reported errors; see tleap_{tag}.log")
    return wall


def prepare(sample: str) -> dict:
    t0 = time.monotonic()
    code_row = next(r for r in csv.DictReader((PACKAGE / "manifests/hou_thrombin_selection.csv").open())
                    if r["sample_id"] == sample)
    code, lig_chain, lig_seq = code_row["ligand_resname"], code_row["ligand_chain"], int(code_row["ligand_resseq"])
    raw = PACKAGE / f"sources/hou2011/pdb/{PDB_ID[sample]}.pdb"
    work = PACKAGE / f"prepared/hou/{sample}/{MODEL_ID}"
    work.mkdir(parents=True, exist_ok=True)
    rec = {"sample": sample, "pdb": PDB_ID[sample], "model_id": MODEL_ID, "ligand": code}

    het = pdb_atoms(raw, ("HETATM",))
    lig_atoms = [x for x in het if x["res"] == code and x["chain"] == lig_chain and x["resseq"] == lig_seq]
    residues, rec["receptor"] = receptor_heavy(sample, raw, work)
    states, rec.setdefault("wall_s", {})["reduce"] = his_states(residues, lig_atoms, work)
    rec["his_states"] = {f"{k[0]}{k[1]}{k[2]}": v for k, v in sorted(states.items())}

    # disulfides from the raw SSBOND records
    ss = []
    for line in raw.read_text().splitlines():
        if line.startswith("SSBOND"):
            ss.append(((line[15], int(line[17:21]), line[21].strip()), (line[29], int(line[31:35]), line[35].strip())))
    ss_keys = {k for pair in ss for k in pair}
    for r in residues:
        key = (r["chain"], r["resseq"], r["icode"])
        if r["res"] == "HIS":
            r["res"] = states[key]
        if r["res"] == "CYS" and key in ss_keys:
            r["res"] = "CYX"
    residues = add_caps(residues, sample)
    numbering = write_pdb(work / "receptor_amber.pdb", residues)
    sg = {k: next(a["xyz"] for a in r["atoms"] if a["name"] == "SG")
          for r in residues for k in [(r["chain"], r["resseq"], r["icode"])] if r["res"] == "CYX"}
    rec["disulfides"] = [dict(a=f"{a[0]}{a[1]}{a[2]}", b=f"{b[0]}{b[1]}{b[2]}",
                              leap=[numbering[a], numbering[b]], sg_sg_A=round(float(np.linalg.norm(sg[a] - sg[b])), 3))
                         for a, b in ss]
    rec["caps"] = [f"{r['chain']}{r['resseq']}:{r['res']}" for r in residues if r["res"] in ("ACE", "NME")]

    # crystal waters / Na+: drop waters clashing with built heavy atoms
    built = np.array([a["xyz"] for r in residues for a in r["atoms"] if a["built"]]).reshape(-1, 3)
    waters = [x for x in het if x["res"] == "HOH" and x["name"] == "O"]
    keep, removed = [], []
    for w in waters:
        dmin = float(np.linalg.norm(built - w["xyz"], axis=1).min()) if len(built) else 99.0
        (removed if dmin < WATER_CLASH_A else keep).append((w, dmin))
    nas = [x for x in het if x["res"] == "NA"]
    rec["crystal_solvent"] = dict(waters_in_pdb=len(waters), waters_kept=len(keep), na_kept=len(nas),
                                  waters_removed=[dict(id=f"{w['chain']}{w['resseq']}", min_dist_to_built_A=round(d, 2))
                                                  for w, d in removed], clash_cutoff_A=WATER_CLASH_A)
    xtal = work / "xtal_water_ions.pdb"
    if keep or nas:
        lines = []
        for k, (w, _) in enumerate(keep, 1):
            lines.append("HETATM%5d  O   WAT W%4d    %8.3f%8.3f%8.3f  1.00  0.00           O" % (k, k, *w["xyz"]))
            lines.append("TER")
        for k, x in enumerate(nas, 1):
            lines.append("HETATM%5d Na+  Na+ I%4d    %8.3f%8.3f%8.3f  1.00  0.00          Na" % (len(keep) + k, k, *x["xyz"]))
            lines.append("TER")
        xtal.write_text("\n".join(lines) + "\nEND\n")
    elif xtal.exists():
        xtal.unlink()

    rec["ligand_prep"] = build_ligand(sample, code, lig_atoms, work)
    rec["wall_s"].update(rec["ligand_prep"].pop("wall_s"))

    # tleap pass 1 (count water, charge) -> ions -> pass 2
    import parmed
    ss_leap = [d["leap"] for d in rec["disulfides"]]
    rec["wall_s"]["tleap_pass1"] = tleap(work, "pass1", len(numbering), ss_leap, None)
    p1 = parmed.load_file(str(work / "pass1.prmtop"))
    q1 = round(sum(a.charge for a in p1.atoms))
    n_wat = sum(1 for r in p1.residues if r.name == "WAT")
    n_salt = round(SALT_M * n_wat / 55.5)
    n_na, n_cl = n_salt + max(0, -q1), n_salt + max(0, q1)
    rec["ions"] = dict(pass1_net_charge=q1, pass1_waters=n_wat, n_salt_pairs=n_salt, na_added=n_na, cl_added=n_cl,
                       rule=f"n_salt = round({SALT_M} * N_wat / 55.5); neutralise with the counter-ion")
    rec["wall_s"]["tleap_final"] = tleap(work, "complex_solv", len(numbering), ss_leap, (n_na, n_cl))
    for f in ("pass1.prmtop", "pass1.rst7"):
        (work / f).unlink()

    # final audit on the real prmtop
    p = parmed.load_file(str(work / "complex_solv.prmtop"), xyz=str(work / "complex_solv.rst7"))
    lig_res = [i for i, r in enumerate(p.residues) if r.name == code]
    if len(lig_res) != 1:
        raise RuntimeError(f"{sample}: {len(lig_res)} residues named {code} in prmtop")
    li = lig_res[0]
    lig_idx = [a.idx for a in p.residues[li].atoms]
    lig0, lig1 = lig_idx[0], lig_idx[-1] + 1
    after = sorted({r.name for r in p.residues[li + 1:]})
    if not set(after) <= {"WAT", "Na+", "Cl-"}:
        raise RuntimeError(f"{sample}: non-solvent residues after the ligand: {after}")
    q = np.array([a.charge for a in p.atoms])
    ss_bonds = sum(1 for b in p.bonds if b.atom1.name == "SG" and b.atom2.name == "SG")
    n_wat_final = sum(1 for r in p.residues if r.name == "WAT")
    rec["final"] = dict(n_atoms=len(p.atoms), n_residues=len(p.residues), ligand_residue_1based=li + 1,
                        ligand_atoms=[lig0, lig1], n_solute_atoms=lig1, residues_after_ligand=after,
                        charge_e=dict(total=float(q.sum()), solute=float(q[:lig1].sum()), receptor=float(q[:lig0].sum()),
                                      ligand=float(q[lig0:lig1].sum())),
                        ss_bonds_in_prmtop=ss_bonds, n_waters=n_wat_final,
                        box_A=[round(float(v), 3) for v in p.box[:3]] if p.box is not None else None)
    rec["inputs_row"] = dict(sample=sample, ligand_residue=code, ligand_residue_1based=li + 1,
                             ligand_atom_start_0based=lig0, ligand_atom_end_exclusive=lig1,
                             n_ligand_atoms=lig1 - lig0, n_solute_atoms=lig1, n_system_atoms=len(p.atoms),
                             ligand_charge=float(q[lig0:lig1].sum()), n_waters=n_wat_final,
                             prmtop=str(work / "complex_solv.prmtop"), coordinates=str(work / "complex_solv.rst7"),
                             status="prepared_ph74_gaff2_not_molecularly_validated")
    rec["tools"] = dict(python=sys.executable, host=os.uname().nodename,
                        antechamber=shutil.which("antechamber", path=str(ENV_BIN)))
    rec["wall_s"]["total"] = round(time.monotonic() - t0, 2)
    (work / "prep_record.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return rec


def write_inputs_manifest():
    rows = []
    for s in SAMPLES:
        f = PACKAGE / f"prepared/hou/{s}/{MODEL_ID}/prep_record.json"
        if f.exists():
            rows.append(json.loads(f.read_text())["inputs_row"])
    if not rows:
        return None
    out = PACKAGE / "manifests/hou_inputs.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out, len(rows)


def one(sample):
    try:
        r = prepare(sample)
        return sample, "ok", dict(total_s=r["wall_s"]["total"], n_atoms=r["final"]["n_atoms"],
                                  solute_q=round(r["final"]["charge_e"]["solute"], 4),
                                  waters_removed=len(r["crystal_solvent"]["waters_removed"]),
                                  cip_mismatch=r["ligand_prep"]["cip_mismatch"],
                                  attn=r["ligand_prep"]["parmchk2_attn_lines"])
    except Exception as e:  # one entry failing does not stop the others (handoff §1)
        return sample, "FAIL", repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="a1..a7 or all")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    todo = SAMPLES if a.sample == "all" else a.sample.split(",")
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for s, st, info in ex.map(one, todo):
            print(json.dumps(dict(sample=s, status=st, info=info), default=str), flush=True)
    print(json.dumps({"inputs_manifest": write_inputs_manifest()}, default=str))


if __name__ == "__main__":
    main()
