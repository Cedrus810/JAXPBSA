#!/usr/bin/env python3
"""Prep protocol v2.1, stage 4: reduce-protonated model -> ff14SB atom names + static checks.

Run with the openmm_dev python on 181 (numpy + XML parsing only; no OpenMM system is
built, no minimisation):

    cd /home/ruigengji/JAXPBSA
    $PY experiments/2026-10-11_public_data_execution_packet/scripts/finalize_allh_models.py --sample PMI --start v2m1
    $PY experiments/2026-10-11_public_data_execution_packet/scripts/finalize_allh_models.py --all

Design: docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §11.  For every residue the atom-name
set is compared with the ff14SB residue template prep will use (N<res> for the first
residue of a chain, C<res> for the last, HID/HIE/HIP by which of HD1/HE2 exist).  Only
the renames in RENAME are applied; any other difference is an error.  Coordinates are
copied as written by reduce.

Static checks (geometry only, the full C2 runs in prep on the parameterised system):
  - every CA L, every non-Pro omega trans (peptide_geometry class definitions)
  - HA on the correct side of every non-Gly CA: angle(CA->HA, ideal) < 90 deg (§11)
  - heavy atoms identical to the stage-1 start (<= 0.001 A) except mutated side chains
  - non-bonded contacts (> 3 bonds): heavy-heavy >= 2.2 A, any pair with H >= 1.0 A (§5)

Input: reduced_v2mN.pdb (add_h_reduce.sh on heavy_v2mN.pdb from build_mutant_modeller.py).
Output: inputs/modeller_v21/<peptide_name>/start_v21mN.pdb and
finalize_record.json.  A model that fails a check gets no start_*.pdb.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import peptide_geometry as geo                         # noqa: E402

FF_XML = Path(sys.prefix) / "lib/python3.12/site-packages/openmmforcefields/ffxml/amber/ff14SB.xml"
AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))
# reduce -> ff14SB names, keyed by template position (reduce already writes H1/H2/H3 on N termini)
RENAME = {}


def ff_templates() -> dict:
    root = ET.parse(FF_XML).getroot()
    return {r.get("name"): {a.get("name") for a in r.findall("Atom")} for r in root.iter("Residue")}


def read_pdb(path: Path):
    """[(chain, resnum, resname, [(name, xyz, line)])] in file order."""
    res, key = [], None
    for ln in path.read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        k = (ln[21], int(ln[22:26]), ln[17:20])
        if k != key:
            res.append((k[0], k[1], k[2], []))
            key = k
        res[-1][3].append((ln[12:16].strip(), np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]), ln))
    return res


def bonds_by_distance(names_xyz):
    """Covalent bonds from distance (1.2 x sum of covalent radii); used only to exclude
    1-2/1-3/1-4 pairs from the contact check."""
    rad = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
    el = [n[0] if n[0] in rad else "C" for n, _ in names_xyz]
    xyz = np.array([x for _, x in names_xyz])
    out = []
    for i in range(len(xyz)):
        d = np.linalg.norm(xyz[i + 1:] - xyz[i], axis=1)
        for j in np.where(d < 1.2 * (rad[el[i]] + np.array([rad[e] for e in el[i + 1:]])))[0]:
            out.append((i, i + 1 + int(j)))
    return out


def finalize(name: str, sid: str, tmpl: dict) -> dict:
    d = PACKET / "inputs" / "modeller_v21" / name
    src = d / f"reduced_{sid}.pdb"
    rec = json.loads((d / "build_record.json").read_text())["models"][f"heavy_{sid}.pdb"]
    mutated = {("B", i) for i in rec["mutated_peptide_positions"]}
    residues = read_pdb(src)
    fails, renamed, lines = [], [], []
    for k, (c, r, rn, atoms) in enumerate(residues):
        first = k == 0 or residues[k - 1][0] != c
        last = k == len(residues) - 1 or residues[k + 1][0] != c
        names = {n for n, _, _ in atoms}
        base = rn
        if rn == "HIS":
            base = {(True, False): "HID", (False, True): "HIE", (True, True): "HIP"}.get(("HD1" in names, "HE2" in names), "HIS")
        tname = ("N" if first else "C" if last else "") + base
        ren = RENAME.get("N", {}) if first else {}
        new = {ren.get(n, n) for n in names}
        if tname not in tmpl:
            fails.append(f"{c}:{rn}{r}: no ff14SB template {tname}")
        elif new != tmpl[tname]:
            fails.append(f"{c}:{rn}{r} vs {tname}: missing {sorted(tmpl[tname] - new)} extra {sorted(new - tmpl[tname])}")
        for n, _, ln in atoms:
            m = ren.get(n, n)
            if m != n:
                renamed.append(f"{c}:{rn}{r}.{n}->{m}")
            lines.append(ln[:12] + (f" {m:<3s}" if len(m) < 4 else m) + ln[16:])

    flat = [(n, x) for _, _, _, atoms in residues for n, x, _ in atoms]
    xyz = np.array([x for _, x in flat])
    idx, i = [], 0
    for c, r, rn, atoms in residues:
        idx.append((c, rn, str(r), {n: i + j for j, (n, _, _) in enumerate(atoms)}))
        i += len(atoms)

    stereo = geo.measure(geo.stereo_terms(idx), xyz)
    bad = geo.prepared_violations(stereo)
    fails += [f"stereo {b['label']} {b['kind']} {b['deg']:.1f}" for b in bad]

    ha = geo.ha_direction(idx, xyz)
    ha_worst = max(((ang, lab_) for lab_, ang in ha), default=(0.0, None))
    fails += [f"HA wrong side {lab_} {ang:.1f} deg" for lab_, ang in ha if ang >= 90.0]

    start = {}
    for ln in Path(rec["start"]).read_text().splitlines():
        if ln.startswith("ATOM"):
            start[(ln[21], int(ln[22:26]), ln[12:16].strip())] = np.array(
                [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    by = {}
    for c, rn, r, at in idx:
        for n, i in at.items():
            if n[0] != "H":
                by.setdefault((c, int(r), n[0]), []).append(i)
    dev = 0.0
    for (c, r, n), v in start.items():
        if (c, r) in mutated and n not in ("N", "CA", "C", "O", "OXT"):
            continue
        cand = by.get((c, r, n[0]), [])            # residue + element: MODELLER swaps symmetric names
        dev = max(dev, min((float(np.linalg.norm(xyz[i] - v)) for i in cand), default=float("inf")))
    if dev > 0.001:
        fails.append(f"heavy atoms moved vs stage-1 start: max {dev:.4f} A")

    is_h = [n[0] == "H" for n, _ in flat]
    clashes, closest = geo.close_contacts(xyz, bonds_by_distance(flat), range(len(flat)), is_h)
    lab = lambda i: next(f"{c}:{rn}{r}.{n}" for c, rn, r, at in idx for n, j in at.items() if j == i)
    fails += [f"clash {dd} A {lab(i)}--{lab(j)}" for dd, i, j in clashes]

    out = dict(source=str(src), n_atoms=len(flat), renamed=renamed, mutated=sorted(f"{c}:{r}" for c, r in mutated),
               ha_worst_deg=[round(ha_worst[0], 1), ha_worst[1]], max_heavy_dev_vs_start_A=round(dev, 4),
               closest_nonbonded=[round(closest[0], 3), lab(closest[1]), lab(closest[2])],
               stereo_terms=len(stereo), failures=fails, passed=not fails)
    dst = d / f"start_{sid.replace('v2', 'v21')}.pdb"
    if not fails:
        dst.write_text("\n".join(lines) + "\nTER\nEND\n")
        out["written"] = str(dst)
    elif dst.exists():
        dst.unlink()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--start", nargs="+", default=["v2m1", "v2m2"])
    a = ap.parse_args()
    names = ([r["peptide_name"] for r in csv.DictReader((PACKET / "sample_structure_audit.csv").open())]
             if a.all else a.sample)
    tmpl = ff_templates()
    any_fail = False
    for name in names:
        rp = PACKET / "inputs" / "modeller_v21" / name / "finalize_record.json"
        record = json.loads(rp.read_text()) if rp.exists() else {}
        for sid in a.start:
            r = finalize(name, sid, tmpl)
            record[sid] = r
            any_fail |= not r["passed"]
            print(json.dumps({f"{name}/{sid}": {k: r[k] for k in ("passed", "n_atoms", "renamed", "ha_worst_deg",
                                                                  "max_heavy_dev_vs_start_A", "closest_nonbonded")},
                              "failures": r["failures"][:10]}), flush=True)
        rp.write_text(json.dumps(record, indent=2) + "\n")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
