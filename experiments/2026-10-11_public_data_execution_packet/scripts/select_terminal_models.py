#!/usr/bin/env python3
"""Prep protocol v2, step 2: C1 checks on the MODELLER models and the two-start choice.

    cd /home/ruigengji/JAXPBSA
    $PY experiments/2026-10-11_public_data_execution_packet/scripts/select_terminal_models.py --res12 PRO

($PY = openmm_dev python; only file parsing + numpy, no simulation.)
Design: docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §4–§5.

C1 per model (heavy atoms; limits = 4 sigma of the ff14SB harmonic term at 298.15 K):
  anchors        every atom outside A:25 / B:12 within 0.001 A of the clean template
                 (matched by residue + element: MODELLER renames symmetric atoms)
  stereo         all CA L, all non-Pro omega trans (peptide_geometry)
  junction       peptide_geometry.junction_report: C-N, N-CA bonds and CA-C-N, O-C-N,
                 C-N-CA (C-N-CD for Pro), N-CA-C angles at A25-A26 and B11-B12
  pro ring       all ring bonds and angles, ring torsions (|t| > 90 deg = not a ring);
                 chi1 sign recorded as pucker
  clash          non-bonded heavy-atom pairs (> 3 bonds apart) touching residues
                 25, 26, 11, 12 closer than 2.2 A
Start 1 = lowest DOPE among passing models (cis Ser11-Pro12 excluded, design §3); start 2 = among the next four passing
models by DOPE, the one with the largest heavy-atom RMSD of A:25 + B:12 to start 1
(anchors fixed, no superposition).  Writes selection.json, start_v2m1.pdb,
start_v2m2.pdb next to the models.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                      # noqa: E402
import peptide_geometry as geo          # noqa: E402

BUILT = (("A", 25), ("B", 12))
JUNCTIONS = ((("A", 25), ("A", 26)), (("B", 11), ("B", 12)))
JUNCTION = (("A", 25), ("A", 26), ("B", 11), ("B", 12))


def read_pdb(path: Path):
    atoms = []
    for ln in path.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            atoms.append(dict(chain=ln[21], resnum=int(ln[22:26]), resname=ln[17:20].strip(),
                              name=ln[12:16].strip(),
                              xyz=np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])))
    return atoms


def bonds_of(path: Path):
    from openmm import app
    top = app.PDBFile(str(path)).topology
    return [(a.index, b.index) for a, b in top.bonds()]


def check(path: Path, tmpl: dict) -> dict:
    atoms = read_pdb(path)
    xyz = np.array([a["xyz"] for a in atoms])
    idx = {(a["chain"], a["resnum"], a["name"]): i for i, a in enumerate(atoms)}
    res = {}
    for a in atoms:
        res.setdefault((a["chain"], a["resnum"]), a["resname"])
    fails = []

    # anchors: MODELLER renames symmetric atoms to its convention (seen: Phe91/Tyr60
    # CD1<->CD2, CE1<->CE2; Glu95 OE1<->OE2), so match each template atom to the nearest
    # model atom of the same residue and element instead of by name
    by_res = {}
    for i, a in enumerate(atoms):
        by_res.setdefault((a["chain"], a["resnum"], a["name"][0]), []).append(i)
    dev, renamed = [], []
    for k, v in tmpl.items():
        cand = by_res.get((k[0], k[1], k[2][0]), [])
        if not cand:
            dev.append(float("inf")); continue
        d = [float(np.linalg.norm(xyz[i] - v)) for i in cand]
        j = int(np.argmin(d))
        dev.append(d[j])
        if atoms[cand[j]]["name"] != k[2] and d[j] <= 0.001:
            renamed.append(f"{k[0]}:{k[1]}.{k[2]}->{atoms[cand[j]]['name']}")
    anchor_max = max(dev) if dev else None
    if anchor_max > 0.001:
        fails.append(f"anchors moved: max {anchor_max:.4f} A")

    # stereo (all residues)
    residues = []
    for i, a in enumerate(atoms):
        key = (a["chain"], a["resname"], str(a["resnum"]))
        if not residues or residues[-1][:3] != key:
            residues.append((*key, {}))
        residues[-1][3][a["name"]] = i
    meas = geo.measure(geo.stereo_terms(residues), xyz)
    for v in geo.prepared_violations(meas):
        fails.append(f"stereo {v['label']} {v['kind']} {v['deg']:.1f}")
    omega = {m["label"]: round(m["deg"], 1) for m in meas
             if m["kind"] == "omega" and m["label"] in ("A:GLU25-THR26", f"B:SER11-{res[('B', 12)]}12")}

    # junction geometry + Pro ring (ff14SB 4 sigma, peptide_geometry.junction_report)
    geom, jf = geo.junction_report(residues, xyz, JUNCTIONS)
    fails += jf
    pucker = None
    if res[("B", 12)] == "PRO":
        pucker = dict(ring_torsions_deg=geom["B12 ring_torsions_deg"], pucker=geom["B12 pucker"])

    # heavy-atom clashes around the junctions (no hydrogens in MODELLER output)
    focus = [i for i, a in enumerate(atoms) if (a["chain"], a["resnum"]) in JUNCTION]
    lab = lambda i: f"{atoms[i]['chain']}:{atoms[i]['resname']}{atoms[i]['resnum']}.{atoms[i]['name']}"
    clashes, closest = geo.close_contacts(xyz, bonds_of(path), focus, [False] * len(atoms))
    for d, i, j in clashes:
        fails.append(f"clash {d:.2f} A {lab(i)}--{lab(j)}")
    closest = (closest[0], f"{lab(closest[1])}--{lab(closest[2])}")

    built_xyz = {k: xyz[i].tolist() for k, i in idx.items() if k[:2] in BUILT}
    return dict(passed=not fails, failures=fails, anchor_max_dev_A=anchor_max, anchor_atoms_renamed=renamed,
                junction=geom, omega=omega,
                pro_ring=pucker, closest_nonbonded_A=round(closest[0], 3), closest_pair=closest[1],
                _built=built_xyz)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res12", required=True)
    a = ap.parse_args()
    out = PACKET / "inputs" / "modeller_v2" / a.res12.upper()
    rec = json.loads((out / "build_record.json").read_text())
    tmpl = {(x["chain"], x["resnum"], x["name"]): x["xyz"] for x in read_pdb(out / "template_3eqs_clean.pdb")}

    results = {}
    for name, m in sorted(rec["models"].items()):
        if "failure" in m:
            results[name] = dict(passed=False, failures=[f"modeller: {m['failure']}"])
            continue
        r = check(out / name, tmpl)
        r.update(dope=m["dope"], molpdf=m["molpdf"], seed=m["seed"])
        results[name] = r
    # design §3: a cis Ser11-Pro12 bond is physically possible but not used as a start
    cis_pro = {n for n, r in results.items() if r.get("pro_ring") and abs(next(iter(
        v for k, v in r["omega"].items() if k.startswith("B:"))) ) < 90}
    passing = sorted((n for n, r in results.items() if r["passed"] and n not in cis_pro),
                     key=lambda n: results[n]["dope"])
    sel = dict(rule="start1 = lowest DOPE among C1-passing models; start2 = among the next 4 passing by DOPE, "
                    "largest heavy-atom RMSD of A:25+B:12 to start1 (no superposition)",
               n_models=len(results), n_passing_c1=sum(r["passed"] for r in results.values()),
               excluded_cis_pro=sorted(cis_pro), n_candidates=len(passing))
    if len(passing) >= 2:
        s1 = passing[0]
        b1 = results[s1]["_built"]
        def rmsd(n):
            b = results[n]["_built"]
            keys = sorted(set(b) & set(b1))
            return math.sqrt(sum(sum((p - q) ** 2 for p, q in zip(b[k], b1[k])) for k in keys) / len(keys))
        cand = [(round(rmsd(n), 3), n) for n in passing[1:5]]
        s2 = max(cand)[1]
        sel.update(start1=s1, start2=s2, start2_candidates_rmsd_A=dict((n, d) for d, n in cand),
                   rmsd_start1_start2_A=max(cand)[0])
        shutil.copyfile(out / s1, out / "start_v2m1.pdb")
        shutil.copyfile(out / s2, out / "start_v2m2.pdb")
    for r in results.values():
        r.pop("_built", None)
    (out / "selection.json").write_text(json.dumps(dict(selection=sel, models=results), indent=2) + "\n")
    print(json.dumps(sel, indent=1))
    for n, r in sorted(results.items(), key=lambda x: x[1].get("dope", 0)):
        print(f"{n} pass={r['passed']} dope={r.get('dope', float('nan')):.1f} omega={r.get('omega')} "
              f"pro={r.get('pro_ring')} closest={r.get('closest_nonbonded_A')} fails={r['failures'][:3]}")


if __name__ == "__main__":
    main()
