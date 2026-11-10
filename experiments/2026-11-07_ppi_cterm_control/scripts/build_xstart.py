#!/usr/bin/env python3
"""C-terminal start control, stage 1x: heavy-atom start with the peptide backbone of 3LNZ.

Plan: docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md §4 R2/R3, §11; user 2026-11-07 chose
"整条肽骨架取自 3LNZ".  One factor differs from model v21m1 of the same sample: the peptide
(B1-12) heavy atoms come from the experimental MDM2/PMI-N8A complex 3LNZ (receptor chain G +
peptide chain H, Pro12 observed) instead of 3EQS B1-11 + MODELLER-built B12.  Receptor A25-109
(3EQS anchors + the MODELLER A25 of start v2m1) is copied byte-identical from
2026-10-11 packet inputs/modeller_v2/<PRO|GLN>/start_v2m1.pdb.

3LNZ is superposed on 3EQS by receptor CA 27-108 (observed in both).  The peptide keeps the 3LNZ
sequence TSFAEYWALLSP; stage 2 (build_xmutant_modeller.py) mutates it to the sample sequence.
Checks (C1 definitions, peptide_geometry.py): every CA L, every non-Pro omega trans, backbone
bond/angle terms within 4 sigma, no non-bonded heavy-atom pair < 2.2 A, receptor atoms identical
to the source start.  Run with the openmm_dev python on 181 (numpy only):

    cd /home/ruigengji/JAXPBSA
    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python experiments/2026-11-07_ppi_cterm_control/scripts/build_xstart.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
NEW = HERE.parents[1]
ROOT = HERE.parents[3]
OLD = ROOT / "experiments/2026-10-11_public_data_execution_packet"
REF = ROOT / "experiments/2026-11-06_ppi_structure_review/r3/raw_pdb/3LNZ.pdb"
sys.path.insert(0, str(ROOT / "experiments/2026-11-06_ppi_structure_review/scripts"))
import stage_trace as T                                          # noqa: E402

SAMPLES = {"PMI": "PRO", "P12Q_L10A": "GLN"}      # stage-1 batch used by that sample's v21m1
REC_CHAIN, PEP_CHAIN = "G", "H"                   # 3LNZ copy with Pro12 observed (first in file)


def atoms(path, chain, altloc=" A"):
    out = []
    for ln in path.read_text().splitlines():
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith("ATOM") and ln[21] == chain and ln[16] in altloc and ln[76:78].strip() != "H":
            out.append(ln)
    return out


def xyz(ln):
    return np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])


def main():
    raw = T.read_pdb(T.RAW, raw=True)
    rec = {(int(l[22:26]), l[12:16].strip()): xyz(l) for l in atoms(REF, REC_CHAIN)}
    keys = [(r, "CA") for r in range(27, 109) if (r, "CA") in rec and ("A", r, "CA") in raw.idx]
    R, t = T.kabsch(np.array([rec[k] for k in keys]), np.array([raw.xyz[raw.idx[("A",) + k]] for k in keys]))
    pep = atoms(REF, PEP_CHAIN)
    resnums = sorted({int(l[22:26]) for l in pep})
    if resnums != list(range(1, 13)):
        raise SystemExit(f"3LNZ chain {PEP_CHAIN}: residues {resnums}")
    report = {}
    for sample, batch in SAMPLES.items():
        src = OLD / "inputs/modeller_v2" / batch / "start_v2m1.pdb"
        rec_lines = [l for l in src.read_text().splitlines() if l.startswith("ATOM") and l[21] == "A"]
        lines = list(rec_lines)
        n = len(rec_lines)
        for l in pep:
            x = R @ xyz(l) + t
            n += 1
            lines.append(f"ATOM  {n:5d} {l[12:16]} {l[17:20]} B{int(l[22:26]):4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}"
                         f"  1.00{l[60:66]}           {l[76:78].strip():>1s}")
        out = NEW / "inputs/modeller_v2x" / sample
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "start_v2mx1.pdb"
        dst.write_text("\n".join(lines) + "\nTER\nEND\n")

        s = T.read_pdb(dst)
        old = T.read_pdb(src)
        # receptor identical to the source start
        rdev = max(float(np.linalg.norm(s.xyz[s.idx[k]] - old.xyz[old.idx[k]])) for k in old.idx if k[0] == "A")
        meas = T.geo.measure(T.geo.stereo_terms(s.geo_residues()), s.xyz)
        bad = T.geo.prepared_violations(meas)
        nt, nout, zw, zl = T.backbone_z(s)
        bonds = T.bonds_from(s)
        pairs = [(s.idx[a], s.idx[b]) for a, b in bonds]
        clashes, closest = T.geo.close_contacts(s.xyz, pairs, range(len(s.name)), list(s.is_h))
        A = np.array([c == "A" for c in s.chain]); B = ~A
        D = np.linalg.norm(s.xyz[B][:, None] - s.xyz[A][None], axis=2)
        i, j = np.unravel_index(D.argmin(), D.shape)
        omega12 = next(round(m["deg"], 1) for m in meas if m["kind"] == "omega" and m["label"].startswith("B:") and m["label"].endswith("12"))
        fails = ([f"receptor moved {rdev:.4f} A"] if rdev > T.PDB_PREC else []) + \
                [f"stereo {b['label']} {b['kind']} {b['deg']:.1f}" for b in bad] + \
                ([f"{nout} backbone terms beyond 4 sigma (worst {zl} z={zw})"] if nout else []) + \
                [f"clash {d} A {s.label(a)}--{s.label(b)}" for d, a, b in clashes]
        report[sample] = dict(
            source_receptor=str(src), source_peptide=f"{REF} receptor {REC_CHAIN} / peptide {PEP_CHAIN}",
            superposition=f"3LNZ:{REC_CHAIN} CA 27-108 onto 3EQS A CA ({len(keys)} atoms)",
            peptide_sequence=s.seq("B"), n_atoms=len(s.name), receptor_max_dev_A=round(rdev, 4),
            stereo_terms=len(meas), omega_B11_B12=omega12, backbone_terms=nt, backbone_worst=[zl, zw],
            closest_nonbonded=f"{closest[0]:.3f}:{s.label(closest[1])}--{s.label(closest[2])}",
            min_receptor_peptide=f"{D.min():.3f}:{s.label(np.where(B)[0][i])}--{s.label(np.where(A)[0][j])}",
            failures=fails, passed=not fails, written=str(dst))
        print(json.dumps({sample: report[sample]}, ensure_ascii=False), flush=True)
    (NEW / "inputs/modeller_v2x/build_xstart_record.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    sys.exit(0 if all(r["passed"] for r in report.values()) else 1)


if __name__ == "__main__":
    main()
