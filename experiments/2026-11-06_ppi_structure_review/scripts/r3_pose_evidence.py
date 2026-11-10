#!/usr/bin/env python3
"""R3: compare the v2.1 template models with the closest experimental MDM2 complexes (read-only).

Plan: docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md §4 R3.  numpy + text only; no OpenMM, no DCD.

References (MDM2 25-109 receptor, same sequence as 3EQS), downloaded from RCSB into
r3/raw_pdb/ and parsed as text:
    3EQS  PMI        TSFAEYWNLLSP  1 complex, B12 unobserved
    3LNZ  PMI-N8A    TSFAEYWALLSP  8 complexes; Pro12 observed in 2 peptide copies
    5UMM  M3         LTFLEYWAQLMQ  2 complexes; Gln12 observed in one copy
Every peptide chain is paired with the receptor chain it contacts, superposed on 3EQS by
receptor CA 27-108 (residues observed in all three entries), then compared residue by residue.
Model stages: prepared, minimized, and every 1-ns production checkpoint of the three replicas.
Rotamer bins use the standard staggered definitions (g+ 0..120, t 120..240, g- 240..360 deg);
no acceptance thresholds.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import stage_trace as T                                          # noqa: E402

R3 = HERE.parents[1] / "r3"
RAW = R3 / "raw_pdb"
SAMPLES = {  # sample -> reference entries carrying an experimental counterpart of its changes
    "PMI": ["3EQS", "3LNZ"], "PMI-L10A": ["3EQS"], "T1L_L10A": ["5UMM"], "A4L_L10A": ["5UMM"],
    "N8A_L10A": ["3LNZ"], "S11M_L10A": ["5UMM"], "P12Q_L10A": ["5UMM"], "PMI-M4": ["5UMM"]}
CHI = {"LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
       "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
       "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
       "THR": [("N", "CA", "CB", "OG1")], "SER": [("N", "CA", "CB", "OG")],
       "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")], "PRO": [("N", "CA", "CB", "CG")]}
REC_CA = [("A", r, "CA") for r in range(27, 109)]


def read_pairs(entry: str):
    """[(label, Struct with chains renamed A=receptor, B=peptide)] for every complex in entry."""
    path = T.RAW if entry == "3EQS" else RAW / f"{entry}.pdb"
    rows = defaultdict(list)
    for ln in path.read_text().splitlines():
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith("ATOM") and ln[16] in " A":
            rows[ln[21]].append((ln[17:20].strip(), int(ln[22:26]), ln[12:16].strip(),
                                 [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    rec = {c: v for c, v in rows.items() if len({r[1] for r in v}) > 40}
    pep = {c: v for c, v in rows.items() if c not in rec}
    out = []
    for pc, pv in sorted(pep.items()):
        P = np.array([x[3] for x in pv if x[2][0] != "H"])
        best = min(rec, key=lambda rc: np.linalg.norm(
            P[:, None] - np.array([x[3] for x in rec[rc] if x[2][0] != "H"])[None], axis=2).min())
        s = T.Struct([("A", r, rn, n, xyz) for rn, r, n, xyz in rec[best]] +
                     [("B", r, rn, n, xyz) for rn, r, n, xyz in pv])
        out.append((f"{entry}:{best}/{pc}", s))
    return out


def dihedral(s, rk, names):
    if not all(rk + (n,) in s.idx for n in names):
        return None
    return T.geo.dihedral(*(s.xyz[s.idx[rk + (n,)]] for n in names)) % 360.0


def rot_bin(a):
    return None if a is None else ("g+" if a < 120 else "t" if a < 240 else "g-")


def per_residue(s: T.Struct, ref: T.Struct):
    """{resnum: dict} for peptide residues present in both (s already in the 3EQS frame)."""
    out = {}
    for r in range(1, 13):
        rk = ("B", r)
        if rk not in s.res or rk not in ref.res:
            continue
        ca = ("B", r, "CA")
        d = dict(CA=float(np.linalg.norm(s.xyz[s.idx[ca]] - ref.xyz[ref.idx[ca]])) if ca in s.idx and ca in ref.idx else None,
                 same_type=s.res[rk] == ref.res[rk])
        if d["same_type"] and s.res[rk] in CHI:
            d["chi"] = [dihedral(s, rk, c) for c in CHI[s.res[rk]]]
            d["chi_ref"] = [dihedral(ref, rk, c) for c in CHI[s.res[rk]]]
            # local side-chain RMSD after superposing that residue's N, CA, C
            bbk = [("B", r, n) for n in ("N", "CA", "C")]
            sck = [k for k in ref.idx if k[:2] == rk and k[2] not in ("N", "CA", "C", "O", "OXT") and k[2][0] != "H" and k in s.idx]
            if sck and all(k in s.idx and k in ref.idx for k in bbk):
                R, t = T.kabsch(s.xyz[[s.idx[k] for k in bbk]], ref.xyz[[ref.idx[k] for k in bbk]])
                X = s.xyz[[s.idx[k] for k in sck]] @ R.T + t
                d["sc_local_rmsd"] = float(np.sqrt(((X - ref.xyz[[ref.idx[k] for k in sck]]) ** 2).sum(1).mean()))
        out[r] = d
    return out


def frame(s):
    return T.align(s, RAW3EQS, REC_CA)[0]


RAW3EQS = T.read_pdb(T.RAW, raw=True)


def main():
    R3.mkdir(exist_ok=True)
    refs = {e: [(lab, frame(s)) for lab, s in read_pairs(e)] for e in ("3EQS", "3LNZ", "5UMM")}
    # ---------- 1. experimental references vs 3EQS and copy-to-copy spread
    ref_rows = []
    for e, lst in refs.items():
        for lab, s in lst:
            seq = s.seq("B")
            pr = per_residue(s, RAW3EQS)
            bb = [k for k in RAW3EQS.idx if k[0] == "B" and k[2] in T.BB and k in s.idx]
            P, Q = s.xyz[[s.idx[k] for k in bb]], RAW3EQS.xyz[[RAW3EQS.idx[k] for k in bb]]
            R, t = T.kabsch(P, Q)
            row = dict(reference=lab, peptide_seq=seq, n_pep_res=len([k for k in s.res if k[0] == "B"]),
                       rec_CA_rmsd_vs3eqs=round(float(np.sqrt(np.mean([np.sum((s.xyz[s.idx[k]] - RAW3EQS.xyz[RAW3EQS.idx[k]]) ** 2)
                                                                       for k in REC_CA if k in s.idx]))), 3),
                       pep_bb_rmsd_vs3eqs=round(float(np.sqrt(((P - Q) ** 2).sum(1).mean())), 3),
                       pep_bb_rmsd_selfaligned=round(float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean())), 3),
                       n_bb_atoms=len(bb))
            for r in range(1, 12):
                row[f"CA{r}"] = round(pr[r]["CA"], 2) if r in pr else None
            ref_rows.append(row)
    # copy-to-copy spread inside multi-copy entries (crystal-level variation of the same peptide)
    spread = []
    for e in ("3LNZ", "5UMM"):
        lst = refs[e]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                pr = per_residue(lst[i][1], lst[j][1])
                spread.append(dict(entry=e, a=lst[i][0], b=lst[j][0], **{f"CA{r}": round(pr[r]["CA"], 2) for r in pr}))
    # ---------- 2. models vs their closest references
    model_rows, chi_rows = [], []
    for sample, ents in SAMPLES.items():
        for model in ("v21m1", "v21m2"):
            prep_dir = T.PACKET / "prepared" / sample / model
            prepared = T.read_pdb(prep_dir / "complex_dry.pdb")
            stages = [("prepared", prepared),
                      ("minimized", T.read_state(T.PACKET / "md" / sample / model / "setup/minimized_state.xml", prepared))]
            for rep in (1, 2, 3):
                rd = T.PACKET / "md" / sample / model / f"rep{rep:02d}"
                n = json.loads((rd / "status.json").read_text())["chunks_done"]
                stages += [(f"r{rep:02d}_prod_{c + 1:02d}ns", T.read_state(rd / "prod" / f"chunk_{c:04d}_state.xml", prepared))
                           for c in range(n)]
            for name, s in stages:
                s = frame(T.image_peptide(s)[0])
                for e in ents + (["3EQS"] if "3EQS" not in ents else []):
                    for lab, ref in refs[e]:
                        pr = per_residue(s, ref)
                        row = dict(sample=sample, model=model, stage=name, reference=lab)
                        for r in range(1, 13):
                            row[f"CA{r}"] = round(pr[r]["CA"], 2) if r in pr and pr[r]["CA"] is not None else None
                        model_rows.append(row)
                        for r, d in pr.items():
                            # rotamer comparison only where the model residue differs from PMI/3EQS
                            # (a mutated or rebuilt position) and the reference has the same type
                            if "chi" in d and (e != "3EQS") and (RAW3EQS.res.get(("B", r)) != s.res[("B", r)] or r == 12):
                                chi_rows.append(dict(sample=sample, model=model, stage=name, reference=lab,
                                                     residue=f"{s.res[('B', r)]}{r}",
                                                     chi=";".join("" if a is None else f"{a:.0f}" for a in d["chi"]),
                                                     chi_ref=";".join("" if a is None else f"{a:.0f}" for a in d["chi_ref"]),
                                                     bins=";".join(str(rot_bin(a)) for a in d["chi"]),
                                                     bins_ref=";".join(str(rot_bin(a)) for a in d["chi_ref"]),
                                                     sc_local_rmsd=round(d.get("sc_local_rmsd", float("nan")), 2),
                                                     CA=round(d["CA"], 2)))
            print(sample, model, len(stages), flush=True)

    def dump(name, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with (R3 / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    dump("reference_vs_3eqs.csv", ref_rows)
    dump("reference_copy_spread.csv", spread)
    dump("model_vs_reference.csv", model_rows)
    dump("model_rotamers_vs_reference.csv", chi_rows)


if __name__ == "__main__":
    main()
