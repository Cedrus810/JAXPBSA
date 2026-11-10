#!/usr/bin/env python3
"""R0/R1 stage trace for the MDM2-PMI peptide models (read-only).

Plan: docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md §4 R0/R1.  Every number here is
recomputed from existing files; nothing in the 2026-10-11 packet is modified.

    cd /home/ruigengji/JAXPBSA
    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-11-06_ppi_structure_review/scripts/stage_trace.py \
        --samples PMI PMI-L10A P12Q_L10A --tag rep6

numpy + PDB text + OpenMM state-XML text only: no OpenMM import, no system, no
minimisation, no DCD reading.  Stages per model:

    3eqs_raw      raw PDB, ATOM records, altloc ' '/'A'          (experimental frame)
    clean         modeller_v2/<PRO|GLN>/template_3eqs_clean.pdb
    stage1        modeller_v2/<PRO|GLN>/start_v2mN.pdb           (A25 + B12 built)
    heavy         modeller_v21/<sample>/heavy_v2mN.pdb           (mutation)
    reduced       modeller_v21/<sample>/reduced_v2mN.pdb         (reduce H)
    start         modeller_v21/<sample>/start_v21mN.pdb          (finalize)
    prepared      prepared/<sample>/<model>/complex_dry.pdb      (= setup input, pre-minimisation)
    minimized     md/<sample>/<model>/setup/minimized_state.xml  (solute = first n atoms)
    rNN_nvt_end / rNN_npt_end / rNN_prod_XXns   per replica state XMLs (unwrapped positions)

Comparisons are by (chain, resnum, atom name) with the symmetric heavy-atom swaps listed in
SYM; 'H1' on chain N-termini is aliased to 'H' (OpenMM rename, v2 design §2.3).  All lengths
in Angstrom.  Regions are reported separately (plan §4 R1 rule 2): receptor observed 26-109,
A25 (built), peptide 1-11 backbone / side chain vs 3EQS, B12 (built, vs the model's own
stage-1 coordinates), mutated side chains.  Bond/angle z-scores use the ff14SB sigma of
peptide_geometry.py (C1-C4 definition); contact cuts are the C2 ones.  Nothing is accepted or
rejected here.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
RAW = ROOT / "experiments/2026-10-10_peptide_samples/raw/mdm2_3EQS.pdb"
OUT = HERE.parents[1] / "r0r1"
sys.path.insert(0, str(ROOT / "scripts"))
import peptide_geometry as geo                                    # noqa: E402

AA1 = dict(ALA="A", CYS="C", ASP="D", GLU="E", PHE="F", GLY="G", HIS="H", HIE="H", HID="H", HIP="H",
           ILE="I", LYS="K", LEU="L", MET="M", ASN="N", PRO="P", GLN="Q", ARG="R", SER="S", THR="T",
           VAL="V", TRP="W", TYR="Y")
SYM = {"PHE": [("CD1", "CD2"), ("CE1", "CE2")], "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
       "ASP": [("OD1", "OD2")], "GLU": [("OE1", "OE2")], "ARG": [("NH1", "NH2")]}
BB = ("N", "CA", "C", "O")
CB_CHIRAL = {"THR": ("CA", "OG1", "CG2"), "ILE": ("CA", "CG1", "CG2")}
PDB_PREC = 0.001          # PDB coordinate precision (C1 anchor rule)
FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


# ------------------------------------------------------------------ structures
class Struct:
    def __init__(self, rows, box=None, forces=None):
        """rows: [(chain, resnum, resname, name, xyz)] in file order."""
        self.chain = [r[0] for r in rows]
        self.resnum = [int(r[1]) for r in rows]
        self.resname = [r[2] for r in rows]
        self.name = ["H" if r[3] == "H1" else r[3] for r in rows]
        self.xyz = np.array([r[4] for r in rows], dtype=float)
        self.box = box
        self.forces = forces
        self.idx = {}
        for i, k in enumerate(zip(self.chain, self.resnum, self.name)):
            self.idx[k] = i
        self.is_h = np.array([n[0] == "H" for n in self.name])
        self.res = {}
        for i, k in enumerate(zip(self.chain, self.resnum)):
            self.res.setdefault(k, self.resname[i])

    def label(self, i):
        return f"{self.chain[i]}:{self.resname[i]}{self.resnum[i]}.{self.name[i]}"

    def with_xyz(self, xyz, box=None, forces=None):
        s = object.__new__(Struct)
        s.__dict__.update(self.__dict__)
        s.xyz, s.box, s.forces = np.asarray(xyz, float), box, forces
        return s

    def seq(self, ch):
        return "".join(AA1.get(self.res[k], "X") for k in sorted(k for k in self.res if k[0] == ch))

    def geo_residues(self):
        out, key = [], None
        for i in range(len(self.name)):
            k = (self.chain[i], self.resnum[i])
            if k != key:
                out.append((k[0], self.resname[i], str(k[1]), {}))
                key = k
            out[-1][3][self.name[i]] = i
        return out


def read_pdb(path: Path, raw=False) -> Struct:
    rows = []
    for ln in path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if raw and (not ln.startswith("ATOM") or ln[16] not in " A" or ln[21] not in "AB"):
            continue
        rows.append((ln[21], int(ln[22:26]), ln[17:20].strip(), ln[12:16].strip(),
                     [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    return Struct(rows)


def read_state(path: Path, ref: Struct, want_forces=False) -> Struct:
    """Solute positions (first len(ref) atoms) of an OpenMM State XML, nm -> A."""
    n = len(ref.name)
    box, pos, frc, mode = [], [], [], None
    pat = re.compile(rf'x="({FLOAT})" y="({FLOAT})" z="({FLOAT})"')
    with path.open() as fh:
        for ln in fh:
            s = ln.strip()
            if s.startswith("<PeriodicBoxVectors"):
                mode = "box"
            elif s.startswith("<Positions"):
                mode = "pos"
            elif s.startswith("<Forces"):
                mode = "frc"
            elif s.startswith("</"):
                if mode == "pos" and not want_forces:
                    break
                mode = None
            elif mode == "box" and s[:3] in ("<A ", "<B ", "<C "):
                box.append([float(v) for v in pat.search(s).groups()])
            elif mode == "pos" and len(pos) < n:
                pos.append([float(v) for v in pat.search(s).groups()])
                if len(pos) == n and not want_forces:
                    break
            elif mode == "frc" and len(frc) < n:
                frc.append([float(v) for v in pat.search(s).groups()])
                if len(frc) == n:
                    break
    if len(pos) != n:
        raise RuntimeError(f"{path}: {len(pos)} positions < {n}")
    box = np.array(box) * 10.0
    if np.abs(box - np.diag(np.diag(box))).max() > 1e-6:
        raise RuntimeError(f"{path}: non-rectangular box")
    return ref.with_xyz(np.array(pos) * 10.0, box=np.diag(box),
                        forces=np.array(frc) if want_forces else None)


# ------------------------------------------------------------------ helpers
def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def rec_ca_keys(s: Struct, lo=26, hi=109):
    return [("A", r, "CA") for r in range(lo, hi + 1) if ("A", r, "CA") in s.idx]


def align(s: Struct, ref: Struct, keys):
    keys = [k for k in keys if k in s.idx and k in ref.idx]
    R, t = kabsch(s.xyz[[s.idx[k] for k in keys]], ref.xyz[[ref.idx[k] for k in keys]])
    return s.with_xyz(s.xyz @ R.T + t, box=s.box, forces=s.forces), len(keys)


def image_peptide(s: Struct):
    """Shift chain B as a whole by the lattice vector that puts its centroid nearest to chain A's
    centroid (unwrapped state positions keep molecules whole).  Returns (struct, integer shift)."""
    if s.box is None:
        return s, (0, 0, 0)
    A = np.array([c == "A" for c in s.chain])
    d = s.xyz[~A].mean(0) - s.xyz[A].mean(0)
    n = np.round(d / s.box)
    xyz = s.xyz.copy()
    xyz[~A] -= n * s.box
    return s.with_xyz(xyz, box=s.box, forces=s.forces), tuple(int(x) for x in n)


def paired_dev(s: Struct, ref: Struct, keys):
    """Per-key distance with the symmetric swaps of SYM applied per residue when they lower the
    residue's summed deviation.  Returns ({key: d}, [swapped residues])."""
    keys = [k for k in keys if k in s.idx and k in ref.idx]
    out, swapped = {}, []
    byres = {}
    for k in keys:
        byres.setdefault(k[:2], []).append(k)
    for rk, ks in byres.items():
        d0 = {k: float(np.linalg.norm(s.xyz[s.idx[k]] - ref.xyz[ref.idx[k]])) for k in ks}
        groups = SYM.get(s.res.get(rk), [])
        names = {k[2] for k in ks}
        if groups and all(a in names and b in names for a, b in groups):
            sw = dict(d0)
            for a, b in groups:
                ka, kb = rk + (a,), rk + (b,)
                sw[ka] = float(np.linalg.norm(s.xyz[s.idx[ka]] - ref.xyz[ref.idx[kb]]))
                sw[kb] = float(np.linalg.norm(s.xyz[s.idx[kb]] - ref.xyz[ref.idx[ka]]))
            if sum(sw.values()) < sum(d0.values()) - 1e-9:
                d0 = sw
                swapped.append(f"{rk[0]}:{s.res[rk]}{rk[1]}")
        out.update(d0)
    return out, swapped


def rmsd_max(dev: dict, s: Struct):
    if not dev:
        return None, None, None
    v = np.array(list(dev.values()))
    k = max(dev, key=dev.get)
    return round(float(np.sqrt((v ** 2).mean())), 3), round(float(v.max()), 3), s.label(s.idx[k])


def bonds_from(s: Struct):
    """Covalent bonds by distance (1.2 x sum of covalent radii, as finalize_allh_models.py),
    returned as atom-key pairs so they can be evaluated on any stage."""
    rad = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
    el = np.array([rad.get(n[0], 0.76) for n in s.name])
    keys = list(zip(s.chain, s.resnum, s.name))
    out = []
    for i in range(len(keys)):
        d = np.linalg.norm(s.xyz[i + 1:] - s.xyz[i], axis=1)
        for j in np.where(d < 1.2 * (el[i] + el[i + 1:]))[0]:
            out.append((keys[i], keys[i + 1 + int(j)]))
    return out


def sigma(kind, table):
    x0, k = table[kind]
    return x0, np.sqrt(geo.KT_298 / k)


def backbone_z(s: Struct):
    """Every backbone bond/angle of the C1-C4 table on every residue: count |z| > 4 and worst."""
    terms = []
    g = s.geo_residues()
    for i, (c, rn, r, at) in enumerate(g):
        if all(x in at for x in ("N", "CA", "C")):
            x0, sg = sigma("N-CA", geo.FF14SB_BOND)
            terms.append((f"{c}{r} N-CA", np.linalg.norm(s.xyz[at["N"]] - s.xyz[at["CA"]]) / 10, x0, sg))
            x0, sg = sigma("N-CA-C", geo.FF14SB_ANGLE)
            terms.append((f"{c}{r} N-CA-C", np.radians(geo._angle_deg(*(s.xyz[at[k]] for k in ("N", "CA", "C")))), x0, sg))
        if i + 1 < len(g) and g[i + 1][0] == c:
            b = g[i + 1][3]
            if all(x in at for x in ("CA", "C", "O")) and all(x in b for x in ("N", "CA")):
                tag = f"{c}{r}-{c}{g[i + 1][2]}"
                x0, sg = sigma("C-N", geo.FF14SB_BOND)
                terms.append((f"{tag} C-N", np.linalg.norm(s.xyz[at["C"]] - s.xyz[b["N"]]) / 10, x0, sg))
                for nm, (p, q, w) in {"CA-C-N": (at["CA"], at["C"], b["N"]), "O-C-N": (at["O"], at["C"], b["N"]),
                                      "C-N-CA": (at["C"], b["N"], b["CA"])}.items():
                    x0, sg = sigma(nm, geo.FF14SB_ANGLE)
                    terms.append((f"{tag} {nm}", np.radians(geo._angle_deg(s.xyz[p], s.xyz[q], s.xyz[w])), x0, sg))
    z = [(abs(v - x0) / sg, lab) for lab, v, x0, sg in terms]
    worst = max(z)
    return len(terms), sum(1 for zz, _ in z if zz > geo.NSIG), round(float(worst[0]), 2), worst[1]


def cb_chirality(s: Struct):
    """Signed volume (CA-CB).[(X1-CB)x(X2-CB)] for Thr (X=OG1,CG2) and Ile (X=CG1,CG2)."""
    out = {}
    for rk, rn in s.res.items():
        if rn in CB_CHIRAL and all(rk + (n,) in s.idx for n in ("CB",) + CB_CHIRAL[rn]):
            cb = s.xyz[s.idx[rk + ("CB",)]]
            a, b, c = (s.xyz[s.idx[rk + (n,)]] - cb for n in CB_CHIRAL[rn])
            out[f"{rk[0]}:{rn}{rk[1]}"] = float(a @ np.cross(b, c))
    return out


# ------------------------------------------------------------------ per-stage metrics
def stage_metrics(s: Struct, raw: Struct, built: Struct, prepared: Struct, bonds, mutated, cb_ref_sign):
    m = {"n_atoms": len(s.name), "n_H": int(s.is_h.sum()), "seq_A": s.seq("A"), "seq_B": s.seq("B")}
    s, n_al = align(s, raw, rec_ca_keys(raw))
    m["align_receptor_CA_n"] = n_al
    heavy = lambda st, f: [k for k in zip(st.chain, st.resnum, st.name) if k[2][0] != "H" and f(k)]
    # receptor observed region vs 3EQS
    dev, sw = paired_dev(s, raw, heavy(raw, lambda k: k[0] == "A"))
    m["rec_obs_heavy_rmsd_vs3eqs"], m["rec_obs_heavy_max_vs3eqs"], m["rec_obs_heavy_max_atom"] = rmsd_max(dev, s)
    m["rec_obs_n_compared"] = len(dev)
    ca = {k: v for k, v in dev.items() if k[2] == "CA"}
    m["rec_CA_rmsd_vs3eqs"] = rmsd_max(ca, s)[0]
    m["sym_swaps_vs3eqs"] = ";".join(sw)
    # peptide 1-11 vs 3EQS: backbone; CB; side chain beyond CB where residue type unchanged
    pep_obs = [k for k in heavy(raw, lambda k: k[0] == "B")]
    bb = {k: v for k, v in paired_dev(s, raw, [k for k in pep_obs if k[2] in BB])[0].items()}
    m["pep1_11_bb_rmsd_vs3eqs"], m["pep1_11_bb_max_vs3eqs"], m["pep1_11_bb_max_atom"] = rmsd_max(bb, s)
    # same atoms after superposing the peptide backbone itself: internal change vs rigid shift
    bk = [k for k in pep_obs if k[2] in BB and k in s.idx]
    P, Q = s.xyz[[s.idx[k] for k in bk]], raw.xyz[[raw.idx[k] for k in bk]]
    R, t = kabsch(P, Q)
    m["pep1_11_bb_rmsd_selfaligned"] = round(float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean())), 3)
    m["pep1_11_centroid_shift"] = round(float(np.linalg.norm(P.mean(0) - Q.mean(0))), 3)
    same = [k for k in pep_obs if k[2] not in BB and s.res.get(k[:2]) == raw.res.get(k[:2])]
    cb_keys = [k for k in pep_obs if k[2] == "CB"]
    scd, _ = paired_dev(s, raw, same + [k for k in cb_keys if k not in same])
    m["pep1_11_sc_rmsd_vs3eqs"], m["pep1_11_sc_max_vs3eqs"], m["pep1_11_sc_max_atom"] = rmsd_max(scd, s)
    m["pep1_11_sc_n_compared"] = len(scd)
    # built regions vs the model's own stage-1 coordinates (no experimental reference)
    for tag, sel in (("A25", lambda k: k[:2] == ("A", 25)), ("B12", lambda k: k[:2] == ("B", 12))):
        d, _ = paired_dev(s, built, heavy(built, sel))
        m[f"{tag}_heavy_rmsd_vs_stage1"], m[f"{tag}_heavy_max_vs_stage1"], _a = rmsd_max(d, s)
        ca_key = (tag[0], int(tag[1:]), "CA")
        m[f"{tag}_CA_shift_vs_stage1"] = round(d[ca_key], 3) if ca_key in d else None
    # mutated side chains vs the model's own heavy (mutation) stage are filled by caller
    # interface
    A = np.array([c == "A" for c in s.chain]) & ~s.is_h
    B = np.array([c == "B" for c in s.chain]) & ~s.is_h
    D = np.linalg.norm(s.xyz[B][:, None] - s.xyz[A][None], axis=2)
    m["min_rec_pep_heavy"] = round(float(D.min()), 3)
    # diagnostics.csv definition (strip_peptide_replica.py): receptor CA (all) -> prepared, peptide CA
    if prepared is not None:
        rk = [k for k in zip(prepared.chain, prepared.resnum, prepared.name) if k[0] == "A" and k[2] == "CA"]
        sp, _ = align(s, prepared, rk)
        pk = [k for k in zip(prepared.chain, prepared.resnum, prepared.name) if k[0] == "B" and k[2] == "CA"]
        d = np.array([np.linalg.norm(sp.xyz[sp.idx[k]] - prepared.xyz[prepared.idx[k]]) for k in pk])
        m["diagdef_pep_CA_rmsd_vs_prepared"] = round(float(np.sqrt((d ** 2).mean())), 3)
        m["diagdef_CA12_shift_vs_prepared"] = round(float(d[-1]), 3)
        m["pep_CA1_11_rmsd_vs_prepared"] = round(float(np.sqrt((d[:-1] ** 2).mean())), 3)
    # stereo
    g = s.geo_residues()
    meas = geo.measure(geo.stereo_terms(g), s.xyz)
    m["n_D_CA"] = sum(1 for x in meas if x["kind"] == "chirality" and x["class"] == "D")
    cis = [x["label"] for x in meas if x["kind"] == "omega" and x["class"] == "cis"]
    m["cis_omega"] = ";".join(cis)
    m["D_CA"] = ";".join(x["label"] for x in meas if x["kind"] == "chirality" and x["class"] == "D")
    w = {x["label"]: round(x["deg"], 1) for x in meas if x["kind"] == "omega"}
    m["omega_B11_B12"] = next((v for k, v in w.items() if k.startswith("B:") and k.endswith("12")), None)
    m["omega_A25_A26"] = next((v for k, v in w.items() if k.startswith("A:") and "25-" in k), None)
    cb = cb_chirality(s)
    m["cbchiral_opposite_to_3eqs"] = ";".join(k for k, v in cb.items() if np.sign(v) != cb_ref_sign[k.split(":")[1][:3]])
    if s.is_h.any():
        ha = geo.ha_direction(g, s.xyz)
        worst = max(ha, key=lambda x: x[1])
        m["HA_worst_deg"], m["HA_worst_res"] = round(worst[1], 1), worst[0]
    nt, nout, zw, zl = backbone_z(s)
    m["bb_terms"], m["bb_terms_gt4sigma"], m["bb_worst_z"], m["bb_worst_term"] = nt, nout, zw, zl
    # bonded continuity and contacts on the reference bond graph
    bl = [(a, b) for a, b in bonds if a in s.idx and b in s.idx]
    L = np.array([np.linalg.norm(s.xyz[s.idx[a]] - s.xyz[s.idx[b]]) for a, b in bl])
    j = int(L.argmax())
    m["max_bond_len"], m["max_bond"] = round(float(L[j]), 3), f"{s.label(s.idx[bl[j][0]])}-{bl[j][1][2]}"
    pairs = [(s.idx[a], s.idx[b]) for a, b in bl]
    clashes, closest = geo.close_contacts(s.xyz, pairs, range(len(s.name)), list(s.is_h))
    m["n_clash_pairs"] = len(clashes)
    m["clashes"] = ";".join(f"{d}:{s.label(i)}--{s.label(j)}" for d, i, j in clashes[:6])
    m["closest_nonbonded"] = f"{closest[0]:.3f}:{s.label(closest[1])}--{s.label(closest[2])}"
    return m, s


def residue_rows(s_al: Struct, raw: Struct, built: Struct, prepared_al: Struct | None):
    """Per peptide residue (and A25): deviations and receptor distance in the 3EQS frame."""
    rows = []
    A = np.array([c == "A" for c in s_al.chain]) & ~s_al.is_h
    for rk in sorted(k for k in s_al.res if k[0] == "B") + [("A", 25)]:
        ks = [k for k in s_al.idx if k[:2] == rk and k[2][0] != "H"]
        if not ks:
            continue
        own = np.array([s_al.idx[k] for k in ks])
        dmin = np.linalg.norm(s_al.xyz[own][:, None] - s_al.xyz[A][None], axis=2)
        ref, refname = (raw, "3eqs") if rk in raw.res else (built, "stage1")
        bb, _ = paired_dev(s_al, ref, [k for k in ks if k[2] in BB])
        sc, _ = paired_dev(s_al, ref, [k for k in ks if k[2] not in BB and (s_al.res[rk] == ref.res.get(rk) or k[2] == "CB")])
        rows.append(dict(residue=f"{rk[0]}:{s_al.res[rk]}{rk[1]}", ref=refname,
                         bb_max=rmsd_max(bb, s_al)[1], sc_max=rmsd_max(sc, s_al)[1], sc_max_atom=rmsd_max(sc, s_al)[2],
                         CA_dev=round(bb[rk + ("CA",)], 3) if rk + ("CA",) in bb else None,
                         min_dist_receptor=round(float(dmin.min()), 3) if rk[0] == "B" else None))
    return rows


def pdb_lines(s: Struct, model: int, remark: str):
    out = [f"MODEL     {model:4d}", f"REMARK   1 {remark}"]
    for i in range(len(s.name)):
        n = s.name[i]
        nm = f" {n:<3s}" if len(n) < 4 else n
        x, y, z = s.xyz[i]
        out.append(f"ATOM  {i + 1:5d} {nm} {s.resname[i]:>3s} {s.chain[i]}{s.resnum[i]:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {n[0]:>2s}")
    out.append("ENDMDL")
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["v21m1", "v21m2"])
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", required=True, help="suffix of the output tables, e.g. rep6 / rest10")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "views").mkdir(exist_ok=True)

    raw = read_pdb(RAW, raw=True)
    cb_ref = {}
    for k, v in cb_chirality(raw).items():
        cb_ref.setdefault(k.split(":")[1][:3], set()).add(int(np.sign(v)))
    if any(len(v) != 1 for v in cb_ref.values()):
        raise RuntimeError(f"3EQS Thr/Ile CB chirality not uniform: {cb_ref}")
    cb_ref = {k: v.pop() for k, v in cb_ref.items()}

    geom, resid, amap, xcheck = [], [], [], []
    for sample in a.samples:
        brec = json.loads((PACKET / "inputs/modeller_v21" / sample / "build_record.json").read_text())
        for model in a.models:
            sid = model.replace("v21", "v2")
            hrec = brec["models"][f"heavy_{sid}.pdb"]
            st1 = Path(hrec["start"])
            mutated = {("B", int(p)) for p in hrec["mutated_peptide_positions"]}
            d21 = PACKET / "inputs/modeller_v21" / sample
            prep_dir = PACKET / "prepared" / sample / model
            setup = PACKET / "md" / sample / model / "setup"
            stages = [("3eqs_raw", raw, str(RAW)),
                      ("clean", read_pdb(st1.parent / "template_3eqs_clean.pdb"), str(st1.parent / "template_3eqs_clean.pdb")),
                      ("stage1", read_pdb(st1), str(st1)),
                      ("heavy", read_pdb(d21 / f"heavy_{sid}.pdb"), str(d21 / f"heavy_{sid}.pdb")),
                      ("reduced", read_pdb(d21 / f"reduced_{sid}.pdb"), str(d21 / f"reduced_{sid}.pdb")),
                      ("start", read_pdb(d21 / f"start_{model}.pdb"), str(d21 / f"start_{model}.pdb")),
                      ("prepared", read_pdb(prep_dir / "complex_dry.pdb"), str(prep_dir / "complex_dry.pdb"))]
            prepared = stages[-1][1]
            built = stages[2][1]
            heavy_stage = stages[3][1]
            npz = np.load(prep_dir / "parameters.npz")
            npz_ok = (list(npz["atom_name"]) == [l[12:16].strip() for l in (prep_dir / "complex_dry.pdb").read_text().splitlines()
                                                 if l.startswith("ATOM")]
                      and float(np.abs(npz["coords_A"] - prepared.xyz).max()) <= PDB_PREC
                      and set(npz["receptor_idx"].tolist()) == {i for i, c in enumerate(prepared.chain) if c == "A"}
                      and set(npz["ligand_idx"].tolist()) == {i for i, c in enumerate(prepared.chain) if c == "B"})
            mini = read_state(setup / "minimized_state.xml", prepared, want_forces=True)
            stages.append(("minimized", mini, str(setup / "minimized_state.xml")))
            for r in a.reps:
                rd = PACKET / "md" / sample / model / f"rep{r:02d}"
                stat = json.loads((rd / "status.json").read_text())
                stages.append((f"r{r:02d}_nvt_end", read_state(rd / "nvt_end.xml", prepared), str(rd / "nvt_end.xml")))
                stages.append((f"r{r:02d}_npt_end", read_state(rd / "npt_end.xml", prepared), str(rd / "npt_end.xml")))
                for c in range(stat["chunks_done"]):
                    p = rd / "prod" / f"chunk_{c:04d}_state.xml"
                    stages.append((f"r{r:02d}_prod_{c + 1:02d}ns", read_state(p, prepared), str(p)))
            # bond graph: prepared (all H) united with stage-1 (pre-mutation heavy atoms), so a
            # side chain removed/added by the mutation step is not read as a non-bonded contact
            bonds = sorted(set(bonds_from(prepared)) | set(bonds_from(built)))
            view, prev = [], None
            diag_cache = {}
            for si, (name, s, src) in enumerate(stages):
                s_img, shift = image_peptide(s)
                m, s_al = stage_metrics(s_img, raw, built, prepared if si >= 6 else None, bonds, mutated, cb_ref)
                # mutated side chains (beyond CB) vs the mutation stage
                mk = [k for k in zip(heavy_stage.chain, heavy_stage.resnum, heavy_stage.name)
                      if k[:2] in mutated and k[2] not in BB + ("CB",) and k[2][0] != "H"]
                if mk:
                    hs_al = heavy_stage
                    dmut, _ = paired_dev(s_al, hs_al, mk)
                    m["mut_sc_rmsd_vs_heavy"], m["mut_sc_max_vs_heavy"], m["mut_sc_max_atom"] = rmsd_max(dmut, s_al)
                m = {"sample": sample, "model": model, "stage": name, "source": src,
                     "pbc_peptide_shift": "" if s.box is None else "%d,%d,%d" % shift, **m}
                if name == "minimized":
                    f = np.linalg.norm(s.forces, axis=1) * 1.0
                    top = np.argsort(f)[::-1][:5]
                    m["minimized_top_solute_forces_kJ_mol_nm"] = ";".join(f"{f[i]:.1f}:{s.label(i)}" for i in top)
                geom.append(m)
                for rr in residue_rows(s_al, raw, built, None):
                    resid.append({"sample": sample, "model": model, "stage": name, **rr})
                # stage map vs previous static stage (same frame for build stages; aligned after)
                if prev is not None and si <= 7:
                    pn, ps = prev
                    keys_s = set(zip(s.chain, s.resnum, s.name))
                    keys_p = set(zip(ps.chain, ps.resnum, ps.name))
                    common = sorted(keys_s & keys_p)
                    cmp_s = s if si < 7 else align(s, ps, [k for k in common if k[0] == "A" and k[2] == "CA"])[0]
                    dv, sw = paired_dev(cmp_s, ps, [k for k in common if k[2][0] != "H"])
                    dh, _ = paired_dev(cmp_s, ps, [k for k in common if k[2][0] == "H"])
                    moved = sorted({f"{k[0]}:{s.res[k[:2]]}{k[1]}" for k, v in dv.items() if v > PDB_PREC},
                                   key=lambda x: (x[0], int(re.sub(r"\D", "", x))))
                    add, rem = sorted(keys_s - keys_p), sorted(keys_p - keys_s)
                    resn = sorted({f"{k[0]}{k[1]}:{ps.res[k]}->{s.res[k]}" for k in set(s.res) & set(ps.res) if ps.res[k] != s.res[k]})
                    amap.append(dict(sample=sample, model=model, from_stage=pn, to_stage=name,
                                     mapping="(chain,resnum,name); H1->H alias; symmetric heavy swaps " + "/".join(SYM),
                                     superposition="none (same frame)" if si < 7 else "receptor CA onto previous stage",
                                     n_from=len(keys_p), n_to=len(keys_s), n_common=len(common),
                                     n_added=len(add), n_removed=len(rem),
                                     added_heavy=";".join(f"{k[0]}{k[1]}.{k[2]}" for k in add if k[2][0] != "H")[:400],
                                     removed_heavy=";".join(f"{k[0]}{k[1]}.{k[2]}" for k in rem if k[2][0] != "H")[:400],
                                     n_added_H=sum(1 for k in add if k[2][0] == "H"),
                                     n_removed_H=sum(1 for k in rem if k[2][0] == "H"),
                                     resname_changes=";".join(resn),
                                     sym_swaps=";".join(sw),
                                     heavy_max_dev=rmsd_max(dv, cmp_s)[1], heavy_max_atom=rmsd_max(dv, cmp_s)[2],
                                     heavy_rmsd=rmsd_max(dv, cmp_s)[0],
                                     H_max_dev=rmsd_max(dh, cmp_s)[1], H_max_atom=rmsd_max(dh, cmp_s)[2],
                                     residues_heavy_moved_gt_0p001=";".join(moved)[:600],
                                     n_residues_heavy_moved=len(moved),
                                     npz_matches_prepared=npz_ok if name == "prepared" else ""))
                if si <= 7:
                    prev = (name, s)
                # cross-check against diagnostics.csv (end-of-chunk frame == chunk state)
                mm = re.match(r"r(\d+)_prod_(\d+)ns", name)
                if mm:
                    r, c = int(mm.group(1)), int(mm.group(2)) - 1
                    if r not in diag_cache:
                        dp = PACKET / "md" / sample / model / f"rep{r:02d}" / "analysis/diagnostics.csv"
                        diag_cache[r] = {int(row["frame"]): row for row in csv.DictReader(dp.open())}
                    row = diag_cache[r].get(c * 100 + 99)
                    if row:
                        xcheck.append(dict(sample=sample, model=model, replica=r, chunk=c, frame=c * 100 + 99,
                                           diag_pep_CA_rmsd=round(float(row["peptide_CA_rmsd_receptor_frame_A"]), 3),
                                           state_pep_CA_rmsd=m.get("diagdef_pep_CA_rmsd_vs_prepared"),
                                           diag_CA12_shift=round(float(row["peptide_CA12_shift_A"]), 3),
                                           state_CA12_shift=m.get("diagdef_CA12_shift_vs_prepared"),
                                           diag_min_rec_pep=round(float(row["min_rec_pep_heavy_A"]), 3),
                                           state_min_rec_pep=m["min_rec_pep_heavy"],
                                           pbc_peptide_shift=m["pbc_peptide_shift"]))
                if name in ("3eqs_raw", "stage1", "heavy", "start", "prepared", "minimized") or \
                        re.match(r"r\d+_(npt_end|prod_(10|20)ns)$", name):
                    view += pdb_lines(s_al, len([x for x in view if x.startswith("MODEL")]) + 1,
                                      f"{sample} {model} {name} aligned on receptor CA 26-109 to 3EQS; {src}")
            (a.out / "views" / f"{sample}_{model}_stages.pdb").write_text("\n".join(view) + "\nEND\n")
            print(f"{sample}/{model}: {len(stages)} stages", flush=True)

    def dump(path, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    tag = a.tag
    dump(a.out / f"stage_geometry_{tag}.csv", geom)
    dump(a.out / f"stage_residue_{tag}.csv", resid)
    dump(a.out / f"stage_atom_map_{tag}.csv", amap)
    dump(a.out / f"diag_crosscheck_{tag}.csv", xcheck)
    (a.out / f"run_{tag}.json").write_text(json.dumps(dict(
        samples=a.samples, models=a.models, reps=a.reps, raw=str(RAW), packet=str(PACKET),
        cb_reference_sign_3eqs=cb_ref, numpy=np.__version__, python=sys.version.split()[0]), indent=2) + "\n")


if __name__ == "__main__":
    main()
