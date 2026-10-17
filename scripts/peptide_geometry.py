"""Backbone stereochemistry bookkeeping shared by the packet prep/MD entries.

numpy only, so it can be checked on existing coordinate files without OpenMM.
Residues are given as [(chain_id, res_name, res_id, {atom_name: atom_index})]
in sequence order; coordinates are any length unit.

Two dihedral classes:
  omega    CA(i)-C(i)-N(i+1)-CA(i+1) within a chain; trans if |omega| >= 90 deg
  chirality N-CA-C-CB improper per non-Gly residue; L-residues are negative
           (3EQS crystal residues measure -110..-131 deg in this convention)
The 90 deg / sign boundaries are the class definitions themselves (cis vs trans,
L vs D), not tuned tolerances.
"""
from __future__ import annotations

import numpy as np


def dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - b0.dot(b1) * b1
    w = b2 - b2.dot(b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v).dot(w), v.dot(w))))


def stereo_terms(residues) -> list[dict]:
    """Every omega and CA-chirality dihedral as {kind, label, atoms}."""
    terms = []
    for i, (ch, rn, rid, at) in enumerate(residues):
        if all(k in at for k in ("N", "CA", "C", "CB")):
            terms.append(dict(kind="chirality", label=f"{ch}:{rn}{rid}",
                              atoms=[at["N"], at["CA"], at["C"], at["CB"]]))
        if i + 1 < len(residues) and residues[i + 1][0] == ch:
            ch2, rn2, rid2, at2 = residues[i + 1]
            if all(k in at for k in ("CA", "C")) and all(k in at2 for k in ("N", "CA")):
                terms.append(dict(kind="omega", label=f"{ch}:{rn}{rid}-{rn2}{rid2}",
                                  atoms=[at["CA"], at["C"], at2["N"], at2["CA"]],
                                  next_is_pro=(rn2 == "PRO")))
    return terms


def measure(terms, xyz) -> list[dict]:
    xyz = np.asarray(xyz, dtype=float)
    out = []
    for t in terms:
        d = dihedral(*(xyz[i] for i in t["atoms"]))
        cls = ("trans" if abs(d) >= 90 else "cis") if t["kind"] == "omega" else ("L" if d < 0 else "D")
        out.append({**t, "deg": d, "class": cls})
    return out


def prepared_violations(measured) -> list[dict]:
    """Prepared model: every CA must be L, every non-Pro peptide bond trans."""
    return [m for m in measured
            if (m["kind"] == "chirality" and m["class"] != "L")
            or (m["kind"] == "omega" and m["class"] != "trans" and not m["next_is_pro"])]


def class_changes(before, after) -> list[dict]:
    """Terms whose cis/trans or L/D class differs between two measurements."""
    return [dict(label=b["label"], kind=b["kind"], before=round(b["deg"], 1), after=round(a["deg"], 1))
            for b, a in zip(before, after) if b["class"] != a["class"]]


def mirror_side_chain(xyz, n, ca, c, side):
    """Reflect atoms `side` through the N-CA-C plane of one residue.

    A reflection is an isometry, so every bond length and angle inside the side
    chain (and to N/CA) is kept; only the CA improper changes sign (D <-> L).
    Valid only for side chains without their own stereocentre (not Thr/Ile).
    """
    xyz = np.array(xyz, dtype=float)
    p0 = xyz[n]
    normal = np.cross(xyz[ca] - p0, xyz[c] - p0)
    normal /= np.linalg.norm(normal)
    side = list(side)
    xyz[side] -= 2.0 * ((xyz[side] - p0) @ normal)[:, None] * normal
    return xyz


def residues_from_pdb(path, n_atoms=None):
    """(residues, xyz) from PDB text; atoms numbered in file order."""
    residues, xyz, key = [], [], None
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if n_atoms is not None and len(xyz) >= n_atoms:
            break
        k = (ln[21], ln[17:20].strip(), ln[22:27].strip())
        if k != key:
            residues.append((k[0], k[1], k[2], {}))
            key = k
        residues[-1][3][ln[12:16].strip()] = len(xyz)
        xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return residues, np.array(xyz)


def residues_from_topology(topology, n_atoms=None):
    """Residue list from an OpenMM Topology (atoms with index < n_atoms)."""
    residues = []
    for r in topology.residues():
        at = {a.name: a.index for a in r.atoms() if n_atoms is None or a.index < n_atoms}
        if at:
            residues.append((r.chain.id, r.name, r.id, at))
    return residues


# ------------------------------------------------------------------ prep protocol v2 checks
# docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §5.  Limits are NSIG thermal standard
# deviations of the ff14SB harmonic term at 298.15 K, sigma = sqrt(kT / k); they catch
# gross build errors (v1 had C-N-CA 77 deg, O-C-N 31 deg), not convergence.
KT_298 = 2.479          # kJ/mol
NSIG = 4.0
# (x0, k) from openmm_dev amber14/protein.ff14SB.xml; bonds nm & kJ/mol/nm^2, angles rad & kJ/mol/rad^2
FF14SB_BOND = {"C-N": (0.1335, 410032.0), "N-CA": (0.1449, 282001.6), "N-CD": (0.1449, 282001.6),
               "CT-CT": (0.1526, 259408.0)}
FF14SB_ANGLE = {"CA-C-N": (2.035053907825388, 585.76), "O-C-N": (2.1450096507010312, 669.44),
                "C-N-CA": (2.1275563581810877, 418.40), "C-N-CD": (2.1275563581810877, 418.40),
                "N-CA-C": (1.9216075064457567, 527.184), "CD-N-CA": (2.059488517353309, 418.40),
                "N-CA-CB(Pro)": (1.9146261894377796, 669.44), "CA-CB-CG(Pro)": (1.911135530933791, 334.72),
                "CB-CG-CD(Pro)": (1.911135530933791, 334.72), "CG-CD-N(Pro)": (1.9146261894377796, 669.44)}
HEAVY_CLASH_A = 2.2     # below the shortest physical heavy-atom contact (short O...O H-bond ~2.4 A)
H_CLASH_A = 1.0         # below the shortest physical H...acceptor distance (~1.5 A)


def _angle_deg(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0))))


def junction_report(residues, xyz_A, junctions, nsig=NSIG):
    """Peptide-junction and Pro-ring geometry against the ff14SB table.

    residues: [(chain, resname, resid, {name: index})]; junctions: [((c1, r1), (c2, r2))]
    with (c1, r1) bonded through its C to the N of (c2, r2).  Hydrogens are ignored.
    Returns (values, failures).
    """
    xyz_A = np.asarray(xyz_A, dtype=float)
    by = {(c, int(r)): (n, at) for c, n, r, at in residues}
    vals, fails = {}, []

    def bond(label, i, j, kind):
        x0, k = FF14SB_BOND[kind]
        d = float(np.linalg.norm(xyz_A[i] - xyz_A[j]))
        lim = nsig * np.sqrt(KT_298 / k) * 10.0
        vals[label] = round(d, 3)
        if abs(d - x0 * 10.0) > lim:
            fails.append(f"{label} {d:.3f} A (ff14SB {x0 * 10:.3f} +- {lim:.3f})")

    def ang(label, i, j, l, kind):
        x0, k = FF14SB_ANGLE[kind]
        th = _angle_deg(xyz_A[i], xyz_A[j], xyz_A[l])
        lim = float(np.degrees(nsig * np.sqrt(KT_298 / k)))
        vals[label] = round(th, 1)
        if abs(th - np.degrees(x0)) > lim:
            fails.append(f"{label} {th:.1f} deg (ff14SB {np.degrees(x0):.1f} +- {lim:.1f})")

    seen = set()
    for (c1, r1), (c2, r2) in junctions:
        (_, a), (n2, b) = by[(c1, r1)], by[(c2, r2)]
        tag = f"{c1}{r1}-{c2}{r2}"
        bond(f"{tag} C-N", a["C"], b["N"], "C-N")
        ang(f"{tag} CA-C-N", a["CA"], a["C"], b["N"], "CA-C-N")
        ang(f"{tag} O-C-N", a["O"], a["C"], b["N"], "O-C-N")
        ang(f"{tag} C-N-CA", a["C"], b["N"], b["CA"], "C-N-CA")
        if n2 == "PRO":
            ang(f"{tag} C-N-CD", a["C"], b["N"], b["CD"], "C-N-CD")
        seen |= {(c1, r1), (c2, r2)}
    for c, r in sorted(seen):
        n, a = by[(c, r)]
        tag = f"{c}{r}"
        bond(f"{tag} N-CA", a["N"], a["CA"], "N-CA")
        ang(f"{tag} N-CA-C", a["N"], a["CA"], a["C"], "N-CA-C")
        if n == "PRO":
            bond(f"{tag} N-CD", a["N"], a["CD"], "N-CD")
            bond(f"{tag} CD-CG", a["CD"], a["CG"], "CT-CT")
            bond(f"{tag} CG-CB", a["CG"], a["CB"], "CT-CT")
            bond(f"{tag} CB-CA", a["CB"], a["CA"], "CT-CT")
            ang(f"{tag} CD-N-CA", a["CD"], a["N"], a["CA"], "CD-N-CA")
            ang(f"{tag} N-CA-CB", a["N"], a["CA"], a["CB"], "N-CA-CB(Pro)")
            ang(f"{tag} CA-CB-CG", a["CA"], a["CB"], a["CG"], "CA-CB-CG(Pro)")
            ang(f"{tag} CB-CG-CD", a["CB"], a["CG"], a["CD"], "CB-CG-CD(Pro)")
            ang(f"{tag} CG-CD-N", a["CG"], a["CD"], a["N"], "CG-CD-N(Pro)")
            ring = [a[x] for x in ("N", "CA", "CB", "CG", "CD")]
            tors = [dihedral(*(xyz_A[ring[(s + t) % 5]] for t in range(4))) for s in range(5)]
            vals[f"{tag} ring_torsions_deg"] = [round(x, 1) for x in tors]
            vals[f"{tag} pucker"] = "Cg-endo" if tors[0] > 0 else "Cg-exo"   # tors[0] = chi1 = N-CA-CB-CG
            # a closed planar-ish 5-ring has endocyclic torsions within ~+-45 deg; > 90 means the
            # ring atoms are not in a ring arrangement (geometric fact, not a tuned tolerance)
            if max(abs(x) for x in tors) > 90.0:
                fails.append(f"{tag} Pro ring torsions {vals[f'{tag} ring_torsions_deg']}")
    return vals, fails


def ha_direction(residues, xyz):
    """Angle (deg) between CA->HA and the ideal tetrahedral direction -(u_N + u_C + u_CB)
    for every non-Gly residue with N, CA, C, CB, HA.  >= 90 deg means HA sits on the
    N/C/CB side of CA, i.e. the hydrogen stereo is inverted (geometric boundary, not a
    tuned tolerance; v2 design §10-§11: PMI/v2m2 Ser11 had 168 deg).  [(label, deg)]"""
    xyz = np.asarray(xyz, dtype=float)
    out = []
    for c, rn, r, at in residues:
        if rn == "GLY" or not all(k in at for k in ("N", "CA", "C", "CB", "HA")):
            continue
        ca = xyz[at["CA"]]
        ideal = -sum((xyz[at[k]] - ca) / np.linalg.norm(xyz[at[k]] - ca) for k in ("N", "C", "CB"))
        h = xyz[at["HA"]] - ca
        out.append((f"{c}:{rn}{r}", float(np.degrees(np.arccos(np.clip(
            h @ ideal / np.linalg.norm(h) / np.linalg.norm(ideal), -1.0, 1.0))))))
    return out


def harmonic_outliers(xyz_nm, bonds, angles, atoms, nsig=NSIG):
    """Bond/angle terms of an OpenMM system touching `atoms` beyond nsig*sqrt(kT/k).

    bonds: [(i, j, r0_nm, k)], angles: [(i, j, l, theta0_rad, k)] as read from
    HarmonicBondForce / HarmonicAngleForce.  Returns [(kind, (indices), value, x0, limit)].
    """
    xyz_nm = np.asarray(xyz_nm, dtype=float)
    atoms = set(atoms)
    out = []
    for i, j, r0, k in bonds:
        if k <= 0 or not ({i, j} & atoms):
            continue
        d = float(np.linalg.norm(xyz_nm[i] - xyz_nm[j]))
        lim = nsig * np.sqrt(KT_298 / k)
        if abs(d - r0) > lim:
            out.append(("bond", (i, j), round(d * 10, 3), round(r0 * 10, 3), round(lim * 10, 3)))
    for i, j, l, t0, k in angles:
        if k <= 0 or not ({i, j, l} & atoms):
            continue
        th = np.radians(_angle_deg(xyz_nm[i], xyz_nm[j], xyz_nm[l]))
        lim = nsig * np.sqrt(KT_298 / k)
        if abs(th - t0) > lim:
            out.append(("angle", (i, j, l), round(float(np.degrees(th)), 1), round(float(np.degrees(t0)), 1),
                        round(float(np.degrees(lim)), 1)))
    return out


def close_contacts(xyz_A, bond_pairs, focus, is_h, heavy_cut=HEAVY_CLASH_A, h_cut=H_CLASH_A):
    """Non-bonded pairs (> 3 bonds apart) with a focus atom closer than the cut.

    The cut is heavy_cut for heavy-heavy pairs and h_cut when either atom is H.
    Returns (clashes [(d, i, j)], closest (d, i, j)).
    """
    xyz_A = np.asarray(xyz_A, dtype=float)
    nbr = {}
    for i, j in bond_pairs:
        nbr.setdefault(i, set()).add(j); nbr.setdefault(j, set()).add(i)
    clashes, closest, seen = [], (float("inf"), None, None), set()
    for i in focus:
        excl = {i} | nbr.get(i, set())
        for _ in range(2):
            excl |= {k for j in excl for k in nbr.get(j, ())}
        d = np.linalg.norm(xyz_A - xyz_A[i], axis=1)
        for j in np.argsort(d)[:40]:
            j = int(j)
            if j in excl:
                continue
            key = (min(i, j), max(i, j))
            if d[j] < closest[0]:
                closest = (float(d[j]), *key)
            cut = h_cut if (is_h[i] or is_h[j]) else heavy_cut
            if d[j] < cut and key not in seen:
                seen.add(key)
                clashes.append((round(float(d[j]), 3), *key))
    return sorted(clashes), closest
