#!/usr/bin/env python
"""M0: prepare S4 (Src SH2 + phosphotyrosyl peptide) from 1SPS chains C+F.

Chain choice (DESIGN.md §5): peptide F is the best-resolved copy (10/11 residues,
only N-term E(-3) absent) and its PTR phosphate sits in the pY pocket of SH2 chain C.

PDBFixer is NOT used. It has four separate failures on PTR:
  1. replaceNonstandardResidues() maps PTR -> TYR, silently dropping the phosphate;
  2. HETATM PTR only gets the downstream peptide bond from createStandardBonds();
  3. Modeller.addHydrogens() has no hydrogen definition for PTR, so it stays bare;
  4. addMissingAtoms() rebuilds the Topology and re-runs createStandardBonds(),
     wiping any bond added by hand -- and the result cannot be edited afterwards.
All four exist only because PDBFixer must repair the structure *after* parsing.
So this script repairs everything in the PDB text instead, and hands OpenMM a file
that already parses correctly:

  filter chains -> rename HIS/CYS to ff14SB variants -> place missing heavy atoms
  -> emit CONECT for PTR (intra-residue + both peptide bonds) -> PDBFile
  -> Modeller.addHydrogens(ff, PTR hydrogen definition) -> ff14SB+phosaa14SB System

Missing N-terminal residues (chain C GLN1, chain F GLU(-3)) are truncated, not
modeled: both are disordered termini with no bearing on the binding interface.

Outputs: data/prepared/1SPS_CF.pdb (repaired), S4_complex.pdb / S4_complex.cif
(with H), S4_meta.json

**S4_complex.cif is the canonical starting point** -- commit it and start from it.
The pipeline above is now bit-reproducible (fixed RNG seed + deterministic platform),
but that only holds for *this* OpenMM version: `Modeller.addHydrogens` places
hydrogens at random positions and fixes them with an internal minimisation, so any
change to its internals moves every atom. Pinning the artifact, not just the recipe,
is what guarantees two people compare the same structure. `S4_meta.json` carries a
SHA-256 of the coordinates so drift is detected rather than silently absorbed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter

import numpy as np
import openmm as mm
import openmm.app as app

from jaxpbsa.openmm_io import assign_radii, chain_indices, extract_nonbonded

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "1SPS.pdb")
OUT_DIR = os.path.join(ROOT, "data", "prepared")
KEEP_CHAINS = ("C", "F")
FF_XML = ("amber/ff14SB.xml", "amber/phosaa14SB.xml")

# ff14SB has no "HIS"/"CYS" templates: neutral His is HID/HIE/HIP, S-S Cys is CYX.
# The name written here is only a *starting* guess -- Modeller.addHydrogens picks the
# protonation itself from its own pH logic and ignores it, so the final name is
# re-derived from the hydrogens that actually got added (see _rename_his_by_protons).
# None of C58/C90/C96 is catalytic or metal-coordinating, so the tautomer is not
# load-bearing; what matters is that the name and the atoms agree.
HIS_VARIANT = "HIE"

#: 结构制备必须逐位可复现, 否则任何跨次的能量比较都是无效的。
#: `Modeller.addHydrogens` **把氢放在随机位置再用内部最小化修好**
#: (openmm/app/modeller.py: `random.random()` / `random.gauss()`, 见其注释
#: "The hydrogens were added at random positions"), 用的是 Python 的全局 `random`。
#: 不固定种子时实测两次运行的最终结构 RMSD 0.81 Å、最大 2.97 Å —— 全部 1835 个原子都动,
#: 足以让同一体系的 G_PB 差 54 kcal/mol(4.2%)。
#:
#: 光固定种子还不够: `addHydrogens` 内部那次"修好氢"的最小化建 Context 时
#: **不指定平台**(modeller.py:1099), OpenMM 会自动挑最快的 —— 本机有 GPU 就是 CUDA,
#: 而 CUDA 的归约顺序不确定。必须显式传一个确定性平台。
PREP_SEED = 20260913

# PTR hydrogens are exactly TYR's minus HH -- phosaa14SB PTR is dianionic, the
# phosphate carries no proton. Fed to Modeller.loadHydrogenDefinitions().
PTR_HYDROGEN_XML = """<Residues>
 <Residue name="PTR">
  <H name="H" parent="N"/>
  <H name="H2" parent="N" terminal="N"/>
  <H name="H3" parent="N" terminal="N"/>
  <H name="HA" parent="CA"/>
  <H name="HB2" parent="CB"/>
  <H name="HB3" parent="CB"/>
  <H name="HD1" parent="CD1"/>
  <H name="HD2" parent="CD2"/>
  <H name="HE1" parent="CE1"/>
  <H name="HE2" parent="CE2"/>
 </Residue>
</Residues>
"""

def _nerf(a, b, c, bond, angle, dihedral):
    """Position X with |X-a| = bond, angle(X,a,b) = angle, dihedral(X,a,b,c) = dihedral."""
    angle, dihedral = np.radians(angle), np.radians(dihedral)
    ab = a - b
    ab /= np.linalg.norm(ab)
    n = np.cross(b - c, ab)
    n /= np.linalg.norm(n)
    m = np.cross(n, ab)
    # -ab: the component along a->b must point *away* from b, or the bond angle
    # comes out as 180 deg minus the requested one.
    d = -ab * np.cos(angle) + m * np.sin(angle) * np.cos(dihedral) + n * np.sin(angle) * np.sin(dihedral)
    return a + bond * d


def _measure(x, a, b, c):
    """Internal coordinates of x relative to (a, b, c), in _nerf's convention."""
    b1, b2, b3 = a - x, b - a, c - b
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    bond = np.linalg.norm(x - a)
    angle = np.degrees(np.arccos(
        (x - a) @ (b - a) / (bond * np.linalg.norm(b - a))))
    dihedral = np.degrees(np.arctan2(
        np.cross(n1, n2) @ (b2 / np.linalg.norm(b2)), n1 @ n2))
    return bond, angle, dihedral


def _ideal_oxt(ca, c, o):
    """Third sp2 substituent on the carboxyl carbon: opposite the CA/O bisector."""
    d = -((ca - c) / np.linalg.norm(ca - c) + (o - c) / np.linalg.norm(o - c))
    return c + 1.25 * d / np.linalg.norm(d)


def _self_check():
    """_measure and _nerf must be exact inverses -- the whole rebuild rests on it."""
    res = {}
    for line in open(SRC):
        if line[:6] == "ATOM  " and line[21] == "C" and line[16] in (" ", "A"):
            res.setdefault(line[22:27].strip(), {})[line[12:16].strip()] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    for quad in (("CB", "CA", "N", "C"), ("O", "C", "CA", "N"),
                 ("CG", "CB", "CA", "N"), ("OE1", "CD", "CG", "CB")):
        r = next(a for a in res.values() if not set(quad) - set(a))
        ic = _measure(*(r[n] for n in quad))
        err = np.linalg.norm(_nerf(*(r[n] for n in quad[1:]), *ic) - r[quad[0]])
        assert err < 1e-6, f"_nerf/_measure disagree on {quad[0]}: {err:.2e} Å"


def repair_pdb(src: str, dst: str) -> dict:
    """Filter to chains C+F and complete every missing heavy atom, in PDB text."""
    ff = app.ForceField(*FF_XML)
    ptr_tmpl = ff._templates["PTR"]
    ptr_bonds = [(ptr_tmpl.atoms[i].name, ptr_tmpl.atoms[j].name) for i, j in ptr_tmpl.bonds]

    # --- read: keep chains C/F, altloc blank or A, drop waters and free phosphate
    residues = []  # [(chain, resseq, resname, {atomname: (xyz, element)})]
    for line in open(src):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        ch, rn, ri, alt = line[21], line[17:20].strip(), line[22:27].strip(), line[16]
        if ch not in KEEP_CHAINS or alt not in (" ", "A") or rn in ("HOH", "PO4"):
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        el = (line[76:78].strip() or line[12:16].strip()[0]).capitalize()
        if el == "H":
            continue  # all hydrogens are rebuilt by Modeller
        if not residues or residues[-1][:3] != (ch, ri, rn):
            residues.append((ch, ri, rn, {}))
        residues[-1][3].setdefault(line[12:16].strip(), (xyz, el))

    # --- rename to ff14SB residue variants (HIS has no template; CYX for disulfides)
    sg = {i: r[3]["SG"][0] for i, r in enumerate(residues) if r[2] == "CYS" and "SG" in r[3]}
    ss = {i for i in sg for j in sg
          if i != j and np.linalg.norm(sg[i] - sg[j]) < 2.5}
    renamed = Counter()
    for i, (ch, ri, rn, atoms) in enumerate(residues):
        new = HIS_VARIANT if rn == "HIS" else ("CYX" if i in ss else rn)
        if new != rn:
            renamed[f"{rn}->{new}"] += 1
            residues[i] = (ch, ri, new, atoms)

    # --- complete missing heavy atoms against the force-field templates.
    #     Internal coordinates are measured from an intact residue of the same type
    #     elsewhere in the structure rather than hard-coded, so the rebuilt geometry
    #     is self-consistent with this file; the vacuum minimisation relaxes it.
    added = []
    last_of_chain = {r[0]: i for i, r in enumerate(residues)}
    adj = {}
    for rn in {r[2] for r in residues}:
        tmpl = ff._templates[rn]
        heavy = {a.name for a in tmpl.atoms if a.element is not None and a.element.symbol != "H"}
        g = {n: [] for n in heavy}
        for i1, i2 in tmpl.bonds:
            n1, n2 = tmpl.atoms[i1].name, tmpl.atoms[i2].name
            if n1 in heavy and n2 in heavy:
                g[n1].append(n2)
                g[n2].append(n1)
        adj[rn] = g

    def _refs(rn, x, have):
        """Three present atoms (a, b, c) defining x: a bonded to x, then outward."""
        for a in adj[rn][x]:
            if a not in have:
                continue
            for b in adj[rn][a]:
                if b == x or b not in have:
                    continue
                for c in adj[rn][b] + adj[rn][a]:
                    if c not in (x, a, b) and c in have:
                        return a, b, c
        return None

    for i, (ch, ri, rn, atoms) in enumerate(residues):
        want = {a.name for a in ff._templates[rn].atoms
                if a.element is not None and a.element.symbol != "H"}
        want = (want | {"OXT"}) if i == last_of_chain[ch] else (want - {"OXT"})
        todo = sorted(want - set(atoms))
        while todo:
            progressed = False
            for name in list(todo):
                if name == "OXT":
                    xyz = _ideal_oxt(atoms["CA"][0], atoms["C"][0], atoms["O"][0])
                else:
                    refs = _refs(rn, name, set(atoms))
                    if refs is None:
                        continue
                    donor = next(
                        (d for _, _, drn, d in residues
                         if drn == rn and not {name, *refs} - set(d)), None)
                    if donor is None:
                        raise RuntimeError(
                            f"no intact {rn} in the structure to copy {name} from")
                    ic = _measure(*(donor[n][0] for n in (name,) + refs))
                    xyz = _nerf(*(atoms[n][0] for n in refs), *ic)
                atoms[name] = (xyz, name[0])
                added.append(f"{ch}/{rn}{ri}/{name}")
                todo.remove(name)
                progressed = True
            if not progressed:
                raise RuntimeError(f"cannot anchor {todo} in {ch}/{rn}{ri}")

    # --- write, renumbering serials so CONECT can reference them
    lines, serial, serial_of = [], 0, {}
    next_chain = {id(r[3]): (residues[i + 1][0] if i + 1 < len(residues) else None)
                  for i, r in enumerate(residues)}
    for ch, ri, rn, atoms in residues:
        for name, (xyz, el) in atoms.items():
            serial += 1
            serial_of[(ch, ri, name)] = serial
            pad = "" if len(name) >= 4 or len(el) == 2 else " "
            lines.append(
                f"ATOM  {serial:5d} {pad}{name:<3s} {rn:>3s} {ch}{ri:>4s}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
                f"          {el:>2s}"
            )
        if ch != next_chain.get(id(atoms)):
            lines.append(f"TER   {serial + 1:5d}      {rn:>3s} {ch}{ri:>4s}")

    # --- CONECT for PTR: PDBFile knows no template for it and 1SPS ships no CONECT.
    #     Both peptide bonds are emitted too; PDBFile drops the duplicate that
    #     createStandardBonds() already made from the following residue's "-C N" rule.
    conect, n_ptr_bonds = {}, 0
    for i, (ch, ri, rn, atoms) in enumerate(residues):
        if rn != "PTR":
            continue
        pairs = [(serial_of[(ch, ri, x)], serial_of[(ch, ri, y)]) for x, y in ptr_bonds
                 if x in atoms and y in atoms]
        for nb, at in ((i - 1, "C"), (i + 1, "N")):
            if not 0 <= nb < len(residues):
                continue
            n_ch, n_ri, _, n_atoms = residues[nb]
            if n_ch == ch and at in n_atoms:
                own = "N" if at == "C" else "C"
                pairs.append((serial_of[(ch, ri, own)], serial_of[(n_ch, n_ri, at)]))
        n_ptr_bonds += len(pairs)
        for a, b in pairs:
            conect.setdefault(a, []).append(b)
            conect.setdefault(b, []).append(a)
    for a in sorted(conect):
        for k in range(0, len(conect[a]), 4):
            lines.append("CONECT" + f"{a:5d}" + "".join(f"{b:5d}" for b in conect[a][k:k + 4]))
    lines.append("END")
    with open(dst, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return {
        "n_residues": len(residues),
        "renamed": dict(renamed),
        "added_heavy_atoms": added,
        "n_ptr_conect_bonds": n_ptr_bonds,
        "truncated": ["chain C GLN1 (N-term, disordered)", "chain F GLU(-3) (N-term, disordered)"],
    }


def _rename_his_by_protons(topology) -> dict:
    """Make each histidine's residue name match the hydrogens it actually carries.

    `Modeller.addHydrogens` chooses the protonation state from its own pH logic and
    ignores whatever name the input PDB used, so a residue written as HIE can come
    back carrying HD1 -- i.e. it is really HID. OpenMM itself matches templates by
    bond graph, so its physics stays correct, but the *name* is then a lie and any
    downstream tool that trusts names (tleap, MDAnalysis, PDB readers) gets it wrong.

    HD1 only -> HID, HE2 only -> HIE, both -> HIP (+1).
    """
    counts = Counter()
    for res in topology.residues():
        if not res.name.startswith("HI"):
            continue
        names = {a.name for a in res.atoms()}
        new = {(True, False): "HID", (False, True): "HIE",
               (True, True): "HIP"}.get(("HD1" in names, "HE2" in names))
        if new is None:
            raise RuntimeError(f"histidine {res.name}{res.id} has neither HD1 nor HE2")
        counts[f"{res.name}{res.id}->{new}"] += 1
        res.name = new
    return dict(counts)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    _self_check()

    repaired = os.path.join(OUT_DIR, "1SPS_CF.pdb")
    info = repair_pdb(SRC, repaired)
    print(f"[1/4] repaired PDB -> {repaired}")
    print(f"       {info['n_residues']} residues; renamed {info['renamed']}")
    print(f"       placed {len(info['added_heavy_atoms'])} missing heavy atoms: "
          f"{info['added_heavy_atoms']}")
    print(f"       truncated (not modeled): {info['truncated']}")
    print(f"       wrote {info['n_ptr_conect_bonds']} CONECT bonds for PTR")

    pdb = app.PDBFile(repaired)
    ff = app.ForceField(*FF_XML)
    ptr = next(r for r in pdb.topology.residues() if r.name == "PTR")
    n_ptr = sum(1 for b in pdb.topology.bonds() if ptr in (b[0].residue, b[1].residue))
    print(f"[2/4] parsed: {pdb.topology.getNumAtoms()} heavy atoms, "
          f"{pdb.topology.getNumBonds()} bonds ({n_ptr} touching PTR)")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.loadHydrogenDefinitions(_write_tmp(PTR_HYDROGEN_XML))
    random.seed(PREP_SEED)  # 见 PREP_SEED 的说明: 不固定种子结构就不可复现
    # Reference 平台是确定性的; 这里只有 1835 个原子、50 步最小化, 慢一点无所谓。
    # 不传 platform 会落到 CUDA 上, 结构就不可复现了。
    modeller.addHydrogens(ff, pH=7.0,
                          platform=mm.Platform.getPlatformByName("Reference"))
    his_fixed = _rename_his_by_protons(modeller.topology)
    if his_fixed:
        print(f"       histidine names re-derived from actual protons: {his_fixed}")
    n_ptr_h = sum(1 for a in modeller.topology.atoms()
                  if a.residue.name == "PTR" and a.element.symbol == "H")
    print(f"[3/4] hydrogens added: {modeller.topology.getNumAtoms()} atoms total, "
          f"{n_ptr_h} on PTR")
    if n_ptr_h != 8:
        raise RuntimeError(f"PTR should carry 8 hydrogens, got {n_ptr_h}")

    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds, rigidWater=False, removeCMMotion=False,
    )
    # **单线程是为了可复现, 不是为了省事。** CPU 平台多线程下力的求和顺序不确定,
    # 逐位不同的力会让 L-BFGS 走上不同轨迹, 2000 步后放大成 RMSD 0.78 Å / 最大 2.42 Å
    # 的结构差异 —— 实测过。结构一变, 所有能量数字就不能跨次比较。
    context = mm.Context(system, mm.VerletIntegrator(0.001 * mm.unit.picoseconds),
                         mm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    context.setPositions(modeller.positions)
    kj = mm.unit.kilojoule_per_mole
    e0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 2000)
    e1 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
    positions = context.getState(getPositions=True).getPositions()
    del context
    print(f"       vacuum minimisation {e0:.2e} -> {e1:.1f} kJ/mol")

    params = extract_nonbonded(system)
    radii = assign_radii(modeller.topology, "mbondi2")
    lig_idx = chain_indices(modeller.topology, "F")
    rec_idx = chain_indices(modeller.topology, "C")
    net_q, lig_q, rec_q = (params.charge.sum(), params.charge[lig_idx].sum(),
                           params.charge[rec_idx].sum())
    res_counts = Counter(r.name for r in modeller.topology.residues())
    print(f"[4/4] atoms={system.getNumParticles()} "
          f"(receptor {rec_idx.size} + ligand {lig_idx.size})\n"
          f"       net charge: complex {net_q:+.3f} e = receptor {rec_q:+.3f} "
          f"+ ligand {lig_q:+.3f}\n"
          f"       radii range: {radii.min():.2f}-{radii.max():.2f} Å")
    if "PTR" not in res_counts:
        raise RuntimeError("PTR lost during preparation -- the phosphate is the point")

    out_pdb = os.path.join(OUT_DIR, "S4_complex.pdb")
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, positions, fh, keepIds=True)
    # 规范起点: 大家从这个文件出发, 而不是各自重跑管道
    out_cif = os.path.join(OUT_DIR, "S4_complex.cif")
    with open(out_cif, "w") as fh:
        app.PDBxFile.writeFile(modeller.topology, positions, fh, keepIds=True)
    # **对写出的文件取哈希, 不是对内存里的坐标** —— 后者别人无法核对, 而且
    # 写文件本身会舍入, 两者对不上。`sha256sum S4_complex.cif` 就能验。
    # 序列化 System: 电荷 / sigma / epsilon / 键 / exceptions 全部固化。
    # 下游从此不需要 ForceField, 也就不需要 openmmforcefields 和它拖来的一整串依赖。
    out_sys = os.path.join(OUT_DIR, "S4_system.xml")
    with open(out_sys, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
    cif_sha, sys_sha = sha(out_cif), sha(out_sys)
    meta = {
        "source": "1SPS chains C (SH2) + F (peptide EPQ-pY-EEIPIYL)",
        "forcefield": list(FF_XML),
        "radii_model": "mbondi2",
        "prep_seed": PREP_SEED,
        "canonical_structure": "S4_complex.cif",
        "canonical_system": "S4_system.xml",
        "canonical_sha256": cif_sha,      # sha256sum data/prepared/S4_complex.cif
        "system_sha256": sys_sha,         # sha256sum data/prepared/S4_system.xml
        "receptor_idx": [int(i) for i in np.asarray(rec_idx)],
        "ligand_idx": [int(i) for i in np.asarray(lig_idx)],
        "n_atoms": system.getNumParticles(),
        "n_receptor": int(rec_idx.size),
        "n_ligand": int(lig_idx.size),
        "net_charge_complex": float(net_q),
        "net_charge_receptor": float(rec_q),
        "net_charge_ligand": float(lig_q),
        "receptor_chain": "C",
        "ligand_chain": "F",
        "residue_counts": dict(res_counts),
        "histidine_renamed": his_fixed,
        **info,
    }
    with open(os.path.join(OUT_DIR, "S4_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"       wrote {out_pdb}, {out_cif}, {out_sys} + S4_meta.json")
    print("       规范起点(下游只读这两个, 不再碰力场):")
    print(f"         S4_complex.cif  sha256 {cif_sha[:16]}…")
    print(f"         S4_system.xml   sha256 {sys_sha[:16]}…")


def _write_tmp(text: str) -> str:
    path = os.path.join(OUT_DIR, "_ptr_hydrogens.xml")
    with open(path, "w") as fh:
        fh.write(text)
    return path


if __name__ == "__main__":
    main()
