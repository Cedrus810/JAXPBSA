#!/usr/bin/env python
"""JCP 545 (2026) 114452 Table 4 的 25 个蛋白: RCSB 原始 PDB -> 我们的 PB 输入(单结构溶剂化能 sanity set)。

    python scripts/prep_pdb25.py            # 读 data/pdb25/raw/{ID}.pdb, 写 data/pdb25/prep/{ID}.npz / .pdb

与 prep_1ycr.py 同一套约定(PDBFixer + Amber ff14SB + mbondi2), 差别: 完整蛋白不加 ACE/NME 帽、不补缺失环
(missingResidues 清空, 只补缺失重原子)、非标准残基换成标准、**去掉全部杂原子**(血红素、Fe–S 簇、金属、
溶剂分子), 不做能量最小化(晶体坐标 + addHydrogens 的氢)。原文的制备方式 JCP26 没写, 所以绝对值只作粗对照。
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# (PDB, JCP26 Table 4 的原子数, GEC h=1.0 / 0.5 / 0.25, FF h=1.0 / 0.5 / 0.25) —— 非线性 PB, 只作参照
TABLE4 = {
    "1AHO": (962, -168.40, -201.04, -199.99, -168.249, -198.64, -199.48),
    "1C75": (985, -588.19, -613.81, -611.79, -590.48, -611.62, -611.33),
    "1J0P": (1597, -844.98, -902.54, -898.89, -848.54, -898.21, -897.99),
    "1TG0": (1029, -1710.41, -1738.68, -1737.93, -1716.73, -1736.84, -1737.56),
    "1X8Q": (2815, -439.36, -515.08, -514.28, -440.25, -509.39, -513.06),
    "1CBN": (639, -57.04, -70.99, -69.91, -56.20, -69.56, -69.61),
    "1G6X": (888, -555.79, -584.14, -583.42, -557.68, -581.84, -582.95),
    "1IUA": (1207, -170.77, -200.60, -197.94, -170.26, -197.88, -197.37),
    "1L9L": (1226, -1564.57, -1600.23, -1601.93, -1571.10, -1597.50, -1601.39),
    "1M1Q": (1265, -734.58, -775.91, -775.30, -736.80, -772.57, -774.58),
    "1NWZ": (1912, -635.43, -686.80, -684.25, -637.52, -683.02, -683.46),
    "1OK0": (1076, -427.88, -457.82, -456.99, -428.34, -455.39, -456.47),
    "1TQG": (1660, -704.44, -733.07, -728.52, -705.60, -729.86, -727.85),
    "1VB0": (913, -217.51, -245.41, -242.24, -216.92, -242.52, -241.63),
    "1VBW": (1056, -778.73, -808.32, -807.54, -780.76, -805.80, -807.03),
    "1W0N": (1756, -537.00, -592.71, -593.75, -537.24, -589.14, -592.99),
    "1X6X": (1732, -321.90, -367.40, -363.27, -321.93, -363.37, -362.43),
    "1XMK": (1268, -256.38, -291.44, -291.80, -256.52, -288.50, -291.17),
    "1ZUU": (868, -411.39, -438.53, -436.94, -413.96, -436.50, -436.52),
    "1ZZK": (1252, -345.79, -382.37, -381.02, -347.78, -379.68, -380.46),
    "2FDN": (731, -820.02, -840.48, -840.65, -824.04, -839.26, -840.41),
    "2FMA": (924, -279.70, -308.20, -306.85, -280.92, -306.05, -306.40),
    "2FWH": (1830, -586.41, -627.40, -626.19, -587.83, -623.83, -625.44),
    "2H5C": (2755, -658.85, -710.01, -707.07, -658.17, -704.87, -705.99),
    "2IDQ": (1596, -428.61, -465.81, -464.29, -430.14, -462.82, -463.66),
}
RAW, OUT = os.path.join(ROOT, "data", "pdb25", "raw"), os.path.join(ROOT, "data", "pdb25", "prep")
SEED = 20260929
# 制备方式的单因素变体(离群蛋白排查用), 名字 = 基 PDB + "_" + 标签; 能量对照仍取基 PDB 的 TABLE4 行。
#   cym:   这些 Cys 取去质子的 CYM(−1)。1C75 的 C32/C35 原本与血红素共价相连, 去掉血红素后原文 985 原子
#          比我们少 2 个 —— 正好是两个 HG。
#   altloc: 取这个 altloc(默认 PDBFixer/OpenMM 取第一个)。1TQG 有 1228 个 altloc 原子。
VARIANTS = {
    "1C75_cym": ("1C75", {"cym": (32, 35)}),
    "1TQG_altB": ("1TQG", {"altloc": "B"}),
}


def _raw_path(base: str, altloc: str | None) -> str:
    src = os.path.join(RAW, f"{base}.pdb")
    if altloc is None:
        return src
    # 只留空白和指定 altloc 的原子行, altloc 列清空(否则 OpenMM 仍按「第一个」取)
    dst = os.path.join(OUT, f"{base}_alt{altloc}.raw.pdb")
    os.makedirs(OUT, exist_ok=True)
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            if line.startswith(("ATOM", "HETATM")):
                if line[16] not in (" ", altloc):
                    continue
                line = line[:16] + " " + line[17:]
            fo.write(line)
    return dst


def prep(pid: str) -> dict:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    from pdbfixer import PDBFixer
    from jaxpbsa.openmm_io import assign_radii, extract_nonbonded

    ref = mm.Platform.getPlatformByName("Reference")
    base, opt = VARIANTS.get(pid, (pid, {}))
    fixer = PDBFixer(filename=_raw_path(base, opt.get("altloc")), platform=ref)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # 不补缺失环(晶体里没有的就没有)
    fixer.findNonstandardResidues()
    nonstd = [(r.name, s) for r, s in fixer.nonstandardResidues]
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    n_missing = sum(len(v) for v in fixer.missingAtoms.values())
    fixer.addMissingAtoms(seed=SEED)
    ff = app.ForceField("amber/ff14SB.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    random.seed(SEED)
    variants = None
    if opt.get("cym"):
        # addHydrogens 只认 CYS/CYX; 去掉 HG 后 SG 没有外键, ff14SB 按图匹配到的是 CYM(−1)
        variants = [("CYX" if (r.name == "CYS" and int(r.id) in opt["cym"]) else None)
                    for r in modeller.topology.residues()]
        assert variants.count("CYX") == len(opt["cym"]), (pid, variants.count("CYM"))
    modeller.addHydrogens(ff, pH=7.0, variants=variants, platform=ref)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    q = np.asarray(extract_nonbonded(system).charge, float)
    radii = np.asarray(assign_radii(modeller.topology, "mbondi2"), float)
    mass = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                     for i in range(system.getNumParticles())])
    xyz = np.asarray(modeller.positions.value_in_unit(unit.angstrom), float)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{pid}.pdb"), "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)
    np.savez(os.path.join(OUT, f"{pid}.npz"), coords_A=xyz, charge=q, radii=radii, masses=mass)
    return {"n": len(q), "q": q.sum(), "nonstd": nonstd, "missing": n_missing,
            "chains": modeller.topology.getNumChains(), "res": modeller.topology.getNumResidues()}


def main() -> None:
    ids = sys.argv[1:] or list(TABLE4)
    print(f"{'PDB':5s} {'原子(我们)':>9s} {'原子(JCP26)':>11s} {'净电荷':>7s} {'链':>3s} {'残基':>5s}  备注", flush=True)
    for pid in ids:
        try:
            s = prep(pid)
            note = []
            if s["nonstd"]:
                note.append(f"非标准→标准 {s['nonstd']}")
            if s["missing"]:
                note.append(f"补重原子 {s['missing']}")
            print(f"{pid:5s} {s['n']:9d} {TABLE4[VARIANTS.get(pid, (pid,))[0]][0]:11d} {s['q']:+7.2f} {s['chains']:3d} {s['res']:5d}  "
                  f"{'; '.join(note)}", flush=True)
        except Exception as e:
            print(f"{pid:5s} 失败: {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
