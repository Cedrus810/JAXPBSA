from .system import MMParams, extract_nonbonded
from .radii import assign_radii
from .topology import chain_indices, species_indices

__all__ = [
    "MMParams",
    "extract_nonbonded",
    "assign_radii",
    "species_indices",
    "chain_indices",
]


def load_canonical(name="S4", root=None, check_hash=True):
    """加载规范起点(`name`: "S4" | "1YCR", 对应 data/prepared/{name}_meta.json): **拓扑 + 坐标 + 已序列化的 System**, 不碰任何力场。

    返回 `dict(topology, positions_A, system, charge, sigma, epsilon, radii,
    receptor_idx, ligand_idx, meta)`。

    `scripts/prep_s4.py` 是个**转换器**: 破损的 PDB、PTR、加氢、力场、残基命名
    全是它的事。它跑一次, 交出自包含的 OpenMM 产物; 下游只读这个, 从此不需要
    ForceField, 也就不需要 `openmmforcefields` 和它拖来的一整串依赖。

    为什么固定产物而不是相信配方: 管道**现在**是逐位可复现的(固定 RNG 种子 +
    给 `addHydrogens` 指定确定性平台 + 单线程最小化), 但那只对**当前这个 OpenMM
    版本**成立 —— `addHydrogens` 把氢放在随机位置再用内部最小化修好, 它的实现一变
    每个原子都会动。实测未固定时两次运行 RMSD 0.78 Å、最大 2.42 Å,
    足以让同一体系的 G_PB 差 54 kcal/mol(4.2%)。

    `check_hash=True` 时对照 meta 里的 sha256, **漂移会报错而不是被悄悄吸收**。
    """
    import hashlib
    import json
    import os

    import numpy as np
    import openmm as mm
    import openmm.app as app

    from .radii import assign_radii
    from .system import extract_nonbonded

    root = root or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "data", "prepared")
    meta = json.load(open(os.path.join(d, f"{name}_meta.json")))
    cif = os.path.join(d, meta["canonical_structure"])
    sysx = os.path.join(d, meta["canonical_system"])
    if check_hash:
        for path, key in ((cif, "canonical_sha256"), (sysx, "system_sha256")):
            got = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if got != meta[key]:
                raise RuntimeError(
                    f"{os.path.basename(path)} 的 sha256 与 {name}_meta.json 不符\n"
                    f"  记录 {meta[key]}\n  实际 {got}\n"
                    "产物已漂移 —— 不要在此基础上比较能量。"
                    "重跑 scripts/prep_s4.py, 或取回原文件。")
    f = app.PDBxFile(cif)
    system = mm.XmlSerializer.deserialize(open(sysx).read())
    p = extract_nonbonded(system)
    return {
        "topology": f.topology,
        "positions_A": np.array([[v.x, v.y, v.z] for v in f.positions]) * 10.0,
        "system": system,
        "charge": np.asarray(p.charge),
        "sigma": np.asarray(p.sigma),
        "epsilon": np.asarray(p.epsilon),
        "radii": np.asarray(assign_radii(f.topology, meta["radii_model"])),
        "receptor_idx": np.asarray(meta["receptor_idx"]),
        "ligand_idx": np.asarray(meta["ligand_idx"]),
        "meta": meta,
    }
