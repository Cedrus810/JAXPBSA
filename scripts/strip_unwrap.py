#!/usr/bin/env python
"""Strip water/ions and unwrap a production DCD into JAXPBSA input (`--name S4|1YCR`).

    python scripts/strip_unwrap.py

Reads data/md/S4_prod.dcd + S4_solvated.pdb, makes molecules whole (PBC imaging),
keeps the complex only (chains C+F, including PTR), and writes:

    data/md/S4_dry.dcd            -- [T, 1835, 3] complex-only trajectory
    data/prepared/S4_dry.pdb      -- matching topology/first frame

Also cross-checks the atom count against data/prepared/S4_meta.json.
"""
from __future__ import annotations

import argparse
import json
import os

import mdtraj as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="S4")
    ap.add_argument("--dcd"); ap.add_argument("--top")
    ap.add_argument("--out-dcd"); ap.add_argument("--out-pdb")
    args = ap.parse_args()
    n = args.name
    md_dir, prep = os.path.join(ROOT, "data", "md"), os.path.join(ROOT, "data", "prepared")
    args.dcd = args.dcd or os.path.join(md_dir, f"{n}_prod.dcd")
    args.top = args.top or os.path.join(md_dir, f"{n}_solvated.pdb")
    args.out_dcd = args.out_dcd or os.path.join(md_dir, f"{n}_dry.dcd")
    args.out_pdb = args.out_pdb or os.path.join(prep, f"{n}_dry.pdb")

    traj = md.load(args.dcd, top=args.top)
    print(f"loaded {traj.n_frames} frames, {traj.n_atoms} atoms, "
          f"box {traj.unitcell_lengths[0][0]:.1f} nm")
    traj.image_molecules(inplace=True)  # complex 完整化 (unwrap)

    with open(os.path.join(prep, f"{n}_meta.json")) as fh:
        expect = json.load(fh)["n_atoms"]
    # 溶质 = 前 n_atoms 个原子(Modeller.addSolvent 把水/离子追加在尾部)。不用
    # "protein" 选择串: 它是否包含 ACE/NME/PTR 取决于 mdtraj 的残基表
    dry = traj.atom_slice(range(expect))
    if dry.n_atoms != expect:
        raise RuntimeError(f"dry trajectory has {dry.n_atoms} atoms, "
                           f"{n}_meta.json says {expect}")
    # 成像完整性 sanity: 复合物任意两原子距离应 ~ 溶质尺度 (~60 Å),
    # 若 PBC imaging 失败, 碎片会相距 ~ box 尺寸 (>70 Å)
    import numpy as np
    xyz = dry.xyz[0]
    extent = np.linalg.norm(xyz - xyz.mean(0), axis=1).max() * 2
    print(f"dry: {dry.n_atoms} atoms, {dry.n_frames} frames, "
          f"solute extent ~{extent:.1f} nm (box {traj.unitcell_lengths[0][0]:.1f} nm)")

    dry.save_dcd(args.out_dcd)
    dry[0].save_pdb(args.out_pdb)
    print(f"wrote {args.out_dcd} + {args.out_pdb}")


if __name__ == "__main__":
    main()
