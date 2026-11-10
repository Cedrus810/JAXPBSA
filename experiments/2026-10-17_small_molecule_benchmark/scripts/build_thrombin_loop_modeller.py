#!/usr/bin/env python3
"""Hou a1-a4 (1D3D/1D3P/1D3Q/1D3T): MODELLER builds the missing autolysis loop B184-190.

Run with the SYSTEM python on 181 (MODELLER 10.8; not openmm_dev), user 2026-10-19:

    cd /home/ruigengji/JAXPBSA
    python3 experiments/2026-10-17_small_molecule_benchmark/scripts/build_thrombin_loop_modeller.py --sample a1 --models 1   # smoke
    python3 experiments/2026-10-17_small_molecule_benchmark/scripts/build_thrombin_loop_modeller.py --sample a1 --models 10

Plan: docs/BRD4_HOU_EXECUTION_HANDOFF.md §3.1, §7.3, §7.10.  Follows the peptide line's
build_terminal_modeller.py (template = ATOM records only, AutoModel.select_atoms limited to the
built residues so every observed atom keeps its crystal xyz, model i seeded rand_seed=-1000-i).

  * template: receptor chains A (light) and B (heavy), ATOM records only -- hirugen chain H,
    NAG, Na+, waters and the ligand are not given to MODELLER (they are carried over from the
    crystal in the later prep step, handoff §3.1 table);
  * target: template sequence with B184-190 inserted, sequence taken from the file's own
    REMARK 465 (WTANVGK, chymotrypsin 148-149E); nothing else is added: all four structures
    observe exactly A6-33 and B37-293, so a1-a4 then share one receptor sequence;
  * optimised atoms: B184-190 only;
  * every model gets the loop DOPE (Selection DOPE of B184-190) and the minimum heavy-atom
    distance from the built loop to the crystal ligand and to the other crystal heteroatoms
    kept later (Na+, waters).  The model with the lowest loop DOPE is copied to
    receptor_heavy.pdb.  Distances are recorded, not used to reject.

Output: <package>/prepared/hou/<sample>/modeller/ with template_clean.pdb, alignment.ali,
model_NN.pdb, build_record.json, receptor_heavy.pdb.  Re-running skips existing models and
redoes the selection.  stdlib + modeller only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

PACKAGE = Path(__file__).resolve().parents[1]
AA1 = dict(zip("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
               "ACDEFGHIKLMNPQRSTVWY"))
SEED0 = -1000
LOOP = (184, 190)


def read_pdb(path: Path):
    atoms, missing = [], []
    for ln in path.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            atoms.append(dict(rec=ln[:6].strip(), name=ln[12:16].strip(), alt=ln[16], res=ln[17:20], ch=ln[21],
                              num=int(ln[22:26]), ic=ln[26], xyz=(float(ln[30:38]), float(ln[38:46]), float(ln[46:54])),
                              el=ln[76:78].strip(), line=ln))
        elif ln.startswith("REMARK 465") and len(ln.split()) == 5 and ln.split()[3] == "B":
            p = ln.split()
            if p[4].isdigit():
                missing.append((int(p[4]), p[2]))
    return atoms, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, choices=["a1", "a2", "a3", "a4"])
    ap.add_argument("--models", type=int, default=10)
    a = ap.parse_args()

    sel = {r["sample_id"]: r for r in csv.DictReader((PACKAGE / "manifests/hou_thrombin_selection.csv").open())}[a.sample]
    pdb_id = {r["sample_id"]: r for r in csv.DictReader((PACKAGE / "manifests/hou2011_table_s1.csv").open())}[a.sample]["pdb_id"]
    src = PACKAGE / f"sources/hou2011/pdb/{pdb_id}.pdb"
    out = PACKAGE / "prepared/hou" / a.sample / "modeller"
    out.mkdir(parents=True, exist_ok=True)
    atoms, missing = read_pdb(src)

    # template: receptor chains, ATOM only, no altlocs present in these files (checked)
    rch = sel["receptor_chains"].split(",")
    tmpl = [x for x in atoms if x["rec"] == "ATOM" and x["ch"] in rch]
    if any(x["alt"] not in " A" for x in tmpl) or any(x["ic"] != " " for x in tmpl):
        raise SystemExit("unexpected altloc/insertion codes in template chains")
    (out / "template_clean.pdb").write_text("\n".join(x["line"][:16] + " " + x["line"][17:] for x in tmpl) + "\nEND\n")
    seq = {}
    for x in tmpl:
        if not seq.get(x["ch"]) or seq[x["ch"]][-1][0] != x["num"]:
            seq.setdefault(x["ch"], []).append((x["num"], x["res"]))
    ra, rb = seq["A"], seq["B"]
    nums_b = [n for n, _ in rb]
    loop = [r for n, r in sorted(missing) if LOOP[0] <= n <= LOOP[1]]
    if (ra[0][0], ra[-1][0], rb[0][0], rb[-1][0]) != (6, 33, 37, 293) or len(loop) != 7 \
            or 183 not in nums_b or 191 not in nums_b or any(LOOP[0] <= n <= LOOP[1] for n in nums_b):
        raise SystemExit(f"unexpected template ranges/loop: A {ra[0][0]}-{ra[-1][0]} B {rb[0][0]}-{rb[-1][0]} loop {loop}")
    sa = "".join(AA1[r] for _, r in ra)
    cut = nums_b.index(183) + 1
    sb_t = "".join(AA1[r] for _, r in rb[:cut]) + "-" * 7 + "".join(AA1[r] for _, r in rb[cut:])
    sb = "".join(AA1[r] for _, r in rb[:cut]) + "".join(AA1[r] for r in loop) + "".join(AA1[r] for _, r in rb[cut:])
    ali = (">P1;tmpl\nstructureX:template_clean:6:A:293:B::::\n" f"{sa}/{sb_t}*\n\n"
           ">P1;target\nsequence:target:::::::0.00: 0.00\n" f"{sa}/{sb}*\n")
    (out / "alignment.ali").write_text(ali)

    import modeller
    from modeller import Environ, Selection, log
    from modeller.automodel import AutoModel, assess

    class LoopOnly(AutoModel):
        def special_patches(self, aln):
            self.rename_segments(segment_ids=["A", "B"], renumber_residues=[6, 37])

        def select_atoms(self):
            return Selection(self.residue_range(f"{LOOP[0]}:B", f"{LOOP[1]}:B"))

    log.none()
    rec_path = out / "build_record.json"
    record = json.loads(rec_path.read_text()) if rec_path.exists() else dict(
        sample=a.sample, pdb=pdb_id, source=str(src), receptor_chains=rch,
        template_clean="ATOM records of the receptor chains; hirugen, NAG, Na+, waters and ligand not given to MODELLER",
        loop_from="REMARK 465 of the source file", loop_sequence="".join(AA1[r] for r in loop),
        alignment=ali, optimised=f"B{LOOP[0]}-{LOOP[1]} only (AutoModel.select_atoms); all other atoms keep crystal xyz",
        seed_rule=f"model i: Environ(rand_seed={SEED0}-i)", models={})
    record.update(modeller=modeller.info.version, python=platform.python_version(), host=platform.node())
    cwd = os.getcwd()
    os.chdir(out)
    try:
        for i in range(1, a.models + 1):
            name = f"model_{i:02d}.pdb"
            if (out / name).exists():
                continue
            t0 = time.monotonic()
            env = Environ(rand_seed=SEED0 - i)
            env.io.atom_files_directory = [str(out)]
            env.io.hetatm = False
            m = LoopOnly(env, alnfile="alignment.ali", knowns="tmpl", sequence="target", assess_methods=(assess.DOPE,))
            m.starting_model = m.ending_model = 1
            m.make()
            res = m.outputs[0]
            if res.get("failure"):
                record["models"][name] = dict(seed=SEED0 - i, failure=str(res["failure"])); continue
            shutil.move(res["name"], name)
            mdl = modeller.Model(env, file=name)
            loop_dope = Selection(mdl.residue_range(f"{LOOP[0]}:B", f"{LOOP[1]}:B")).assess_dope()
            record["models"][name] = dict(seed=SEED0 - i, molpdf=res["molpdf"], dope=res["DOPE score"],
                                          loop_dope=loop_dope, wall_s=round(time.monotonic() - t0, 2))
            rec_path.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({name: record["models"][name]}), flush=True)
    finally:
        for f in os.listdir(out):
            if f.startswith("target.") and not f.endswith(".pdb"):
                os.remove(out / f)
        os.chdir(cwd)

    # recorded per model: loop vs crystal ligand / kept heteroatoms; observed atoms unchanged
    lig = [x["xyz"] for x in atoms if x["rec"] == "HETATM" and x["res"] == sel["ligand_resname"]
           and x["ch"] == sel["ligand_chain"] and x["num"] == int(sel["ligand_resseq"])]
    kept_het = [x["xyz"] for x in atoms if x["rec"] == "HETATM" and x["res"] in ("NA", "HOH")]
    tmpl_keys = {(t["ch"], t["num"], t["name"]) for t in tmpl}
    for name, info in sorted(record["models"].items()):
        if "failure" in info or not (out / name).exists():
            continue
        m_atoms, _ = read_pdb(out / name)
        loop_h = [x["xyz"] for x in m_atoms if x["ch"] == "B" and LOOP[0] <= x["num"] <= LOOP[1] and x["el"] != "H"]
        # MODELLER relabels symmetric atoms (PHE/TYR CD1/CD2, CE1/CE2, ...) by its own convention,
        # so compare each crystal atom with the nearest model atom of the same residue and element
        by_res = {}
        for x in m_atoms:
            by_res.setdefault((x["ch"], x["num"], x["name"][0]), []).append(x["xyz"])
        moved = max((min(math.dist(x["xyz"], q) for q in by_res.get((x["ch"], x["num"], x["name"][0]), [(1e9,) * 3]))
                     for x in tmpl), default=None)
        info["added_atoms_outside_loop"] = sorted(f"{x['ch']}{x['num']}.{x['name']}" for x in m_atoms
                                                  if not (x["ch"] == "B" and LOOP[0] <= x["num"] <= LOOP[1])
                                                  and (x["ch"], x["num"], x["name"]) not in tmpl_keys)
        info.update(n_loop_heavy=len(loop_h),
                    min_loop_ligand_A=round(min(math.dist(p, q) for p in loop_h for q in lig), 2),
                    min_loop_kept_het_A=round(min(math.dist(p, q) for p in loop_h for q in kept_het), 2) if kept_het else None,
                    max_shift_observed_atoms_A=round(moved, 4) if moved is not None else None)
    ok = {k: v for k, v in record["models"].items() if "loop_dope" in v}
    if ok:
        best = min(ok, key=lambda k: ok[k]["loop_dope"])
        shutil.copy(out / best, out / "receptor_heavy.pdb")
        record["selected"] = dict(model=best, rule="lowest loop DOPE (Selection DOPE of B184-190)", **ok[best])
    rec_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"sample": a.sample, "selected": record.get("selected")}), flush=True)


if __name__ == "__main__":
    main()
