#!/usr/bin/env python3
"""Prep protocol v2.1, stage 2: per-sample heavy-atom model with the sample's mutations (MODELLER).

Run with the SYSTEM python on 181 (MODELLER 10.8 from pacman; not openmm_dev):

    cd /home/ruigengji/JAXPBSA
    python3 experiments/2026-10-11_public_data_execution_packet/scripts/build_mutant_modeller.py --all --start v2m1 v2m2

Design: docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §11.  The template is the stage-1
start inputs/modeller_v2/<RES12>/start_<v2mN>.pdb (heavy atoms, A:25-109 + B:1-12,
passed C1); the target is the sample's real sequence.  AutoModel optimises only the
side-chain atoms beyond CB of mutated residues (CB is in the template; no Gly here), so
every other atom keeps the stage-1 coordinates.  A sample without mutations gets the
stage-1 start copied unchanged.  Hydrogens are added afterwards by reduce
(add_h_reduce.sh): MODELLER's IC-built hydrogens failed prep C2 three times (§10).

Output: <packet>/inputs/modeller_v21/<peptide_name>/heavy_<v2mN>.pdb and
build_record.json.  Re-running skips models whose PDB exists.  stdlib + modeller.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import time

PACKET = Path(__file__).resolve().parents[1]
AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))
AA1 = {v: k for k, v in AA3.items()}
SEED = -3000            # stage-2 seed (stage 1 used -1001..-1020)


def chain_seq(pdb: Path) -> dict:
    seq, last = {}, {}
    for ln in pdb.read_text().splitlines():
        if ln.startswith("ATOM"):
            c, r = ln[21], int(ln[22:26])
            if last.get(c) != r:
                seq.setdefault(c, []).append((r, ln[17:20]))
                last[c] = r
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", nargs="+", help="peptide_name from sample_structure_audit.csv")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--start", nargs="+", default=["v2m1", "v2m2"], help="stage-1 start ids")
    a = ap.parse_args()

    rows = list(csv.DictReader((PACKET / "sample_structure_audit.csv").open()))
    if not a.all:
        rows = [r for r in rows if r["peptide_name"] in set(a.sample)]
        if len(rows) != len(set(a.sample)):
            raise SystemExit(f"unknown samples in {a.sample}")

    import modeller
    from modeller import Environ, Selection, log
    from modeller.automodel import AutoModel

    log.none()
    cwd = os.getcwd()
    for row in rows:
        name, pep = row["peptide_name"], row["peptide_sequence"]
        res12 = AA3[pep[11]]
        out = PACKET / "inputs" / "modeller_v21" / name
        out.mkdir(parents=True, exist_ok=True)
        rec_path = out / "build_record.json"
        record = json.loads(rec_path.read_text()) if rec_path.exists() else dict(
            protocol="v2.1_modeller_heavy_then_reduce", peptide_name=name, sequence=pep, models={})
        for sid in a.start:
            dst = out / f"heavy_{sid}.pdb"
            if dst.exists():
                continue
            start = PACKET / "inputs" / "modeller_v2" / res12 / f"start_{sid}.pdb"
            seq = chain_seq(start)
            ra, rb = seq["A"], seq["B"]
            if ra[0][0] != 25 or ra[-1][0] != 109 or [r for r, _ in rb] != list(range(1, 13)):
                raise SystemExit(f"{start}: unexpected ranges A {ra[0][0]}-{ra[-1][0]}, B {rb[0][0]}-{rb[-1][0]}")
            tb = "".join(AA1[x] for _, x in rb)
            mutated = [i + 1 for i, (t, s) in enumerate(zip(tb, pep)) if t != s]
            if 12 in mutated:
                raise SystemExit(f"{name}: residue 12 differs from stage-1 start {start} ({tb} vs {pep})")
            t0 = time.monotonic()
            to_build = [i for i in mutated if pep[i - 1] != "A"]
            if not to_build:
                # no mutation, or only X->Ala: Ala keeps the template N, CA, C, O, CB, so the
                # mutation is exact truncation (nothing to optimise; AutoModel refuses an
                # empty selection)
                keep = {"N", "CA", "C", "O", "CB", "OXT"}
                out_lines = []
                for ln in start.read_text().splitlines():
                    if ln.startswith("ATOM") and ln[21] == "B" and int(ln[22:26]) in mutated:
                        if ln[12:16].strip() not in keep:
                            continue
                        ln = ln[:17] + "ALA" + ln[20:]
                    out_lines.append(ln)
                dst.write_text("\n".join(out_lines) + "\n")
                res = dict(molpdf=None)
            else:
                sa = "".join(AA1[x] for _, x in ra)
                ali = (f">P1;tmpl\nstructureX:{start.stem}:25:A:12:B::::\n{sa}/{tb}*\n\n"
                       f">P1;target\nsequence:target:::::::0.00: 0.00\n{sa}/{pep}*\n")
                (out / "alignment.ali").write_text(ali)

                class Stage2(AutoModel):
                    def special_patches(self, aln):
                        self.rename_segments(segment_ids=["A", "B"], renumber_residues=[25, 1])

                    def select_atoms(self):
                        # only_sidechain() also holds CB; CB is in the template and stays fixed
                        s = Selection()
                        for i in mutated:
                            s.add(Selection(self.residue_range(f"{i}:B", f"{i}:B")).only_sidechain()
                                  - Selection(self.atoms[f"CB:{i}:B"]))
                        return s

                os.chdir(out)
                try:
                    env = Environ(rand_seed=SEED)
                    env.io.atom_files_directory = [str(start.parent)]
                    env.io.hetatm = False
                    m = Stage2(env, alnfile="alignment.ali", knowns="tmpl", sequence="target")
                    m.starting_model = m.ending_model = 1
                    m.make()
                    res = m.outputs[0]
                    if res.get("failure"):
                        raise RuntimeError(f"{name}/{sid}: MODELLER failure {res['failure']}")
                    shutil.move(res["name"], dst)
                finally:
                    for f in os.listdir(out):           # MODELLER scratch files
                        if f.startswith("target.") and not f.endswith(".pdb"):
                            os.remove(out / f)
                    os.chdir(cwd)
            record["models"][dst.name] = dict(
                start=str(start), seed=SEED if to_build else None, mutated_peptide_positions=mutated,
                method=("AutoModel (heavy atoms)" if to_build else
                        "stage-1 start; X->Ala by side-chain truncation beyond CB" if mutated else
                        "copy of stage-1 start (no mutation)"),
                optimised="side-chain atoms beyond CB of mutated residues; all other atoms = stage-1 xyz",
                molpdf=res["molpdf"], wall_s=round(time.monotonic() - t0, 2))
            record.update(modeller=modeller.info.version, python=platform.python_version(), host=platform.node())
            rec_path.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({f"{name}/{dst.name}": record["models"][dst.name]}), flush=True)


if __name__ == "__main__":
    main()
