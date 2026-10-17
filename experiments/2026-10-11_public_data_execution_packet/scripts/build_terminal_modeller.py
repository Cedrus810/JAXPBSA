#!/usr/bin/env python3
"""Prep protocol v2, step 1: MODELLER builds the two residues 3EQS lacks (A:25, B:12).

Run with the SYSTEM python on 181 (MODELLER 10.8 from pacman; not openmm_dev):

    cd /home/ruigengji/JAXPBSA
    python3 experiments/2026-10-11_public_data_execution_packet/scripts/build_terminal_modeller.py --res12 PRO --models 1      # V0 smoke
    python3 experiments/2026-10-11_public_data_execution_packet/scripts/build_terminal_modeller.py --res12 PRO --models 20     # V1
    python3 experiments/2026-10-11_public_data_execution_packet/scripts/build_terminal_modeller.py --res12 GLN --models 20

Design: docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §1–§4.  The clean template is
3EQS chains A/B, ATOM records only (GAI and all waters dropped), altloc A kept
(protocol §1.2).  The target sequence is the template sequence plus A:Glu25 and the
sample's residue 12; observed positions keep the template residue (mutations are
applied later by prep).  AutoModel optimises only A:25 and B:12; every other atom
keeps its template coordinates.  Model i uses its own Environ(rand_seed=-1000-i),
so any single model can be rebuilt alone.  stdlib + modeller only (no numpy).

Output: <packet>/inputs/modeller_v2/<RES12>/ with template_3eqs_clean.pdb,
alignment.ali, model_NN.pdb and build_record.json (seeds, molpdf, DOPE, timing).
Checks and start selection are a separate step (select_terminal_models.py).
Re-running skips models whose model_NN.pdb exists.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import time

PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[1]
TEMPLATE = ROOT / "experiments/2026-10-10_peptide_samples/raw/mdm2_3EQS.pdb"
AA1 = dict(zip("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
               "ACDEFGHIKLMNPQRSTVWY"))
SEED0 = -1000          # model i: rand_seed = SEED0 - i  (MODELLER requires -50000..-2)


def clean_template(src: Path, dst: Path) -> dict:
    """ATOM records of chains A/B, altloc ' ' or 'A' (altloc column blanked)."""
    lines, seq, last = [], {"A": [], "B": []}, {}
    for ln in src.read_text().splitlines():
        if not ln.startswith("ATOM") or ln[21] not in "AB" or ln[16] not in " A":
            continue
        lines.append(ln[:16] + " " + ln[17:])
        key = (ln[21], int(ln[22:26]))
        if last.get(ln[21]) != key:
            seq[ln[21]].append((key[1], ln[17:20]))
            last[ln[21]] = key
    dst.write_text("\n".join(lines) + "\nEND\n")
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res12", required=True, help="three-letter residue built at B:12 (PRO, GLN, ...)")
    ap.add_argument("--models", type=int, default=20)
    a = ap.parse_args()
    res12 = a.res12.upper()

    out = PACKET / "inputs" / "modeller_v2" / res12
    out.mkdir(parents=True, exist_ok=True)
    seq = clean_template(TEMPLATE, out / "template_3eqs_clean.pdb")
    ra, rb = seq["A"], seq["B"]
    if ra[0][0] != 26 or ra[-1][0] != 109 or rb[0][0] != 1 or rb[-1][0] != 11:
        raise SystemExit(f"unexpected template ranges A {ra[0][0]}-{ra[-1][0]}, B {rb[0][0]}-{rb[-1][0]}")
    sa = "".join(AA1[r] for _, r in ra)
    sb = "".join(AA1[r] for _, r in rb)
    ali = (">P1;tmpl\nstructureX:template_3eqs_clean:26:A:11:B::::\n"
           f"-{sa}/{sb}-*\n\n"
           ">P1;target\nsequence:target:::::::0.00: 0.00\n"
           f"E{sa}/{sb}{AA1[res12]}*\n")
    (out / "alignment.ali").write_text(ali)

    import modeller
    from modeller import Environ, Selection, log
    from modeller.automodel import AutoModel, assess

    class TerminalOnly(AutoModel):
        def special_patches(self, aln):
            self.rename_segments(segment_ids=["A", "B"], renumber_residues=[25, 1])

        def select_atoms(self):
            return Selection(self.residue_range("25:A", "25:A"), self.residue_range("12:B", "12:B"))

    log.none()
    rec_path = out / "build_record.json"
    record = json.loads(rec_path.read_text()) if rec_path.exists() else dict(
        protocol="v2_modeller", res12=res12, template=str(TEMPLATE),
        template_clean="ATOM only, chains A/B, altloc ' '/'A'; GAI and waters dropped (protocol §1.2)",
        alignment=ali, optimised="A:25 and B:12 only (AutoModel.select_atoms); all other atoms keep template xyz",
        seed_rule=f"model i: Environ(rand_seed={SEED0}-i)",
        dope_scope="AutoModel DOPE over the optimised atoms only (A:25 + B:12; checked = Selection DOPE of those residues)",
        models={})
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
            m = TerminalOnly(env, alnfile="alignment.ali", knowns="tmpl", sequence="target",
                             assess_methods=(assess.DOPE,))
            m.starting_model = m.ending_model = 1
            m.make()
            res = m.outputs[0]
            if res.get("failure"):
                record["models"][name] = dict(seed=SEED0 - i, failure=str(res["failure"]))
                continue
            shutil.move(res["name"], name)
            record["models"][name] = dict(seed=SEED0 - i, molpdf=res["molpdf"], dope=res["DOPE score"],
                                          wall_s=round(time.monotonic() - t0, 2))
            rec_path.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({name: record["models"][name]}), flush=True)
    finally:
        for f in os.listdir(out):           # MODELLER scratch files (ini/rsr/sch/D/V)
            if f.startswith("target.") and not f.endswith(".pdb"):
                os.remove(out / f)
        os.chdir(cwd)
    rec_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
