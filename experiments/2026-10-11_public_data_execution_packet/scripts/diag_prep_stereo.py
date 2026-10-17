#!/usr/bin/env python3
"""Where does PDBFixer's rebuild invert a CA (PMI-L10A A:GLU25 came out D, P2 log 2026-10-12)?

    cd /home/ruigengji/JAXPBSA
    $PY experiments/2026-10-11_public_data_execution_packet/scripts/diag_prep_stereo.py \
        --sample PMI PMI-L10A P12Q_L10A

Runs the unchanged `scripts/prep_peptide_sample.py::prepare` (same template,
mutations, seed, Reference platform) with read-only hooks on PDBFixer internals
and records the omega / CA-chirality classes of the heavy-atom model at each step:

    overlay        det of every template-overlay matrix (_overlayPoints: SVD without
                   a reflection check; det = -1 means a mirrored template = D residue)
    placement      template positions before any minimisation
    nearest#0      after the first soft-core minimisation
    nearest#1..10  after each 200-step soft-core dynamics block (C = 0.15..1.5);
                   `kept` marks blocks PDBFixer keeps as the new best state
    after_addMissingAtoms   PDBFixer's final heavy-atom positions (after the last minimisation)
    after_addHydrogens      the model prepare() checks (the stage that raised in P2)

Outputs go to <packet>/diagnostics/prep_stereo_<date>/<sample>/ (trace.json, plus the
prepare() outputs when the model passes); prepared/ is never touched.  CPU only,
Reference platform; nothing is decided here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import peptide_geometry as geo            # noqa: E402  same module object prepare() imports
import prep_peptide_sample as prep        # noqa: E402

FOCUS = ("A:GLU25", "A:GLU25-THR26")      # plus every term of the rebuilt peptide residue 12


def run_one(row, tpl, out_root: Path, seed: int) -> dict:
    import numpy as np
    from openmm import unit
    import pdbfixer.pdbfixer as pf

    plan = prep.plan_sample(row, tpl, PACKET, "model01")
    out = out_root / row["peptide_name"]
    plan["out"] = str(out)
    pc = plan["peptide_chain"]
    focus = lambda lab: lab in FOCUS or lab.startswith(f"{pc}:") and lab.split("-")[-1].endswith("12")

    trace = dict(sample=row["peptide_name"], mutations=plan["mutations"],
                 rebuilt_residues=plan["rebuilt_residues"], seed=seed, overlay=[], stages=[])
    st = dict(top=None, terms=None, ref=None, best=None, n=0)

    key = lambda x: x["label"] + "/" + x["kind"]

    def summarise(name, m, **extra):
        if st["ref"] is None:
            st["ref"] = {key(x): x for x in m}
        paired = [x for x in m if key(x) in st["ref"]]
        trace["stages"].append(dict(
            stage=name, **extra,
            violations=[(v["label"], round(v["deg"], 1)) for v in o_viol(m)],
            focus={x["label"] + ("" if x["kind"] == "chirality" else " omega"): round(x["deg"], 1)
                   for x in m if focus(x["label"])},
            class_changes_vs_placement=geo.class_changes([st["ref"][key(x)] for x in paired], paired)))

    def record(name, xyz_A, **extra):
        summarise(name, geo.measure(st["terms"], xyz_A), **extra)

    o_overlay, o_add, o_near, o_ama = (pf._overlayPoints, pf.PDBFixer._addAtomsToTopology,
                                       pf.PDBFixer._findNearestDistance, pf.PDBFixer.addMissingAtoms)
    o_viol = geo.prepared_violations

    def overlay(points1, points2):
        r = o_overlay(points1, points2)
        loc = sys._getframe(1).f_locals
        if "orientTo" in loc:
            o = loc["orientTo"]
            who = f"new {loc.get('residueName')} oriented on {o.chain.id}:{o.name}{o.id}"
        else:
            res = loc.get("residue")
            who = f"missing atoms of {res.chain.id}:{res.name}{res.id}" if res is not None else "?"
        trace["overlay"].append(dict(target=who, n_points=len(points1),
                                     det=round(float(np.linalg.det(np.asarray(r[1]))), 4)))
        return r

    def add_atoms(self, heavyAtomsOnly, omitUnknownMolecules):
        r = o_add(self, heavyAtomsOnly, omitUnknownMolecules)
        if heavyAtomsOnly and st["top"] is None:
            st["top"] = r[0]
            st["terms"] = geo.stereo_terms(geo.residues_from_topology(r[0]))
            record("placement", np.array([p.value_in_unit(unit.angstrom) for p in r[1]]))
        return r

    def nearest(self, context, newAtoms, cutoff, exclusions):
        d = float(o_near(self, context, newAtoms, cutoff, exclusions))
        xyz = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        n = st["n"]
        kept = n == 0 or d > st["best"]
        st["best"] = d if kept else st["best"]
        record(f"nearest#{n}", xyz[:st["top"].getNumAtoms()], C=round(context.getParameter("C"), 3),
               nearest_nm=round(d, 4), kept=kept)
        st["n"] += 1
        return d

    def add_missing(self, seed=None):
        o_ama(self, seed=seed)
        heavy = [a.index for a in self.topology.atoms() if a.element is not None and a.element.symbol != "H"]
        # PDBFixer's final topology has different atom indices; same residue labels
        terms = geo.stereo_terms(geo.residues_from_topology(self.topology))
        summarise("after_addMissingAtoms", geo.measure(terms, self.positions.value_in_unit(unit.angstrom)),
                  heavy_atoms=len(heavy))

    def viol(measured):
        summarise("after_addHydrogens", measured)
        return o_viol(measured)

    pf._overlayPoints = overlay
    pf.PDBFixer._addAtomsToTopology = add_atoms
    pf.PDBFixer._findNearestDistance = nearest
    pf.PDBFixer.addMissingAtoms = add_missing
    geo.prepared_violations = viol
    t0 = time.monotonic()
    try:
        prep.prepare(plan, prep.TEMPLATE, tpl, "model01", seed)
        trace["prepare_result"] = "passed (outputs in this directory)"
    except RuntimeError as e:
        trace["prepare_result"] = f"raised: {e}"
    finally:
        pf._overlayPoints, pf.PDBFixer._addAtomsToTopology = o_overlay, o_add
        pf.PDBFixer._findNearestDistance, pf.PDBFixer.addMissingAtoms = o_near, o_ama
        geo.prepared_violations = o_viol
    trace["wall_s"] = round(time.monotonic() - t0, 2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    return trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=prep.DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=PACKET / "diagnostics" / f"prep_stereo_{time.strftime('%Y%m%d')}")
    a = ap.parse_args()
    rows = {r["peptide_name"]: r for r in csv.DictReader((PACKET / "sample_structure_audit.csv").open())}
    unknown = set(a.sample) - set(rows)
    if unknown:
        raise SystemExit(f"unknown samples: {sorted(unknown)}")
    tpl = prep.read_template(prep.TEMPLATE)
    for s in a.sample:
        t = run_one(rows[s], tpl, a.out, a.seed)
        print(f"== {s}: {t['prepare_result']}  ({t['wall_s']} s)")
        for o in t["overlay"]:
            if o["det"] < 0:
                print(f"   MIRRORED overlay: {o}")
        for g in t["stages"]:
            extra = {k: g[k] for k in ("C", "nearest_nm", "kept") if k in g}
            print(f"   {g['stage']:<22} {extra} violations={g['violations']} focus={g['focus']}")
        print(f"   trace: {a.out / s / 'trace.json'}", flush=True)


if __name__ == "__main__":
    main()
