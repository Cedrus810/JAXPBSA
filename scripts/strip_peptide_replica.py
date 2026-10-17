#!/usr/bin/env python3
"""Dry trajectory + structural diagnostics for one packet replica (P1/P2 entry).

    python scripts/strip_peptide_replica.py --run-id PMI_model01_rep01 --dry-run
    python scripts/strip_peptide_replica.py --run-id PMI_model01_rep01 --diag-stride 10

Uses only production chunks that `run_peptide_md.py` has marked complete in
status.json.  Writes into `<replica>/analysis/`:

    solute_dry.dcd + solute_dry.pdb   receptor+peptide, peptide imaged next to the
                                      receptor; input for scoring (stride --dry-stride)
    diagnostics.csv                   per sampled frame (--diag-stride): omega cis/trans and
                                      CA L/D class changes vs the prepared model, receptor Cα RMSD
                                      to the prepared model, peptide Cα RMSD and residue-12
                                      Cα shift in the receptor frame, min receptor–peptide
                                      heavy-atom distance, nearest water O to each of the
                                      four 3EQS interface water sites (protocol §1.2), and the
                                      A25-A26 / B11-B12 junction bond and angles (prep v2
                                      design §5 C4; frames with a term outside 4 sigma are
                                      counted, not rejected)
    diagnostics_summary.json          quantiles of the above + T/density from chunk logs

Everything is reported, nothing is accepted or rejected here; dissociation,
drift and empty water sites are recorded for the P2 decision (plan §5.2).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
TEMPLATE = ROOT / "experiments/2026-10-10_peptide_samples/raw/mdm2_3EQS.pdb"
WATER_SITES = ("A:HOH1", "A:HOH185", "A:HOH186", "B:HOH74")
JUNCTIONS = ((("A", 25), ("A", 26)), (("B", 11), ("B", 12)))
JUNCTION_COLS = ("A25-A26 C-N", "A25-A26 O-C-N", "A25-A26 C-N-CA", "B11-B12 C-N", "B11-B12 O-C-N", "B11-B12 C-N-CA")


def site_coords(template: Path) -> dict:
    out = {}
    for ln in template.read_text().splitlines():
        if ln.startswith("HETATM") and ln[17:20] == "HOH":
            key = f"{ln[21]}:HOH{int(ln[22:26])}"
            if key in WATER_SITES:
                out[key] = [float(ln[30:38]) / 10, float(ln[38:46]) / 10, float(ln[46:54]) / 10]  # nm
    missing = set(WATER_SITES) - set(out)
    if missing:
        raise RuntimeError(f"water sites not in template: {missing}")
    return out


def kabsch(P, Q):
    """R, t with R @ p + t ≈ q (rows of P onto rows of Q)."""
    import numpy as np
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=PACKET)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--diag-stride", type=int, default=10, help="frames between diagnostic samples")
    ap.add_argument("--dry-stride", type=int, default=1, help="frames between saved dry frames")
    ap.add_argument("--dry-run", action="store_true", help="print inputs/outputs; no mdtraj import")
    a = ap.parse_args()

    row = next((r for r in csv.DictReader((a.packet / "proposed_runs.csv").open()) if r["run_id"] == a.run_id), None)
    if row is None:
        raise SystemExit(f"run_id {a.run_id} not in proposed_runs.csv")
    out = Path(row["output_directory"])
    peptide_name, model_id = out.parent.parent.name, out.parent.name
    prep = a.packet / "prepared" / peptide_name / model_id
    setup = a.packet / "md" / peptide_name / model_id / "setup"
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
    chunks = [] if status is None else [out / "prod" / f"chunk_{c:04d}.dcd" for c in range(status["chunks_done"])]
    plan = dict(run_id=a.run_id, prepared=str(prep), setup=str(setup), replica=str(out),
                chunks_complete=len(chunks), outputs=str(out / "analysis"), water_sites=site_coords(a.template))
    if a.dry_run:
        print(json.dumps(plan)); return
    if not chunks:
        raise SystemExit(f"no completed production chunks in {out}")

    import numpy as np
    import mdtraj as md
    prep_rec = json.loads((prep / "prep_record.json").read_text())
    ref = np.load(prep / "parameters.npz")
    n_sol = prep_rec["n_atoms"]
    top = md.load_topology(str(setup / "solvated.pdb"))
    names = [x.name for x in list(top.atoms)[:n_sol]]
    if names != list(ref["atom_name"]):
        raise RuntimeError("solvated topology atom order differs from prepared solute")
    rec_idx, lig_idx = ref["receptor_idx"], ref["ligand_idx"]
    ca = np.array([i for i in range(n_sol) if ref["atom_name"][i] == "CA"])
    rec_ca = np.intersect1d(ca, rec_idx)
    lig_ca = np.intersect1d(ca, lig_idx)
    lig_ca12 = lig_ca[-1]
    heavy = np.array([i for i in range(n_sol) if not str(ref["atom_name"][i]).startswith("H")])
    rec_h, lig_h = np.intersect1d(heavy, rec_idx), np.intersect1d(heavy, lig_idx)
    water_o = top.select("water and name O")
    ref_nm = ref["coords_A"] / 10.0
    sites = np.array(list(plan["water_sites"].values()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import peptide_geometry as geo
    res, xyz_prep = geo.residues_from_pdb(prep / "complex_dry.pdb")
    terms = geo.stereo_terms(res)
    stereo_ref = geo.measure(terms, xyz_prep)
    stereo_changed = {}

    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    sol_top = top.subset(range(n_sol))
    rec_set = {a.index for a in sol_top.atoms if a.index in set(rec_idx.tolist())}
    anchors = [m for m in sol_top.find_molecules() if any(x.index in rec_set for x in m)]
    rows, n_dry, frame = [], 0, 0
    dcd_out = md.formats.DCDTrajectoryFile(str(adir / "solute_dry.dcd"), "w")
    first = None
    for path in chunks:
        for tr in md.iterload(str(path), top=top, chunk=500):
            for k in range(tr.n_frames):
                f = frame + k
                if f % a.diag_stride == 0:
                    X = tr.xyz[k]
                    R, t = kabsch(X[rec_ca], ref_nm[rec_ca])
                    Y = X[:n_sol] @ R.T + t
                    rmsd = lambda idx: float(np.sqrt(((Y[idx] - ref_nm[idx]) ** 2).sum(1).mean()) * 10)
                    d = np.linalg.norm(Y[lig_h][:, None] - Y[rec_h][None], axis=2).min() * 10
                    # sites into the raw frame, then minimum-image distance to water O
                    Rinv = R.T
                    s_raw = (sites - t) @ Rinv.T
                    box = tr.unitcell_lengths[k]
                    dw = X[water_o][None] - s_raw[:, None]
                    dw -= box * np.round(dw / box)
                    near = np.linalg.norm(dw, axis=2).min(1) * 10
                    ch = geo.class_changes(stereo_ref, geo.measure(terms, X[:n_sol]))
                    jv, jf = geo.junction_report(res, X[:n_sol] * 10.0, JUNCTIONS)
                    for c in ch:
                        stereo_changed[c["label"]] = stereo_changed.get(c["label"], 0) + 1
                    rows.append(dict(frame=f, time_ps=float(tr.time[k]), stereo_class_changes=len(ch),
                                     receptor_CA_rmsd_A=rmsd(rec_ca),
                                     peptide_CA_rmsd_receptor_frame_A=rmsd(lig_ca),
                                     peptide_CA12_shift_A=float(np.linalg.norm(Y[lig_ca12] - ref_nm[lig_ca12]) * 10),
                                     min_rec_pep_heavy_A=float(d),
                                     **{f"nearest_waterO_{s.replace(':', '_')}_A": float(v) for s, v in zip(WATER_SITES, near)},
                                     **{"junction_" + c.replace(" ", "_"): jv[c] for c in JUNCTION_COLS},
                                     junction_terms_outside_4sigma=len(jf)))
            sl = tr.atom_slice(range(n_sol))
            sl.image_molecules(inplace=True, anchor_molecules=anchors)
            keep = [k for k in range(sl.n_frames) if (frame + k) % a.dry_stride == 0]
            if keep:
                sub = sl[keep]
                dcd_out.write(sub.xyz * 10, cell_lengths=sub.unitcell_lengths * 10, cell_angles=sub.unitcell_angles)
                n_dry += len(keep)
                if first is None:
                    first = sub[0]
            frame += tr.n_frames
    dcd_out.close()
    first.save_pdb(str(adir / "solute_dry.pdb"))

    with (adir / "diagnostics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    q = lambda key: {p: round(float(np.percentile([r[key] for r in rows], p)), 3) for p in (5, 50, 95)}
    logs = {}
    for c in range(len(chunks)):
        for ln in (out / "prod" / f"chunk_{c:04d}.log").read_text().splitlines()[1:]:
            parts = ln.split(",")
            if len(parts) >= 6:
                logs.setdefault("T_K", []).append(float(parts[3])); logs.setdefault("density", []).append(float(parts[5]))
    summary = dict(run_id=a.run_id, frames_total=frame, chunks_used=len(chunks), dry_frames=n_dry,
                   dry_stride=a.dry_stride, diag_stride=a.diag_stride, n_solute_atoms=n_sol,
                   quantiles_p5_p50_p95={k: q(k) for k in rows[0] if k not in ("frame", "time_ps", "stereo_class_changes",
                                                                                  "junction_terms_outside_4sigma")},
                   junction_frames_outside_4sigma=sum(r["junction_terms_outside_4sigma"] > 0 for r in rows),
                   log_means={k: round(float(np.mean(v)), 4) for k, v in logs.items()},
                   stereo_terms=len(terms),
                   stereo_class_changes_frames_by_term=stereo_changed,
                   note="diagnostic report only; no acceptance thresholds applied")
    (adir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
