#!/usr/bin/env python3
"""R0/R1 side checks (read-only, numpy + text):

1. minimized_state.xml top solute forces: are they equal/opposite pairs along an
   HBonds-constrained X-H bond (constraint component the minimiser cannot relax)?
2. all 48 v2.1 replicas: diagnostics.csv / scores_1_20 frames.csv / solute_dry.pdb ranges that
   would expose a periodic-image or atom-order error (peptide far from receptor, interaction
   energy near zero).  Reported as ranges, no thresholds.
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import stage_trace as T

OUT = HERE.parents[1] / "r0r1"
samples = [r["peptide_name"] for r in csv.DictReader((T.PACKET / "sample_structure_audit.csv").open())]

# 1. forces
rows = []
for s in samples:
    for m in ("v21m1", "v21m2"):
        prep = T.read_pdb(T.PACKET / "prepared" / s / m / "complex_dry.pdb")
        st = T.read_state(T.PACKET / "md" / s / m / "setup/minimized_state.xml", prep, want_forces=True)
        f = np.linalg.norm(st.forces, axis=1)
        heavy_f = np.where(~prep.is_h, f, 0)
        bonds = T.bonds_from(prep)
        partner = {}
        for a, b in bonds:
            ia, ib = prep.idx[a], prep.idx[b]
            if prep.is_h[ia] != prep.is_h[ib]:
                partner.setdefault(ia, []).append(ib); partner.setdefault(ib, []).append(ia)
        for i in np.argsort(f)[::-1][:6]:
            i = int(i)
            j = max(partner.get(i, [i]), key=lambda k: f[k])
            u = st.xyz[j] - st.xyz[i]; u /= np.linalg.norm(u) if np.linalg.norm(u) > 0 else 1
            fi, fj = st.forces[i], st.forces[j]
            rows.append(dict(sample=s, model=m, atom=prep.label(i), F=round(float(f[i]), 1),
                             XH_partner=prep.label(j) if j != i else "", F_partner=round(float(f[j]), 1),
                             net_pair_force=round(float(np.linalg.norm(fi + fj)), 1),
                             frac_along_bond=round(float(abs(fi @ u) / f[i]), 3)))
        # largest force with constrained X-H pair forces removed (sum over each X-H group)
        grp_net = []
        seen = set()
        for i in range(len(f)):
            if i in seen:
                continue
            g = {i} | set(partner.get(i, [])) if not prep.is_h[i] else {i}
            if prep.is_h[i]:
                continue
            seen |= g
            grp_net.append((float(np.linalg.norm(st.forces[list(g)].sum(0))), prep.label(i)))
        grp_net.sort(reverse=True)
        rows.append(dict(sample=s, model=m, atom="max |sum F| over heavy atom + its H (constraint components cancel)",
                         F=round(grp_net[0][0], 1), XH_partner=grp_net[0][1]))
with (OUT / "minimized_forces_all16.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# 2. imaging scan
scan = []
for s in samples:
    for m in ("v21m1", "v21m2"):
        for r in (1, 2, 3):
            rd = T.PACKET / "md" / s / m / f"rep{r:02d}"
            d = list(csv.DictReader((rd / "analysis/diagnostics.csv").open()))
            pr = np.array([float(x["peptide_CA_rmsd_receptor_frame_A"]) for x in d])
            md_ = np.array([float(x["min_rec_pep_heavy_A"]) for x in d])
            fr = list(csv.DictReader((T.PACKET / "scores_1_20" / s / m / f"rep{r:02d}" / "frames.csv").open()))
            lj = np.array([float(x["e_lj_rl"]) for x in fr]); co = np.array([float(x["e_coul_rl"]) for x in fr])
            tot = np.array([float(x["delta_g_mmpbsa"]) for x in fr])
            dry = T.read_pdb(rd / "analysis/solute_dry.pdb")
            A = np.array([c == "A" for c in dry.chain]) & ~dry.is_h
            B = np.array([c == "B" for c in dry.chain]) & ~dry.is_h
            dmin = float(np.linalg.norm(dry.xyz[B][:, None] - dry.xyz[A][None], axis=2).min())
            frames = [int(x["frame"]) for x in d]
            scan.append(dict(sample=s, model=m, replica=r, diag_rows=len(d),
                             diag_frames_contiguous=frames == list(range(len(d))),
                             diag_pep_CA_rmsd_max=round(pr.max(), 3), diag_pep_CA_rmsd_max_step=round(float(np.abs(np.diff(pr)).max()), 3),
                             diag_min_rec_pep_max=round(md_.max(), 3),
                             score_frames=len(fr), score_frame_first=fr[0]["frame"], score_frame_last=fr[-1]["frame"],
                             e_lj_rl_max=round(lj.max(), 2), e_coul_rl_max=round(co.max(), 2),
                             dG_min=round(tot.min(), 2), dG_max=round(tot.max(), 2),
                             flagged=sum(1 for x in fr if x["flag"]),
                             solute_dry_pdb_min_rec_pep=round(dmin, 3)))
with (OUT / "imaging_scan_48.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(scan[0])); w.writeheader(); w.writerows(scan)
print(json.dumps(dict(
    forces_rows=len(rows),
    max_group_net=[(r["sample"], r["model"], r["F"], r["XH_partner"]) for r in rows if r["atom"].startswith("max")],
    scan_reps=len(scan),
    diag_pep_rmsd_max=max(x["diag_pep_CA_rmsd_max"] for x in scan),
    diag_step_max=max(x["diag_pep_CA_rmsd_max_step"] for x in scan),
    diag_minrecpep_max=max(x["diag_min_rec_pep_max"] for x in scan),
    elj_max=max(x["e_lj_rl_max"] for x in scan), ecoul_max=max(x["e_coul_rl_max"] for x in scan),
    dG_range=(min(x["dG_min"] for x in scan), max(x["dG_max"] for x in scan)),
    flagged=sum(x["flagged"] for x in scan), dry_pdb_min_max=max(x["solute_dry_pdb_min_rec_pep"] for x in scan),
    noncontig=[(x["sample"], x["model"], x["replica"]) for x in scan if not x["diag_frames_contiguous"]],
    score_frames=sorted({(x["score_frames"], x["score_frame_first"], x["score_frame_last"]) for x in scan}),
    diag_rows=sorted({x["diag_rows"] for x in scan})), indent=1))
