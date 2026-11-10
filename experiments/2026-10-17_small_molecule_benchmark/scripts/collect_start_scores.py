#!/usr/bin/env python3
"""Collect the B1 start scores into reports/baseline_20261101/start_scores.csv; read-only.

    python3 experiments/2026-10-17_small_molecule_benchmark/scripts/collect_start_scores.py

Plan docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §4.2-4.3.  Input: scores/start_default/<target>/
<sample>/<model>/{frames.csv,summary.json} written by scripts/score_smallmol_start.py (one
minimised start per sample, no replica SE).  Also prints the smoke-vs-full difference for the
samples scored in both (scores/smoke/start_default), which differ only in GPU/host.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
SAMPLES = [("brd4", f"BRD4-{i}", "mobley_amber") for i in range(1, 11)] + \
          [("hou", f"a{i}", "ph74_gaff2") for i in range(1, 8)]
COMP = ["e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand", "delta_g_pb", "delta_g_sa"]


def main():
    out = []
    for t, s, m in SAMPLES:
        d = P / "scores/start_default" / t / s / m
        fr = next(csv.DictReader((d / "frames.csv").open()))
        sm = json.loads((d / "summary.json").read_text())
        out.append(dict(sample=s, target=t, model_id=m, valid=sm["valid"], flag=sm["flag"] or "",
                        delta_g_mmpbsa=fr["delta_g_mmpbsa"], **{k: fr[k] for k in COMP},
                        solver_iters=fr["solver_iters"], margin_A=fr["margin_A"], margin_lig_A=fr["margin_lig_A"],
                        grid="x".join(map(str, sm["sizing"]["grid"])),
                        grid_lig="x".join(map(str, sm["sizing"]["grid_lig"])),
                        h=sm["grid"]["h"], h_lig=sm["grid"]["h_lig"], sizing="start structure, fluctuation_allowance 0",
                        device=sm["device"], host=sm["host"], init_s=sm["init_incl_warmup_s"], ms_score=sm["ms_score"],
                        source=str(d.relative_to(P))))
        smoke = P / "scores/smoke/start_default" / t / s / m / "frames.csv"
        if smoke.exists():
            g0 = float(next(csv.DictReader(smoke.open()))["delta_g_mmpbsa"])
            print(f"{s}: full - smoke = {float(fr['delta_g_mmpbsa']) - g0:+.2e} kcal/mol")
    dst = P / "reports/baseline_20261101/start_scores.csv"
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {dst} ({len(out)} rows, {sum(x['valid'] for x in out)} valid)")


if __name__ == "__main__":
    main()
