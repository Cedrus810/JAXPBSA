#!/usr/bin/env python3
"""Write manifests/runs.csv: replica identities, seeds and output directories (handoff §4.3:
three independent replicas per entry).  Row i gets integrator base 10_000_000 + 1_000_000*i + 1
and barostat base = integrator + 500_000; run_smallmol_md.py derives per-context seeds as
base + 7919*k (k <= 52 for 5 ns in 100-ps chunks -> < 420k), so rows never overlap.
Velocity seed 20261018 + i.  Hou rows will be appended with i continuing after BRD4.
Re-running rewrites the same rows (deterministic); do not renumber rows that have run.
"""
import csv
from pathlib import Path

P = Path(__file__).resolve().parents[1]
rows = []
for s in range(1, 11):
    for rep in (1, 2, 3):
        i = len(rows)
        integ = 10_000_000 + 1_000_000 * i + 1
        sample, model = f"BRD4-{s}", "mobley_amber"
        rows.append(dict(run_id=f"{sample}_{model}_rep{rep:02d}", target="brd4", sample_id=sample, model_id=model,
                         replica=rep, velocity_seed=20261018 + i, integrator_seed=integ, barostat_seed=integ + 500_000,
                         output_directory=f"md/brd4/{sample}/{model}/rep{rep:02d}"))
for s in range(1, 8):           # Hou alpha-thrombin a1-a7, scripts/prepare_hou_thrombin.py
    for rep in (1, 2, 3):
        i = len(rows)
        integ = 10_000_000 + 1_000_000 * i + 1
        sample, model = f"a{s}", "ph74_gaff2"
        rows.append(dict(run_id=f"{sample}_{model}_rep{rep:02d}", target="hou", sample_id=sample, model_id=model,
                         replica=rep, velocity_seed=20261018 + i, integrator_seed=integ, barostat_seed=integ + 500_000,
                         output_directory=f"md/hou/{sample}/{model}/rep{rep:02d}"))
with (P / "manifests/runs.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {P / 'manifests/runs.csv'}")
