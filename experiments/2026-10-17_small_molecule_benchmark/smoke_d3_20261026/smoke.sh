#!/usr/bin/env bash
# D3 smoke (docs/BRD4_GRID_DIAGNOSIS_PLAN_20261025.md §3 D3) on 180 (2080 Ti): Hou a1 rep01, q3 frame 124.
# Amber MM/PBSA+GBSA (CPU, 1 thread) and grid sensitivity h0.4 / h0.35 / 3 rotations (GPU) run concurrently;
# our default values come from the replica's committed online CSV (--ours online).  Logs amber.log, grid.log.
set -euo pipefail
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
S=$P/smoke_d3_20261026
PY=/home/canna/miniforge3/envs/openmm_dev/bin/python
C=(--inputs "$P/manifests/hou_inputs.csv" --run-id a1_ph74_gaff2_rep01 --frames 124 --ours online)
cd "$J"
echo "== start host $(hostname) $(date -Is) pid $$" > "$S/smoke.log"
OMP_NUM_THREADS=1 "$PY" scripts/amber_reference_smallmol.py "${C[@]}" --out "$S/amber/a1_rep01" > "$S/amber.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$PY" -u scripts/grid_sensitivity_smallmol.py "${C[@]}" \
    --out "$S/grid/a1_rep01" > "$S/grid.log" 2>&1 &
G=$!
wait $A && echo "== amber rc=0 $(date -Is)" >> "$S/smoke.log" || echo "== amber rc=$? $(date -Is)" >> "$S/smoke.log"
wait $G && echo "== grid rc=0 $(date -Is)" >> "$S/smoke.log" || echo "== grid rc=$? $(date -Is)" >> "$S/smoke.log"
echo "== done $(date -Is)" >> "$S/smoke.log"
