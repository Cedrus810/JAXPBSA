#!/usr/bin/env bash
# B1 baseline: frozen default MM/PBSA of all 17 minimised starts (BRD4-1..10, Hou a1..a7) on one GPU.
# Plan docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §4.2.  Resumable: rerun skips samples already in frames.csv.
set -euo pipefail
cd /home/ruigengji/JAXPBSA
PY=${PY:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/python}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "== start $(date -Is) $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$PY scripts/score_smallmol_start.py --target brd4 --samples BRD4-1 BRD4-2 BRD4-3 BRD4-4 BRD4-5 BRD4-6 BRD4-7 BRD4-8 BRD4-9 BRD4-10 \
  || echo "FAIL brd4 rc=$?"
$PY scripts/score_smallmol_start.py --target hou --samples a1 a2 a3 a4 a5 a6 a7 || echo "FAIL hou rc=$?"
echo "== done $(date -Is)"
