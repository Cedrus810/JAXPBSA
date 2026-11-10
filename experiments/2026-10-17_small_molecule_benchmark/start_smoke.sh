#!/usr/bin/env bash
# B1 start-scoring smoke on 183 (2080 Ti): BRD4-3 and a1, default grid, then the same commands again (must skip).
set -euo pipefail
cd /home/ruigengji/JAXPBSA
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
OUT=experiments/2026-10-17_small_molecule_benchmark/scores/smoke/start_default
for pass in 1 2; do
  echo "== pass $pass $(date -Is)"
  $PY scripts/score_smallmol_start.py --target brd4 --samples BRD4-3 --out-root $OUT
  $PY scripts/score_smallmol_start.py --target hou --samples a1 --out-root $OUT
done
echo "== done $(date -Is)"
