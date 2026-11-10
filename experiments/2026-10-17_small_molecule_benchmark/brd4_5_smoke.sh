#!/usr/bin/env bash
# BRD4-5 alprazolam re-prep smoke on 183 (docs/BRD4_5_REPREP_PLAN_20261101.md §5): prep -> setup -> B1 start score.
set -euo pipefail
cd /home/ruigengji/JAXPBSA
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-0}
PKG=experiments/2026-10-17_small_molecule_benchmark/brd4_5_alprazolam_20261101_smoke
echo "== prep $(date -Is) $(hostname)"
$PY experiments/2026-10-17_small_molecule_benchmark/scripts/prepare_brd4_5_alprazolam.py --out $PKG
echo "== setup $(date -Is)"
$PY scripts/run_smallmol_md.py setup --package $PKG --inputs $PKG/manifests/brd4_inputs.csv \
  --sample BRD4-5 --target brd4 --model-id alprazolam_gaff_fix --device-index 0
echo "== start score $(date -Is)"
$PY scripts/score_smallmol_start.py --package $PKG --inputs $PKG/manifests/brd4_inputs.csv \
  --target brd4 --model-id alprazolam_gaff_fix --samples BRD4-5
echo "== done $(date -Is)"
