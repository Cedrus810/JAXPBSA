#!/usr/bin/env bash
# B1 minimisation sensitivity (docs/P1_P2_SESSION_HANDOFF_20261101.md §6): redo the unrestrained setup
# minimisation of all 17 starts on another GPU into an independent package, then score each start with the
# frozen default grid; compare with the official B1 (reports/baseline_20261101_v2).  Existing entries only
# (run_smallmol_md.py setup --package, score_smallmol_start.py --package); nothing in the main package changes.
# Resumable: finished setups/scores are skipped.
set -euo pipefail
cd /home/ruigengji/JAXPBSA
PY=${PY:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/python}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-0}
E=experiments/2026-10-17_small_molecule_benchmark
K=$E/b1_minimisation_repeat_20261101
F=$E/brd4_5_alprazolam_20261101
echo "== start $(date -Is) $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for S in BRD4-1 BRD4-2 BRD4-3 BRD4-4 BRD4-6 BRD4-7 BRD4-8 BRD4-9 BRD4-10; do
  $PY scripts/run_smallmol_md.py setup --package $K --inputs $E/manifests/brd4_inputs.csv --sample $S --target brd4 \
    --model-id mobley_amber --device-index 0 || echo "FAIL setup $S rc=$?"
done
$PY scripts/run_smallmol_md.py setup --package $K --inputs $F/manifests/brd4_inputs.csv --sample BRD4-5 --target brd4 \
  --model-id alprazolam_gaff_fix --device-index 0 || echo "FAIL setup BRD4-5 rc=$?"
for S in a1 a2 a3 a4 a5 a6 a7; do
  $PY scripts/run_smallmol_md.py setup --package $K --inputs $E/manifests/hou_inputs.csv --sample $S --target hou \
    --model-id ph74_gaff2 --device-index 0 || echo "FAIL setup $S rc=$?"
done
$PY scripts/score_smallmol_start.py --package $K --inputs $E/manifests/brd4_inputs.csv --target brd4 \
  --samples BRD4-1 BRD4-2 BRD4-3 BRD4-4 BRD4-6 BRD4-7 BRD4-8 BRD4-9 BRD4-10 || echo "FAIL score brd4 rc=$?"
$PY scripts/score_smallmol_start.py --package $K --inputs $F/manifests/brd4_inputs.csv --target brd4 \
  --model-id alprazolam_gaff_fix --samples BRD4-5 || echo "FAIL score BRD4-5 rc=$?"
$PY scripts/score_smallmol_start.py --package $K --inputs $E/manifests/hou_inputs.csv --target hou \
  --samples a1 a2 a3 a4 a5 a6 a7 || echo "FAIL score hou rc=$?"
echo "== done $(date -Is)"
