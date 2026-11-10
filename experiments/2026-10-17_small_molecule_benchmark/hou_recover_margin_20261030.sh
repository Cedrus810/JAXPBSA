#!/usr/bin/env bash
# Whole-replica offline recovery; original online scores/status/trajectories remain read-only.
# Dry run (default): bash hou_recover_margin_20261030.sh
# Execute on an authorized free GPU: GPU=0 DRY_RUN=0 bash hou_recover_margin_20261030.sh
set -euo pipefail
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
PY=${PY:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/python}
RECOVERY=$P/scores/margin_recovery_20261030/default/hou
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$J"
EXTRA=(--dry-run)
if [[ ${DRY_RUN:-1} == 0 ]]; then
  : "${GPU:?set the authorized physical GPU index}"
  export CUDA_VISIBLE_DEVICES=$GPU
  BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
  [[ -z "$BUSY" ]] || { echo "GPU $GPU has compute processes; stopping"; exit 1; }
  EXTRA=()
fi
for RID in a1_ph74_gaff2_rep05 a3_ph74_gaff2_rep01 a3_ph74_gaff2_rep02; do
  S=${RID%%_ph74_gaff2_rep*}
  R=${RID##*_rep}
  "$PY" scripts/score_smallmol_replica.py --package "$P" --inputs "$P/manifests/hou_inputs.csv" \
    --run-id "$RID" --start-ns 0 --end-ns 10 --stride 1 --block-ns 1 --grid default --pilot window \
    --out "$RECOVERY/$S/ph74_gaff2/rep$R" "${EXTRA[@]}"
done
if [[ ${DRY_RUN:-1} == 0 ]]; then
  "$PY" scripts/summarize_hou_sampling.py --package "$P" --recovery-root "$RECOVERY" \
    --out "$P/reports/hou_sampling_20261030_recovered"
fi
