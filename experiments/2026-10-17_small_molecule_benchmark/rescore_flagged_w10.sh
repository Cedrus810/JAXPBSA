#!/usr/bin/env bash
# Offline rescoring of the three replicas whose online scores have ligand-box-margin flags in 5-10 ns
# (docs/HOU_BOUNDED_SAMPLING_PLAN_20261028.md §7.3; user 2026-10-30 "是的"): default grid, pilot = all frames of the
# (0, 10] ns window so the ligand box covers every frame.  New --out; online values are kept for comparison.
# Resumable: frames already in frames.csv are skipped.  Run on 183: GPU=0 bash rescore_flagged_w10.sh
set -euo pipefail
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
PY=${PY:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:?} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$J"
exec >> "$P/rescore_flagged_w10.log" 2>&1
echo "== start host $(hostname) $(date -Is) GPU $GPU pid $$"
for RID in a3_ph74_gaff2_rep01 a3_ph74_gaff2_rep02 a1_ph74_gaff2_rep05; do
  S=${RID%%_ph74_gaff2_rep*}; R=${RID##*_rep}
  echo "== [$RID] $(date -Is)"
  "$PY" scripts/score_smallmol_replica.py --inputs "$P/manifests/hou_inputs.csv" --run-id "$RID" --end-ns 10 \
    --out "$P/scores/window10_default/hou/$S/ph74_gaff2/rep$R" || echo "== [$RID] FAIL rc=$? $(date -Is)"
done
echo "== all done $(date -Is)"
