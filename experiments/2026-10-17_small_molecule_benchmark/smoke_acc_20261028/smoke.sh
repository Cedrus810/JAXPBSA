#!/usr/bin/env bash
# One smoke for docs/HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md §4 (183, 2080 Ti + CPU), Hou a1 rep01:
#   grid: new config h0.3 on frame 249 (time, init, GPU memory with preallocation off, sampled by nvidia-smi)
#   amber: pbsa h 0.35 and 0.25 on frame 249, one thread each (time per frame)
set -euo pipefail
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
D=$P/smoke_acc_20261028
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
cd "$J"
echo "== start $(hostname) $(date -Is)"
for h in 0.35 0.25; do
  ( t0=$(date +%s)
    OMP_NUM_THREADS=1 "$PY" scripts/amber_reference_smallmol.py --inputs "$P/manifests/hou_inputs.csv" \
      --run-id a1_ph74_gaff2_rep01 --frames 249 --frame-set q3 --ours online --pbsa-h $h --out "$D/amber_pbsa$h" \
      > "$D/amber_pbsa$h.log" 2>&1 && echo "== amber pbsa $h done $(( $(date +%s) - t0 )) s" \
      || echo "== amber pbsa $h FAIL rc=$? after $(( $(date +%s) - t0 )) s" ) &
done
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 > "$D/gpu_mem.csv" &
MON=$!
t0=$(date +%s)
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" scripts/grid_sensitivity_smallmol.py \
  --inputs "$P/manifests/hou_inputs.csv" --run-id a1_ph74_gaff2_rep01 --frames 249 --frame-set smoke \
  --configs h0.3 --ours online --out "$D/grid_h03" && echo "== grid h0.3 done $(( $(date +%s) - t0 )) s" \
  || echo "== grid h0.3 FAIL rc=$? after $(( $(date +%s) - t0 )) s"
kill $MON
echo "== peak GPU MiB $(cut -d, -f2 "$D/gpu_mem.csv" | tr -dc '0-9\n' | sort -n | tail -1)"
wait
echo "== all done $(date -Is)"
