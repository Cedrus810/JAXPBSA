#!/usr/bin/env bash
# One smoke for docs/HOU_BOUNDED_SAMPLING_PLAN_20261028.md §4 (183, 2080 Ti): extend the finished online
# replica acconline (1 ns) to 1.2 ns in place with --online, then strip.  Production replicas are not touched.
set -euo pipefail
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
A=$P/acceptance_online_20261028
R=$A/md/hou/a1/ph74_gaff2/acconline
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_smokeext CUDA_MPS_LOG_DIRECTORY=/tmp/mps_smokeext_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
cd "$J"
echo "== start $(hostname) $(date -Is)"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control || true' EXIT
cp -n "$R/status.json" "$R/status_before_extend_smoke.json"
"$PY" scripts/run_smallmol_md.py strip --package "$A" --inputs "$P/manifests/hou_inputs.csv" --run-id a1_ph74_gaff2_acconline
cp "$R/analysis/frame_times.csv" "$R/frame_times_before_extend_smoke.csv"
"$PY" scripts/run_smallmol_md.py run --package "$A" --inputs "$P/manifests/hou_inputs.csv" --run-id a1_ph74_gaff2_acconline \
  --platform CUDA --device-index 0 --prod-ns 1.2 --online --mps-percentage 30
"$PY" scripts/run_smallmol_md.py strip --package "$A" --inputs "$P/manifests/hou_inputs.csv" --run-id a1_ph74_gaff2_acconline
echo "== all done $(date -Is)"
