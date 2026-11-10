#!/bin/bash
# One end-to-end smoke of the new path (Hou prmtop -> setup -> MD with AsyncPBSAReporter + MPS),
# on 180 (2080 Ti).  Output stays in this smoke package, not in the real rep01.
set -euo pipefail
ROOT=/home/ruigengji/JAXPBSA
PKG=$ROOT/experiments/2026-10-17_small_molecule_benchmark
SMK=$PKG/smoke_hou_online
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python   # 180's local openmm_dev has a CPU-only jaxlib
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_kasuga_hou_smoke CUDA_MPS_LOG_DIRECTORY=/tmp/mps_kasuga_hou_smoke_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control || true' EXIT
cd "$ROOT"
COMMON=(--package "$SMK" --inputs "$PKG/manifests/hou_inputs.csv")
echo "== setup $(date -Iseconds)"
$PY scripts/run_smallmol_md.py setup "${COMMON[@]}" --sample a1 --target hou --model-id ph74_gaff2 --platform CUDA --device-index 0
echo "== run 0.2 ns $(date -Iseconds)"
$PY scripts/run_smallmol_md.py run "${COMMON[@]}" --run-id a1_ph74_gaff2_smoke --platform CUDA --device-index 0 \
    --nvt-ps 10 --npt-ps 100 --prod-ns 0.2 --online --mps-percentage 30
echo "== extend to 0.3 ns (resume: new process, new CSV) $(date -Iseconds)"
$PY scripts/run_smallmol_md.py run "${COMMON[@]}" --run-id a1_ph74_gaff2_smoke --platform CUDA --device-index 0 \
    --nvt-ps 10 --npt-ps 100 --prod-ns 0.3 --online --mps-percentage 30
echo "== done $(date -Iseconds)"
