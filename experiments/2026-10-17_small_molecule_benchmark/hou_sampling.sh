#!/usr/bin/env bash
# Hou bounded sampling (docs/HOU_BOUNDED_SAMPLING_PLAN_20261028.md §2, §5): every sample to 6 replicas x 10 ns.
#   S1  rep01-03 extended in place from 5 to 10 ns (run --prod-ns 10 resumes at chunk 50; a new online segment)
#   S2  rep04-06 (manifests/runs.csv rows added 2026-10-28), same protocol as the first Hou round, 10 ns
# Items: the 21 S1 extensions first, then the 21 S2 replicas; item i goes to worker i % NW.  Each item: run with
# online MM/PBSA (AsyncPBSAReporter, MPS 30 %, priority 1), then strip.  Before the first extension of an S1
# replica its status.json is copied to status_before_extend_20261028.json (kept if it already exists).
#   MODE=worker W=<k> NW=<n> GPU=<id> TAG=<job tag>
# Each worker starts its own MPS daemon for its one GPU (pipe dir /tmp/mps_<TAG>_g<GPU>, sockets only) and quits
# it on exit; if the daemon cannot start, the worker stops (the plan's runs are all MPS 30 %).  A failing item is
# logged as FAIL and the worker goes on.  Resumable: re-run; committed chunks / online segments are skipped.
set -euo pipefail
: "${MODE:?MODE=worker}" "${W:?}" "${NW:?}" "${GPU:?}" "${TAG:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
PY=${PY:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/python}
IN=(--inputs "$P/manifests/hou_inputs.csv")
export CUDA_VISIBLE_DEVICES=$GPU
cd "$J"
exec >> "$P/hou_sampling_w${W}.log" 2>&1
echo "== worker $W/$NW start host $(hostname) $(date -Is) GPU $GPU pid $$"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${TAG}_g${GPU} CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${TAG}_g${GPU}_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || { echo "== MPS daemon did not start; stop"; exit 1; }
trap 'echo quit | nvidia-cuda-mps-control || true' EXIT
echo "== MPS daemon started ($CUDA_MPS_PIPE_DIRECTORY)"
ITEMS=()
for S in a1 a2 a3 a4 a5 a6 a7; do for R in 01 02 03; do ITEMS+=("${S}_ph74_gaff2_rep$R"); done; done
for S in a1 a2 a3 a4 a5 a6 a7; do for R in 04 05 06; do ITEMS+=("${S}_ph74_gaff2_rep$R"); done; done
for (( i = W; i < ${#ITEMS[@]}; i += NW )); do
  RID=${ITEMS[$i]}
  S=${RID%%_ph74_gaff2_rep*}; R=${RID##*_rep}
  OUT=$P/md/hou/$S/ph74_gaff2/rep$R
  if [[ "$R" == 0[123] && -f "$OUT/status.json" && ! -f "$OUT/status_before_extend_20261028.json" ]]; then
    cp "$OUT/status.json" "$OUT/status_before_extend_20261028.json"
  fi
  echo "== [$RID] run $(date -Is)"
  "$PY" scripts/run_smallmol_md.py run "${IN[@]}" --run-id "$RID" --platform CUDA --device-index 0 \
    --prod-ns 10 --online --mps-percentage 30 \
    || { rc=$?; echo "== [$RID] FAIL run rc=$rc $(date -Is)"; continue; }
  echo "== [$RID] strip $(date -Is)"
  "$PY" scripts/run_smallmol_md.py strip "${IN[@]}" --run-id "$RID" \
    || { rc=$?; echo "== [$RID] FAIL strip rc=$rc $(date -Is)"; }
done
echo "== worker $W done $(date -Is)"
