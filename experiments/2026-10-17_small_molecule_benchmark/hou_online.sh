#!/usr/bin/env bash
# Hou alpha-thrombin a1-a7 first round with ONLINE MM/PBSA (docs/BRD4_HOU_EXECUTION_HANDOFF.md
# §3.1, §4.3; ONLINE_PLAN.md; docs/ONLINE_SCHEDULING.md): 7 entries x 3 replicas x 5 ns
# (manifests/runs.csv, target hou), inputs from scripts/prepare_hou_thrombin.py
# (manifests/hou_inputs.csv).  The frozen OnlineMMPBSA scores every saved frame (10 ps) during MD
# through AsyncPBSAReporter on the same GPU, MPS thread limit 30 %, priority 1.
#   MODE=launch  [NGPU=n]          pick up to n GPUs without compute processes, setup, then workers
#   MODE=setup   GPU=<id>          setup of a1..a7 in turn (finished ones are skipped)
#   MODE=worker  W=<k> NW=<n> GPU=<id>   Hou replicas with (hou row index) % NW == W
# Each worker starts its own MPS daemon for its one GPU (pipe dir /tmp/mps_<job>_g<id>, sockets
# only) and quits it on exit; if the daemon cannot start, the worker runs plain async
# (--mps-percentage 0) and says so in its log.  A replica that fails is logged as FAIL and the
# worker moves on (handoff §1).  Resumable: re-run; finished stages/chunks are skipped, a resumed
# production writes a new online/pbsa_from_chunk_NNNN.csv.
set -euo pipefail
: "${MODE:?MODE=launch|setup|worker}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
PY=${PY:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/python}
IN=(--inputs "$P/manifests/hou_inputs.csv")
JOB=${PBS_JOBID:-local$$}
cd "$J"

if [[ "$MODE" == launch ]]; then
  echo "job $JOB host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} $(date -Is)"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
  BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  GPUS=()
  while IFS=, read -r idx uuid; do
    uuid=${uuid// /}
    [[ "$BUSY" == *"$uuid"* ]] || GPUS+=("$idx")
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  GPUS=("${GPUS[@]:0:${NGPU:-4}}")
  echo "GPUs without compute processes, using: ${GPUS[*]:-none}"
  (( ${#GPUS[@]} > 0 )) || { echo "no free GPU"; exit 1; }
  MODE=setup GPU=${GPUS[0]} bash "$P/hou_online.sh" || { echo "setup failed, see hou_online_setup.log"; exit 1; }
  NW=${#GPUS[@]}
  for (( w = 0; w < NW; w++ )); do
    MODE=worker W=$w NW=$NW GPU=${GPUS[$w]} bash "$P/hou_online.sh" &
  done
  wait
  echo "all workers returned $(date -Is)"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU:?}"     # this process (and its PBSA worker) sees one GPU
if [[ "$MODE" == setup ]]; then
  exec >> "$P/hou_online_setup.log" 2>&1
  echo "== setup start host $(hostname) $(date -Is) GPU $GPU"
  for S in a1 a2 a3 a4 a5 a6 a7; do
    echo "== [$S] setup $(date -Is)"
    "$PY" scripts/run_smallmol_md.py setup "${IN[@]}" --sample "$S" --target hou --model-id ph74_gaff2 \
      --platform CUDA --device-index 0
  done
  echo "== setup done $(date -Is)"
  exit 0
fi

: "${W:?}" "${NW:?}"
exec >> "$P/hou_online_w${W}.log" 2>&1
echo "== worker $W/$NW start host $(hostname) $(date -Is) GPU $GPU pid $$"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${JOB}_g${GPU} CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${JOB}_g${GPU}_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS=30
if nvidia-cuda-mps-control -d; then
  trap 'echo quit | nvidia-cuda-mps-control || true' EXIT
  echo "== MPS daemon started ($CUDA_MPS_PIPE_DIRECTORY)"
else
  MPS=0
  echo "== MPS daemon did not start: plain async worker (--mps-percentage 0)"
fi
mapfile -t RIDS < <(awk -F, -v w="$W" -v nw="$NW" 'NR > 1 && $2 == "hou" {if (k++ % nw == w) print $1}' "$P/manifests/runs.csv")
echo "== worker $W replicas: ${RIDS[*]}"
for RID in "${RIDS[@]}"; do
  echo "== [$RID] run $(date -Is)"
  "$PY" scripts/run_smallmol_md.py run "${IN[@]}" --run-id "$RID" --platform CUDA --device-index 0 \
    --prod-ns 5 --online --mps-percentage "$MPS" \
    || { rc=$?; echo "== [$RID] FAIL run rc=$rc $(date -Is)"; continue; }
  echo "== [$RID] strip $(date -Is)"
  "$PY" scripts/run_smallmol_md.py strip "${IN[@]}" --run-id "$RID" \
    || { rc=$?; echo "== [$RID] FAIL strip rc=$rc $(date -Is)"; }
done
echo "== worker $W done $(date -Is)"
