#!/usr/bin/env bash
# Hou same-card online cost and recovery acceptance (docs/HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md §4).
# One GPU, three stages in series so that nothing else runs during the timings:
#   A  scripts/online_overhead.py on Hou a1 rep01 (5000-step interval = 10 ps): baseline at start / middle /
#      end; for grid default and h0.4: sync and plain async (both orders), async + MPS 30 % priority 1
#      (both orders).  One fresh XLA cache directory per job: the first process of each grid is the cold start.
#   B  run_smallmol_md.py in the separate package acceptance_online_20261028/: accbase (MD only, 1 ns) and
#      acconline (--online --mps-percentage 30, 1 ns) interrupted by SIGTERM to its own PID once
#      chunks_done >= 2, re-run with the same command, then run once more (must skip).
#   C  score_smallmol_replica.py --pilot npt --stride 20 on rep01 of a1..a7 (online-pilot replay, 25 frames each).
#   MODE=full  GPU=<id> TAG=<job tag>      the plan's run
#   MODE=smoke GPU=<id> TAG=<tag>          A 10 000 steps (baseline, sync default, async+MPS h0.4), B accsmoke
#                                          (NPT 100 ps, prod 0.2 ns, no interruption), C 5 frames of a1 rep01
# The MPS daemon is started for the job's one GPU (pipe dir /tmp/mps_<TAG>, sockets only) and quit on exit.
# Resumable: B and C skip finished chunks / frames; A has no skip (each configuration is a timing; re-running
# A re-measures it).
set -euo pipefail
: "${MODE:?MODE=full|smoke}" "${GPU:?}" "${TAG:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
A=$P/acceptance_online_20261028
PY=${PY:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/python}
IN=(--inputs "$P/manifests/hou_inputs.csv")
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
LOG=$A/${MODE}_${TAG}.log
CACHE=$A/xla_cache_${TAG}
mkdir -p "$CACHE"
cd "$J"
exec >> "$LOG" 2>&1
echo "== $MODE start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=$GPU pid $$"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${TAG} CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${TAG}_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || { echo "== MPS daemon did not start; stop"; exit 1; }
trap 'echo quit | nvidia-cuda-mps-control || true' EXIT
echo "== MPS daemon started ($CUDA_MPS_PIPE_DIRECTORY)"

ovh() {   # one online_overhead.py call; a failure is logged and the stage goes on
  echo "== A $* $(date -Is)"
  "$PY" scripts/online_overhead.py --hou-run-id a1_ph74_gaff2_rep01 --steps "$STEPS" --cache-dir "$CACHE" "$@" \
    || echo "== A FAIL rc=$? $* $(date -Is)"
}
MPS=(--mode async --mps-percentage 30 --mps-priority 1)

if [[ "$MODE" == smoke ]]; then
  STEPS=10000
  ovh --config baseline
  ovh --config 5000 --mode sync --grid default
  ovh --config 5000 "${MPS[@]}" --grid h0.4
  echo "== B accsmoke $(date -Is)"
  "$PY" scripts/run_smallmol_md.py run --package "$A" "${IN[@]}" --run-id a1_ph74_gaff2_accsmoke \
    --platform CUDA --device-index 0 --npt-ps 100 --prod-ns 0.2 --online --mps-percentage 30 \
    || echo "== B FAIL rc=$? $(date -Is)"
  echo "== C replay smoke $(date -Is)"
  "$PY" scripts/score_smallmol_replica.py "${IN[@]}" --run-id a1_ph74_gaff2_rep01 --pilot npt --end-ns 0.05 \
    --out "$A/replay_smoke_${TAG}" || echo "== C FAIL rc=$? $(date -Is)"
  echo "== smoke all done $(date -Is)"
  exit 0
fi

STEPS=200000
ovh --config baseline
for G in default h0.4; do
  ovh --config 5000 --mode both --grid "$G"
  ovh --config 5000 "${MPS[@]}" --grid "$G"
  [[ "$G" == default ]] && ovh --config baseline
done
ovh --config baseline

RUN=("$PY" scripts/run_smallmol_md.py run --package "$A" "${IN[@]}" --platform CUDA --device-index 0 --prod-ns 1)
echo "== B accbase $(date -Is)"
"${RUN[@]}" --run-id a1_ph74_gaff2_accbase || echo "== B FAIL accbase rc=$? $(date -Is)"
ST=$A/md/hou/a1/ph74_gaff2/acconline/status.json
ONL=(--run-id a1_ph74_gaff2_acconline --online --mps-percentage 30)
if [[ -f "$ST" ]] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$ST')).get('chunks_done', 0) >= 2 else 1)"; then
  echo "== B acconline already past the interruption point; resume only"
else
  echo "== B acconline run 1 (to be interrupted) $(date -Is)"
  "${RUN[@]}" "${ONL[@]}" &
  MD=$!
  echo "== B acconline pid $MD"
  until [[ -f "$ST" ]] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$ST')).get('chunks_done', 0) >= 2 else 1)" 2>/dev/null; do
    kill -0 "$MD" 2>/dev/null || break
    sleep 5
  done
  if kill -0 "$MD" 2>/dev/null; then
    ps -o pid,etime,cmd -p "$MD"
    kill -TERM "$MD"
    echo "== B acconline SIGTERM sent to $MD $(date -Is)"
  fi
  wait "$MD" && echo "== B acconline run 1 rc=0 (not interrupted?)" || echo "== B acconline run 1 rc=$?"
fi
echo "== B acconline resume $(date -Is)"
"${RUN[@]}" "${ONL[@]}" || echo "== B FAIL acconline resume rc=$? $(date -Is)"
echo "== B acconline re-run (must skip) $(date -Is)"
"${RUN[@]}" "${ONL[@]}" || echo "== B FAIL acconline re-run rc=$? $(date -Is)"

for S in a1 a2 a3 a4 a5 a6 a7; do
  echo "== C replay ${S} rep01 $(date -Is)"
  "$PY" scripts/score_smallmol_replica.py "${IN[@]}" --run-id "${S}_ph74_gaff2_rep01" --pilot npt --stride 20 \
    || echo "== C FAIL $S rc=$? $(date -Is)"
done
echo "== full all done $(date -Is)"
