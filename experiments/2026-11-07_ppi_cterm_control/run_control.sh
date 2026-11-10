#!/usr/bin/env bash
# C-terminal start control (docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md §11; user 2026-11-07 "可",
# option "整条肽骨架取自 3LNZ"): PMI and P12Q_L10A, model v21mx1 (peptide backbone of 3LNZ chain H,
# receptor/A25/protocol as v21m1), 3 replicas x 20 ns each, then strip and score 1-20 ns with the
# rules of the 2026-10-11 packet (protocol §2.1: every 10-ps frame, frozen OnlineMMPBSA defaults).
# Unchanged project entry scripts, all pointed at this packet with --packet.
#
# prep (C2) and setup (C3) of both models were done as the smoke on 183 (2080 Ti, 2026-11-07,
# logs/smoke_183_prep_setup.log); here they are skipped because their records exist.
# GPU: if PBS leaves CUDA_VISIBLE_DEVICES unset (yayoi07, project CLAUDE.md 2026-10-18), the
# GPU_SLOT-th GPU with no compute process (default 1st) is taken and exposed alone; device index 0
# inside; if PBS sets a list (yayoi08 2026-11-07: "0,1"), the GPU_SLOT-th entry of it is used.
# Two drivers started together with GPU_SLOT=1 / 2 therefore get different GPUs.
# Resumable: run again; finished stages/chunks/frames are skipped, a replica whose
# diagnostics_summary.json already covers 2000 frames is not stripped again.
#
#   PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python bash run_control.sh      # log: logs/run_control_<SAMPLES>.log
#   PY=... SAMPLES=PMI GPU_SLOT=1 bash run_control.sh & PY=... SAMPLES=P12Q_L10A GPU_SLOT=2 bash run_control.sh & wait
set -euo pipefail
: "${PY:?set PY to the openmm_dev python}"
J=/home/ruigengji/JAXPBSA
N=$J/experiments/2026-11-07_ppi_cterm_control
SAMPLES="${SAMPLES:-PMI P12Q_L10A}"
REPS="${REPS:-01 02 03}"
GPU_SLOT="${GPU_SLOT:-1}"
TAG=$(echo "$SAMPLES" | tr ' ' '_')
M=v21mx1
mkdir -p "$N/logs"
exec >> "$N/logs/run_control_${TAG}.log" 2>&1
cd "$J"

echo "== start host $(hostname) $(date -Is) pid $$ samples $SAMPLES slot $GPU_SLOT OMP ${OMP_NUM_THREADS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | sort -u)
  GPU=""; k=0
  while IFS=, read -r idx uuid; do
    uuid=$(echo "$uuid" | tr -d ' ')
    if [ -z "$BUSY" ] || ! grep -qx "$uuid" <<< "$BUSY"; then
      k=$((k + 1)); if [ "$k" -eq "$GPU_SLOT" ]; then GPU=$(echo "$idx" | tr -d ' '); break; fi
    fi
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  if [ -z "$GPU" ]; then echo "no GPU without compute processes"; exit 1; fi
  export CUDA_VISIBLE_DEVICES=$GPU
elif [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  # PBS already lists the allocated GPUs (yayoi08 2026-11-07: "0,1"); take the GPU_SLOT-th of
  # them so that two drivers do not share one card (job 24803 ran both on GPU 0)
  IFS=, read -r -a LIST <<< "$CUDA_VISIBLE_DEVICES"
  if [ "$GPU_SLOT" -gt "${#LIST[@]}" ]; then echo "GPU_SLOT $GPU_SLOT > ${#LIST[@]} GPUs in $CUDA_VISIBLE_DEVICES"; exit 1; fi
  export CUDA_VISIBLE_DEVICES="${LIST[$((GPU_SLOT - 1))]}"
fi
echo "== using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for S in $SAMPLES; do
  echo "== [$S/$M] prep + setup (skip when done) $(date -Is)"
  "$PY" scripts/prep_peptide_sample.py --packet "$N" --sample "$S" --model-id "$M" --protocol v2.1
  "$PY" scripts/run_peptide_md.py setup --packet "$N" --sample "$S" --model-id "$M" --platform CUDA --device-index 0
done

for S in $SAMPLES; do
  for R in $REPS; do
    RID="${S}_${M}_rep${R}"
    REP="$N/md/$S/$M/rep$R"
    echo "== [$RID] run $(date -Is)"
    "$PY" scripts/run_peptide_md.py run --packet "$N" --run-id "$RID" --platform CUDA --device-index 0 \
        --prod-ns 20 --dcd-ps 10
    if grep -qs '"frames_total": 2000' "$REP/analysis/diagnostics_summary.json"; then
      echo "== [$RID] strip already covers 2000 frames"
    else
      echo "== [$RID] strip $(date -Is)"
      "$PY" scripts/strip_peptide_replica.py --packet "$N" --run-id "$RID" --diag-stride 1 --dry-stride 1
    fi
  done
done

for S in $SAMPLES; do
  for R in $REPS; do
    RID="${S}_${M}_rep${R}"
    echo "== [$RID] score 1-20 ns $(date -Is)"
    "$PY" scripts/score_peptide_replica.py --packet "$N" --run-id "$RID" --start-ns 1 --end-ns 20 \
        --stride 1 --out "$N/scores_1_20/$S/$M/rep$R"
  done
done
echo "== all done $(date -Is)"
