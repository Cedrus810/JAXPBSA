#!/usr/bin/env bash
# R4 numerical checks and baselines (r4_protocol.md §2): one worker per role.
#   ROLE=gpu GPU_SLOT=1 PART=1 NPART=2  N4 (16 models, first part only) then N1 for its share of the 42
#                                       replicas of the 14 models (P12Q_L10A excluded, user 2026-11-08)
#   ROLE=cpu NPROC=16                   N2/N3 Amber for all 48 replicas (P12Q included so that the
#                                       8-candidate baseline set is complete; r4_protocol.md §6.1)
# GPU choice as experiments/2026-11-07_ppi_cterm_control/run_control.sh: if CUDA_VISIBLE_DEVICES lists
# several cards take the GPU_SLOT-th, if unset take the GPU_SLOT-th card without compute processes.
# Resumable: every entry skips finished rows/configs/models; Amber replicas with a record.json are skipped.
#   PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python ROLE=gpu GPU_SLOT=1 PART=1 NPART=2 bash r4_run.sh
set -euo pipefail
: "${PY:?set PY to the openmm_dev python}"
: "${ROLE:?gpu or cpu}"
J=/home/ruigengji/JAXPBSA
R=$J/experiments/2026-11-08_ppi_r4
P=$J/experiments/2026-10-11_public_data_execution_packet
SAMPLES14="PMI PMI-L10A T1L_L10A A4L_L10A N8A_L10A S11M_L10A PMI-M4"
TAG="${ROLE}${GPU_SLOT:-}"
mkdir -p "$R/logs"
exec >> "$R/logs/r4_${TAG}.log" 2>&1
cd "$J"
echo "== start $ROLE host $(hostname) $(date -Is) pid $$ OMP ${OMP_NUM_THREADS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "$ROLE" = cpu ]; then
  NPROC="${NPROC:-16}"
  ids=()
  for S in $SAMPLES14 P12Q_L10A; do for M in v21m1 v21m2; do for X in 01 02 03; do ids+=("${S}_${M}_rep${X}"); done; done; done
  printf '%s\n' "${ids[@]}" | xargs -P "$NPROC" -I{} bash -c '
    rid={}; s=${rid%_v21m*}; rest=${rid#${s}_}; m=${rest%_rep*}; r=rep${rid##*_rep}
    if [ -s "'"$R"'/n2n3_amber/$s/$m/$r/record.json" ]; then echo "== [$rid] amber done"; exit 0; fi
    echo "== [$rid] amber $(date -Is)"
    OMP_NUM_THREADS=1 "'"$PY"'" "'"$R"'/scripts/r4_amber_peptide.py" --run-id "$rid" || echo "== [$rid] amber FAILED"'
  echo "== cpu all done $(date -Is)"
  exit 0
fi

# GPU role
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
GPU_SLOT="${GPU_SLOT:-1}"
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
  IFS=, read -r -a LIST <<< "$CUDA_VISIBLE_DEVICES"
  export CUDA_VISIBLE_DEVICES="${LIST[$((GPU_SLOT - 1))]}"
fi
echo "== using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
PART="${PART:-1}"; NPART="${NPART:-1}"
if [ "$PART" -eq 1 ]; then
  echo "== N4 start baselines $(date -Is)"
  "$PY" "$R/scripts/r4_start_baselines.py" --all
fi
i=0
for S in $SAMPLES14; do for M in v21m1 v21m2; do for X in 01 02 03; do
  i=$((i + 1)); [ $(( (i - 1) % NPART + 1 )) -eq "$PART" ] || continue
  RID="${S}_${M}_rep${X}"
  echo "== [$RID] N1 $(date -Is)"
  "$PY" "$R/scripts/r4_grid_peptide.py" --run-id "$RID"
done; done; done
echo "== gpu part $PART all done $(date -Is)"
