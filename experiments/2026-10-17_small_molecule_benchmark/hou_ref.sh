#!/usr/bin/env bash
# Same-frame reference for the 21 Hou replicas (D3 of docs/BRD4_GRID_DIAGNOSIS_PLAN_20261025.md),
# frame set q3 = frames 124/249/374; our default values come from the committed online CSVs (--ours online).
#   MODE=amber NPROC=<n>            Amber MM/PBSA+GBSA, n replicas at a time, 1 thread each (CPU)
#                                   -> reference/amber_mmpbsa/q3/hou/..., log hou_ref_amber.log
#   MODE=grid W=<k> NW=<n> GPU=<id> grid sensitivity h0.4 / h0.35 / 3 rotations, every NW-th replica
#                                   -> reference/grid_sensitivity/q3/hou/..., log hou_ref_grid_w<k>.log
# A failing replica is logged as FAIL and the others continue.  Resumable: amber skips replicas
# with comparison.csv; grid skips configs already in sensitivity.csv.
set -euo pipefail
: "${PY:?}" "${MODE:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
FRAMES=124,249,374
cd "$J"
RIDS=($(awk -F, 'NR > 1 && $2 == "hou" {print $1}' "$P/manifests/runs.csv" | tr -d '\r'))
export PY P FRAMES
if [[ "$MODE" == amber ]]; then
  exec >> "$P/hou_ref_amber.log" 2>&1
  echo "== amber start host $(hostname) $(date -Is) NPROC=${NPROC:?} pid $$"
  one() {
    local rid=$1 s=${1%%_ph74_gaff2_rep*} r=${1##*_rep}
    if [[ -f "$P/reference/amber_mmpbsa/q3/hou/$s/ph74_gaff2/rep$r/comparison.csv" ]]; then
      echo "== [$rid] skip (done)"; return 0; fi
    if OMP_NUM_THREADS=1 "$PY" scripts/amber_reference_smallmol.py --inputs "$P/manifests/hou_inputs.csv" \
         --run-id "$rid" --frames "$FRAMES" --frame-set q3 --ours online > "$P/reference/.amber_$rid.log" 2>&1; then
      echo "== [$rid] done $(date -Is) $(tail -1 "$P/reference/.amber_$rid.log")"
    else
      echo "== [$rid] FAIL rc=$? $(date -Is) $(tail -2 "$P/reference/.amber_$rid.log" | tr '\n' ' ')"
    fi
  }
  export -f one
  mkdir -p "$P/reference"
  printf '%s\n' "${RIDS[@]}" | xargs -P "$NPROC" -I{} bash -c 'one "$@"' _ {}
  echo "== amber all done $(date -Is)"
else
  : "${W:?}" "${NW:?}"
  if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
  exec >> "$P/hou_ref_grid_w$W.log" 2>&1
  echo "== grid worker $W/$NW start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
  for (( i = W; i < ${#RIDS[@]}; i += NW )); do
    rid=${RIDS[$i]}
    echo "== [$rid] grid $(date -Is)"
    "$PY" scripts/grid_sensitivity_smallmol.py --inputs "$P/manifests/hou_inputs.csv" --run-id "$rid" \
      --frames "$FRAMES" --frame-set q3 --ours online \
      || { rc=$?; echo "== [$rid] FAIL rc=$rc $(date -Is)"; }
  done
  echo "== grid worker $W all done $(date -Is)"
fi
