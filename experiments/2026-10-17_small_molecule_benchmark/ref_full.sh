#!/usr/bin/env bash
# Same-frame reference for all BRD4 replicas (docs/BRD4_HOU_EXECUTION_HANDOFF.md §4.6, §7.9),
# frame set q3 = frames 124/249/374 (reference/frame_sets/q3.csv).  Two modes, run concurrently:
#   MODE=amber NPROC=<n>   Amber MM/PBSA+GBSA, n replicas at a time, 1 thread each (CPU)
#                          -> reference/amber_mmpbsa/q3/..., log ref_full_amber.log
#   MODE=grid GPU=<id>     PB grid sensitivity h0.4 / h0.35 / 3 rotations, replicas in turn
#                          -> reference/grid_sensitivity/q3/..., log ref_full_grid.log
# A failing replica is logged as FAIL and the others continue.  Resumable: amber skips replicas
# with comparison.csv; grid skips configs already in sensitivity.csv.
set -euo pipefail
: "${PY:?}" "${MODE:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
FRAMES=124,249,374
cd "$J"
RIDS=$(awk -F, 'NR > 1 && $2 == "brd4" {print $1}' "$P/manifests/runs.csv")
export PY P FRAMES
if [[ "$MODE" == amber ]]; then
  exec >> "$P/ref_full_amber.log" 2>&1
  echo "== amber start host $(hostname) $(date -Is) NPROC=${NPROC:?} pid $$"
  one() {
    local rid=$1 s=${1%%_mobley_amber_rep*} r=${1##*_rep}
    if [[ -f "$P/reference/amber_mmpbsa/q3/brd4/$s/mobley_amber/rep$r/comparison.csv" ]]; then
      echo "== [$rid] skip (done)"; return 0; fi
    if OMP_NUM_THREADS=1 "$PY" scripts/amber_reference_smallmol.py --run-id "$rid" --frames "$FRAMES" --frame-set q3 \
         > "$P/reference/.amber_$rid.log" 2>&1; then
      echo "== [$rid] done $(date -Is) $(tail -1 "$P/reference/.amber_$rid.log")"
    else
      echo "== [$rid] FAIL rc=$? $(date -Is) $(tail -2 "$P/reference/.amber_$rid.log" | tr '\n' ' ')"
    fi
  }
  export -f one
  mkdir -p "$P/reference"
  printf '%s\n' $RIDS | xargs -P "$NPROC" -I{} bash -c 'one "$@"' _ {}
  echo "== amber all done $(date -Is)"
else
  if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
  exec >> "$P/ref_full_grid.log" 2>&1
  echo "== grid start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
  for rid in $RIDS; do
    echo "== [$rid] grid $(date -Is)"
    "$PY" scripts/grid_sensitivity_smallmol.py --run-id "$rid" --frames "$FRAMES" --frame-set q3 \
      || { rc=$?; echo "== [$rid] FAIL rc=$rc $(date -Is)"; }
  done
  echo "== grid all done $(date -Is)"
fi
