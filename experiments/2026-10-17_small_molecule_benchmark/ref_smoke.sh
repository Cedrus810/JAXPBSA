#!/usr/bin/env bash
# Same-frame reference smoke (handoff §4.6, §7.9): BRD4-3 rep01, frame 124 only.
#   A  Amber MM/PBSA + MM/GBSA (CPU, MMPBSA.py)        -> reference/amber_mmpbsa/smoke1/...
#   B  PB grid sensitivity h0.4 / h0.35 / rotations (GPU) -> reference/grid_sensitivity/smoke1/...
# A and B run concurrently.  Checks file handling and parsing; measures pbsa s/frame and the
# refined-grid warm-up and ms/frame, which price the full 90-frame plan.  Log ref_smoke.log.
set -euo pipefail
: "${PY:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
RID=BRD4-3_mobley_amber_rep01
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
exec >> "$P/ref_smoke.log" 2>&1
cd "$J"
echo "== start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
( echo "== [A] amber $(date -Is)"
  OMP_NUM_THREADS=1 "$PY" scripts/amber_reference_smallmol.py --run-id "$RID" --frames 124 --frame-set smoke1 \
    && echo "== [A] done $(date -Is)" || echo "== [A] FAIL rc=$? $(date -Is)" ) &
( echo "== [B] grid $(date -Is)"
  "$PY" scripts/grid_sensitivity_smallmol.py --run-id "$RID" --frames 124 --frame-set smoke1 \
    && echo "== [B] done $(date -Is)" || echo "== [B] FAIL rc=$? $(date -Is)" ) &
wait
echo "== all done $(date -Is)"
