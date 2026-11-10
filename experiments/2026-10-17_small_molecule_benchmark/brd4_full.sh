#!/usr/bin/env bash
# BRD4 first-round production (docs/BRD4_HOU_EXECUTION_HANDOFF.md §4.3, §7.6): 10 samples x 3
# replicas x 5 ns (manifests/runs.csv), each replica stripped and scored (0, 5] ns with five
# 1-ns blocks (§4.7) right after its MD.  Two modes:
#   MODE=setup                       setup of BRD4-1..10 in turn (finished ones are skipped)
#   MODE=worker W=<k> NW=<n> GPU=<id>  replicas with row index % NW == W, on physical GPU <id>
# A replica that fails (MD, strip or score) is logged as FAIL with its exit code and the worker
# moves on (handoff §1: one entry's problem does not stop the others).  Scoring of
# BRD4-1/5/6/7/10 is expected to FAIL at OnlineMMPBSA init (integer-charge check, §7.4);
# their MD and strip still run.  Resumable: re-run; finished stages/chunks/frames are skipped.
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
: "${MODE:?MODE=setup or MODE=worker}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
cd "$J"
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi   # this process sees one GPU
DEV=0

if [[ "$MODE" == setup ]]; then
  exec >> "$P/brd4_full_setup.log" 2>&1
  echo "== setup start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
  for S in BRD4-1 BRD4-2 BRD4-3 BRD4-4 BRD4-5 BRD4-6 BRD4-7 BRD4-8 BRD4-9 BRD4-10; do
    echo "== [$S] setup $(date -Is)"
    "$PY" scripts/run_smallmol_md.py setup --sample "$S" --platform CUDA --device-index "$DEV"
  done
  echo "== setup done $(date -Is)"
  exit 0
fi

: "${W:?}" "${NW:?}"
exec >> "$P/brd4_full_w${W}.log" 2>&1
echo "== worker $W/$NW start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv || true
mapfile -t RIDS < <(awk -F, -v w="$W" -v nw="$NW" 'NR > 1 && $2 == "brd4" && (NR - 2) % nw == w {print $1}' "$P/manifests/runs.csv")
echo "== worker $W replicas: ${RIDS[*]}"
for RID in "${RIDS[@]}"; do
  echo "== [$RID] run $(date -Is)"
  "$PY" scripts/run_smallmol_md.py run --run-id "$RID" --platform CUDA --device-index "$DEV" --prod-ns 5 \
    || { rc=$?; echo "== [$RID] FAIL run rc=$rc $(date -Is)"; continue; }
  echo "== [$RID] strip $(date -Is)"
  "$PY" scripts/run_smallmol_md.py strip --run-id "$RID" \
    || { rc=$?; echo "== [$RID] FAIL strip rc=$rc $(date -Is)"; continue; }
  echo "== [$RID] score $(date -Is)"
  "$PY" scripts/score_smallmol_replica.py --run-id "$RID" --start-ns 0 --end-ns 5 --block-ns 1 \
    || { rc=$?; echo "== [$RID] FAIL score rc=$rc $(date -Is)"; }
done
echo "== worker $W done $(date -Is)"
