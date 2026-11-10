#!/usr/bin/env bash
# Rescore all 30 BRD4 replicas with --grid h0.4 (C/R 0.4 / ligand box 0.2), same window/blocks
# as round 1: (0, 5] ns, 1-ns blocks; output scores/h0.4/brd4/... (docs/BRD4_H04_RESCORE_PLAN_20261025.md).
#   PY=... W=<k> NW=<n> GPU=<physical id> bash brd4_h04.sh       # log brd4_h04_w<k>.log
# A failing replica is logged as FAIL and the worker moves on.  Resumable: frames already in
# frames.csv are skipped.
set -euo pipefail
: "${PY:?}" "${W:?}" "${NW:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
cd "$J"
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
exec >> "$P/brd4_h04_w${W}.log" 2>&1
echo "== h0.4 worker $W/$NW host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
RIDS=()
for i in 1 2 3 4 5 6 7 8 9 10; do for R in 01 02 03; do RIDS+=("BRD4-${i}_mobley_amber_rep${R}"); done; done
for (( i = W; i < ${#RIDS[@]}; i += NW )); do
  RID=${RIDS[$i]}
  echo "== [$RID] score h0.4 $(date -Is)"
  "$PY" scripts/score_smallmol_replica.py --run-id "$RID" --grid h0.4 --start-ns 0 --end-ns 5 --block-ns 1 \
    || { rc=$?; echo "== [$RID] FAIL score rc=$rc $(date -Is)"; }
done
echo "== worker $W done $(date -Is)"
