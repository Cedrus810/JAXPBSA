#!/usr/bin/env bash
# Rescore all 21 Hou replicas with --grid h0.4 (C/R 0.4 / ligand box 0.2), window (0, 5] ns, 1-ns blocks
# (same as the online round); output scores/h0.4/hou/... (RESULTS §20.4; BRD4 counterpart brd4_h04.sh).
#   PY=... W=<k> NW=<n> GPU=<physical id> bash hou_h04.sh       # log hou_h04_w<k>.log
# A failing replica is logged as FAIL and the worker moves on.  Resumable: frames already in
# frames.csv are skipped.
set -euo pipefail
: "${PY:?}" "${W:?}" "${NW:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
cd "$J"
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
exec >> "$P/hou_h04_w${W}.log" 2>&1
echo "== h0.4 worker $W/$NW host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
RIDS=()
for i in 1 2 3 4 5 6 7; do for R in 01 02 03; do RIDS+=("a${i}_ph74_gaff2_rep${R}"); done; done
for (( i = W; i < ${#RIDS[@]}; i += NW )); do
  RID=${RIDS[$i]}
  echo "== [$RID] score h0.4 $(date -Is)"
  "$PY" scripts/score_smallmol_replica.py --inputs "$P/manifests/hou_inputs.csv" --run-id "$RID" --grid h0.4 --start-ns 0 --end-ns 5 --block-ns 1 \
    || { rc=$?; echo "== [$RID] FAIL score rc=$rc $(date -Is)"; }
done
echo "== worker $W done $(date -Is)"
