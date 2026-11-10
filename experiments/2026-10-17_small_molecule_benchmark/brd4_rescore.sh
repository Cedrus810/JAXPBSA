#!/usr/bin/env bash
# Score the 15 BRD4 replicas that OnlineMMPBSA's integer-charge check rejected in round 1
# (BRD4-1/5/6/7/10; docs/BRD4_HOU_EXECUTION_HANDOFF.md §7.4, §7.8).  The check is now 1e-2 e
# (jaxpbsa/online.py, 2026-10-18).  MD and strip are complete; only scoring runs here,
# same window/blocks/output as round 1: (0, 5] ns, 1-ns blocks, scores/default/brd4/...
#   PY=... W=<k> NW=<n> GPU=<physical id> bash brd4_rescore.sh       # log brd4_rescore_w<k>.log
# A failing replica is logged as FAIL and the worker moves on.  Resumable: frames already in
# frames.csv are skipped.
set -euo pipefail
: "${PY:?}" "${W:?}" "${NW:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
cd "$J"
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
exec >> "$P/brd4_rescore_w${W}.log" 2>&1
echo "== rescore worker $W/$NW host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
RIDS=()
for S in BRD4-1 BRD4-5 BRD4-6 BRD4-7 BRD4-10; do for R in 01 02 03; do RIDS+=("${S}_mobley_amber_rep${R}"); done; done
for (( i = W; i < ${#RIDS[@]}; i += NW )); do
  RID=${RIDS[$i]}
  echo "== [$RID] score $(date -Is)"
  "$PY" scripts/score_smallmol_replica.py --run-id "$RID" --start-ns 0 --end-ns 5 --block-ns 1 \
    || { rc=$?; echo "== [$RID] FAIL score rc=$rc $(date -Is)"; }
done
echo "== worker $W done $(date -Is)"
