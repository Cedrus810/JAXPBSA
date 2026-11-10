#!/usr/bin/env bash
# Hou h0.4 bounded numerical acceptance (docs/HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md §3, §5).
#   MODE=grid W=<k> NW=<n> GPU=<id>   batch A + C: our h0.35 on frame set s25 for all 21 replicas, plus h0.3 for
#                                     a1/a2/a6 in the same call (protocol.json binds the config list, so each
#                                     replica always runs with the same --configs); every NW-th replica
#                                     -> reference/grid_sensitivity/s25/hou/..., log hou_acc_grid_w<k>.log
#   MODE=amber NPROC=<n>              batch B: Amber pbsa h 0.35 and 0.25 on the q3 frames, n replicas at a time,
#                                     1 thread each (CPU); new --out per spacing, never the default q3 path
#                                     (amber_reference_smallmol.py rmtree's an existing --out)
#                                     -> reference/amber_mmpbsa/q3_pbsa<h>/hou/..., log hou_acc_amber.log
# A failing replica is logged as FAIL and the others continue.  Resumable: grid skips (config, frame) rows
# already in sensitivity.csv; amber skips (replica, spacing) with comparison.csv.
set -euo pipefail
: "${PY:?}" "${MODE:?}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
S25=$(awk -F, 'NR == 2 {print $0}' "$P/reference/frame_sets/s25.csv" | grep -o '9;29;[0-9;]*' | tr ';' ',')
Q3=124,249,374
cd "$J"
RIDS=($(awk -F, 'NR > 1 && $2 == "hou" {print $1}' "$P/manifests/runs.csv" | tr -d '\r'))
export PY P Q3
if [[ "$MODE" == amber ]]; then
  exec >> "$P/hou_acc_amber.log" 2>&1
  echo "== amber start host $(hostname) $(date -Is) NPROC=${NPROC:?} pid $$"
  one() {
    local rid=${1%:*} h=${1#*:} s r out
    s=${rid%%_ph74_gaff2_rep*}; r=${rid##*_rep}
    out=$P/reference/amber_mmpbsa/q3_pbsa$h/hou/$s/ph74_gaff2/rep$r
    if [[ -f "$out/comparison.csv" ]]; then echo "== [$rid pbsa $h] skip (done)"; return 0; fi
    if OMP_NUM_THREADS=1 "$PY" scripts/amber_reference_smallmol.py --inputs "$P/manifests/hou_inputs.csv" \
         --run-id "$rid" --frames "$Q3" --frame-set q3 --ours online --pbsa-h "$h" --out "$out" \
         > "$P/reference/.acc_amber_${rid}_$h.log" 2>&1; then
      echo "== [$rid pbsa $h] done $(date -Is)"
    else
      echo "== [$rid pbsa $h] FAIL rc=$? $(date -Is) $(tail -2 "$P/reference/.acc_amber_${rid}_$h.log" | tr '\n' ' ')"
    fi
  }
  export -f one
  { for h in 0.35 0.25; do for rid in "${RIDS[@]}"; do echo "$rid:$h"; done; done; } \
    | xargs -P "$NPROC" -I{} bash -c 'one "$@"' _ {}
  echo "== amber all done $(date -Is)"
else
  : "${W:?}" "${NW:?}"
  [[ "$S25" == 9,29,*,489 ]] || { echo "bad s25 frame list: $S25"; exit 1; }
  if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
  exec >> "$P/hou_acc_grid_w$W.log" 2>&1
  echo "== grid worker $W/$NW start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
  for (( i = W; i < ${#RIDS[@]}; i += NW )); do
    rid=${RIDS[$i]}
    case ${rid%%_ph74_gaff2_rep*} in a1|a2|a6) cfg=h0.35,h0.3 ;; *) cfg=h0.35 ;; esac
    echo "== [$rid] $cfg $(date -Is)"
    "$PY" scripts/grid_sensitivity_smallmol.py --inputs "$P/manifests/hou_inputs.csv" --run-id "$rid" \
      --frames "$S25" --frame-set s25 --configs "$cfg" --ours online \
      || { rc=$?; echo "== [$rid] FAIL rc=$rc $(date -Is)"; }
  done
  echo "== grid worker $W all done $(date -Is)"
fi
