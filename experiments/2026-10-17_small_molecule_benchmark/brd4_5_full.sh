#!/usr/bin/env bash
# BRD4-5 with alprazolam (docs/BRD4_5_REPREP_PLAN_20261101.md §5), independent package
# brd4_5_alprazolam_20261101/.  Two modes:
#   MODE=setup                        prep (prepare_brd4_5_alprazolam.py) + setup + B1 start score (default grid)
#   MODE=worker W=<k> NW=<n> GPU=<id> replicas with row index % NW == W: run 5 ns -> strip -> score default and h0.4
# Protocol, window and blocks as brd4_full.sh / brd4_h04.sh.  A failing replica is logged as FAIL and the
# worker moves on.  Resumable: re-run; finished prep/setup/chunks/frames are skipped.
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
: "${MODE:?MODE=setup or MODE=worker}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
K=$P/brd4_5_alprazolam_20261101
M=alprazolam_gaff_fix
cd "$J"
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
DEV=0
mkdir -p "$K"

if [[ "$MODE" == setup ]]; then
  exec >> "$K/setup.log" 2>&1
  echo "== setup start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
  "$PY" "$P/scripts/prepare_brd4_5_alprazolam.py" --out "$K"
  "$PY" scripts/run_smallmol_md.py setup --package "$K" --inputs "$K/manifests/brd4_inputs.csv" \
    --sample BRD4-5 --target brd4 --model-id "$M" --platform CUDA --device-index "$DEV"
  "$PY" scripts/score_smallmol_start.py --package "$K" --inputs "$K/manifests/brd4_inputs.csv" \
    --target brd4 --model-id "$M" --samples BRD4-5 || echo "== FAIL start score rc=$?"
  echo "== setup done $(date -Is)"
  exit 0
fi

: "${W:?}" "${NW:?}"
exec >> "$K/w${W}.log" 2>&1
echo "== worker $W/$NW start host $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pid $$"
mapfile -t RIDS < <(awk -F, -v w="$W" -v nw="$NW" 'NR > 1 && (NR - 2) % nw == w {print $1}' "$K/manifests/runs.csv")
echo "== worker $W replicas: ${RIDS[*]}"
A=(--package "$K" --inputs "$K/manifests/brd4_inputs.csv")
for RID in "${RIDS[@]}"; do
  echo "== [$RID] run $(date -Is)"
  "$PY" scripts/run_smallmol_md.py run "${A[@]}" --run-id "$RID" --platform CUDA --device-index "$DEV" --prod-ns 5 \
    || { rc=$?; echo "== [$RID] FAIL run rc=$rc $(date -Is)"; continue; }
  echo "== [$RID] strip $(date -Is)"
  "$PY" scripts/run_smallmol_md.py strip "${A[@]}" --run-id "$RID" \
    || { rc=$?; echo "== [$RID] FAIL strip rc=$rc $(date -Is)"; continue; }
  for G in default h0.4; do
    echo "== [$RID] score $G $(date -Is)"
    "$PY" scripts/score_smallmol_replica.py "${A[@]}" --run-id "$RID" --grid "$G" --start-ns 0 --end-ns 5 --block-ns 1 \
      || { rc=$?; echo "== [$RID] FAIL score $G rc=$rc $(date -Is)"; }
  done
done
echo "== worker $W done $(date -Is)"
