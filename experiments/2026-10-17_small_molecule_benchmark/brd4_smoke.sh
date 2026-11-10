#!/usr/bin/env bash
# BRD4 end-to-end smoke (docs/BRD4_HOU_EXECUTION_HANDOFF.md §4.1, §7.5): one replica of SAMPLE,
# short production.  Checks parameters/indices (setup), MD + checkpoints, resume/extension,
# de-PBC dry trajectory, exact time axis, per-frame scoring with frozen defaults, scoring
# resume; and measures setup s, s/ns and ms/frame on this node for the full-run plan.
#   1 setup (manifest<->prmtop checks, force classes, charges, minimisation)
#   2 run rep01 to 0.2 ns, 3 extend to 0.4 ns (new context, resumes from chunk state),
#   4 same command again (must skip), 5 strip, 6 score (0, 0.4] ns, 7 score again (must skip)
# rep01 is the real rep01 of SAMPLE (same protocol, 100-ps chunks): if the smoke is accepted it
# is extended to 5 ns rather than redone.  Resumable: re-run the same command.
#   PY="$(command -v python)" SAMPLE=BRD4-3 GPU_SLOT=0 bash brd4_smoke.sh   # log: brd4_smoke_BRD4-3.log
# GPU_SLOT picks one entry of the job's CUDA_VISIBLE_DEVICES and hides the others from this
# process (OpenMM and JAX then both see exactly one GPU, index 0), so two smokes can share a
# 2-GPU allocation without JAX preallocating on the other GPU.
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
SAMPLE="${SAMPLE:-BRD4-3}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
RID="${SAMPLE}_mobley_amber_rep01"
exec >> "$P/brd4_smoke_${SAMPLE}.log" 2>&1
VIS_IN="${CUDA_VISIBLE_DEVICES:-unset}"
if [[ -n "${GPU_SLOT:-}" ]]; then
  IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export CUDA_VISIBLE_DEVICES="${G[$GPU_SLOT]}"
fi
DEV=0
cd "$J"
echo "== start $SAMPLE host $(hostname) $(date -Is) PY=$PY pid $$ CUDA_VISIBLE_DEVICES in=$VIS_IN now=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv || true
echo "== [1] setup $(date -Is)"
"$PY" scripts/run_smallmol_md.py setup --sample "$SAMPLE" --platform CUDA --device-index "$DEV"
echo "== [2] run 0.2 ns $(date -Is)"
"$PY" scripts/run_smallmol_md.py run --run-id "$RID" --platform CUDA --device-index "$DEV" --prod-ns 0.2
echo "== [3] extend to 0.4 ns $(date -Is)"
"$PY" scripts/run_smallmol_md.py run --run-id "$RID" --platform CUDA --device-index "$DEV" --prod-ns 0.4
echo "== [4] same command again, expect no new chunk $(date -Is)"
"$PY" scripts/run_smallmol_md.py run --run-id "$RID" --platform CUDA --device-index "$DEV" --prod-ns 0.4
echo "== [5] strip $(date -Is)"
"$PY" scripts/run_smallmol_md.py strip --run-id "$RID"
echo "== [6] score (0, 0.4] ns $(date -Is)"
"$PY" scripts/score_smallmol_replica.py --run-id "$RID" --start-ns 0 --end-ns 0.4 --block-ns 0.1 \
    --out "$P/scores/smoke/brd4/$SAMPLE/mobley_amber/rep01"
echo "== [7] score again, expect 0 new frames $(date -Is)"
"$PY" scripts/score_smallmol_replica.py --run-id "$RID" --start-ns 0 --end-ns 0.4 --block-ns 0.1 \
    --out "$P/scores/smoke/brd4/$SAMPLE/mobley_amber/rep01"
echo "== all done $(date -Is)"
