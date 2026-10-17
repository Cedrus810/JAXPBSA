#!/usr/bin/env bash
# Full P4 scoring (design §10, 2026-10-15; protocol §2.1): every v2.1 replica, every saved
# frame (10 ps), frozen OnlineMMPBSA defaults.  Resumable: finished frames are skipped, so a
# re-run continues where it stopped.
#   PY="$(command -v python)" bash score_all.sh                                   # 1-10 ns -> scores/, score_all.log
#   PY=... START_NS=10 END_NS=20 OUT_ROOT=scores_10_20 LOG=score_ext20.log bash score_all.sh
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
START_NS="${START_NS:-1}"
END_NS="${END_NS:-10}"
OUT_ROOT="${OUT_ROOT:-scores}"
LOG="${LOG:-score_all.log}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-11_public_data_execution_packet
exec >> "$P/$LOG" 2>&1
cd "$J"
echo "== start host $(hostname) $(date -Is) pid $$ window ${START_NS}-${END_NS} ns out $OUT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
for S in PMI PMI-L10A T1L_L10A A4L_L10A N8A_L10A S11M_L10A P12Q_L10A PMI-M4; do
  for M in v21m1 v21m2; do
    for R in 01 02 03; do
      RID="${S}_${M}_rep${R}"
      echo "== [$RID] score $(date -Is)"
      "$PY" scripts/score_peptide_replica.py --run-id "$RID" --start-ns "$START_NS" --end-ns "$END_NS" \
          --stride 1 --out "$P/$OUT_ROOT/$S/$M/rep$R"
    done
  done
done
echo "== all done $(date -Is)"
