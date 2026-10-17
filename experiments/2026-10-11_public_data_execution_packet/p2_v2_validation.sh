#!/usr/bin/env bash
# Prep protocol v2 validation (docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §6), run on fulda.
#   V2: prep (--protocol v2, check C2) + setup (check C3) for PMI, PMI-L10A, P12Q_L10A x v2m1, v2m2
#   V3: one short replica per model (rep01, PROD_NS ns, default 1) + strip (C4 per-frame record)
# The MODELLER starts (inputs/modeller_v2/{PRO,GLN}/start_v2m*.pdb) were built on 181.
# V3 starts only after all six V2 models passed (set -e).  Resumable: re-run the same
# command; finished prep/setup/stages/chunks are skipped.
#
#   cd /home/ruigengji/JAXPBSA/experiments/2026-10-11_public_data_execution_packet
#   PY="$(command -v python)" DEV=0 setsid nohup bash p2_v2_validation.sh > /dev/null 2>&1 < /dev/null &
#   tail -f p2_v2_validation.log
# Later, V4 (after approval) extends the same replicas:
#   PY=... DEV=0 PROD_NS=10 REPS="01 02 03" setsid nohup bash p2_v2_validation.sh ...
#
# The shell prints "Done" at once and $! is only setsid's parent; the real PID is on
# the "== start ... pid N" line of the log.
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
DEV="${DEV:-0}"
PROD_NS="${PROD_NS:-1}"
REPS="${REPS:-01}"
DCD_PS=10
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-11_public_data_execution_packet
exec >> "$P/p2_v2_validation.log" 2>&1
cd "$J"

SAMPLES="PMI PMI-L10A P12Q_L10A"
MODELS="v2m1 v2m2"
echo "== start host $(hostname) $(date -Is) PY=$PY DEV=$DEV PROD_NS=$PROD_NS REPS=$REPS DCD_PS=$DCD_PS pid $$"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true

for S in $SAMPLES; do
  for M in $MODELS; do
    echo "== [$S/$M] prep (v2, C2) $(date -Is)"
    "$PY" scripts/prep_peptide_sample.py --sample "$S" --model-id "$M" --protocol v2
    echo "== [$S/$M] setup (C3) $(date -Is)"
    "$PY" scripts/run_peptide_md.py setup --sample "$S" --model-id "$M" --platform CUDA --device-index "$DEV"
  done
done
echo "== V2 passed for all models $(date -Is)"

for S in $SAMPLES; do
  for M in $MODELS; do
    for R in $REPS; do
      RID="${S}_${M}_rep${R}"
      echo "== [$RID] run $(date -Is)"
      "$PY" scripts/run_peptide_md.py run --run-id "$RID" --platform CUDA --device-index "$DEV" \
          --prod-ns "$PROD_NS" --dcd-ps "$DCD_PS"
      echo "== [$RID] strip $(date -Is)"
      "$PY" scripts/strip_peptide_replica.py --run-id "$RID" --diag-stride 1 --dry-stride 1
    done
  done
done
echo "== all done $(date -Is)"
