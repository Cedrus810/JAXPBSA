#!/usr/bin/env bash
# Prep protocol v2.1 validation (docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §6, §11).
#   V2: prep (--protocol v2.1, check C2; skipped when already prepared) + setup (C3)
#       for SAMPLES x MODELS (default: all 8 samples x v21m1, v21m2)
#   V3: REPS replicas of PROD_NS ns per model (default rep01, 1 ns) + strip (C4 per-frame record)
# Inputs: inputs/modeller_v21/<sample>/start_v21m{1,2}.pdb (MODELLER heavy atoms on 181 +
# reduce H on 183 + finalize_allh_models.py).  V3 starts only after every V2 model passed
# (set -e).  Resumable: re-run the same command; finished prep/setup/stages/chunks are skipped.
#
#   PY="$(command -v python)" DEV=0 bash p2_v21_validation.sh        # log: p2_v21_validation.log
# V4 (after V3 review): PROD_NS=10 REPS="01 02 03" with the samples chosen then.
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
DEV="${DEV:-0}"
PROD_NS="${PROD_NS:-1}"
REPS="${REPS:-01}"
SAMPLES="${SAMPLES:-PMI PMI-L10A T1L_L10A A4L_L10A N8A_L10A S11M_L10A P12Q_L10A PMI-M4}"
MODELS="${MODELS:-v21m1 v21m2}"
DCD_PS=10
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-11_public_data_execution_packet
exec >> "$P/p2_v21_validation.log" 2>&1
cd "$J"

echo "== start host $(hostname) $(date -Is) PY=$PY DEV=$DEV PROD_NS=$PROD_NS REPS=$REPS DCD_PS=$DCD_PS pid $$"
echo "== samples: $SAMPLES / models: $MODELS"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true

for S in $SAMPLES; do
  for M in $MODELS; do
    echo "== [$S/$M] prep (v2.1, C2) $(date -Is)"
    "$PY" scripts/prep_peptide_sample.py --sample "$S" --model-id "$M" --protocol v2.1
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
