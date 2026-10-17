#!/usr/bin/env bash
# Molecular smoke for the packet entries: PMI prep -> setup -> 1 ns replica -> strip.
# Run on the user's own machine (it must see this NFS tree at /home/ruigengji/JAXPBSA,
# because proposed_runs.csv stores absolute output paths):
#
#   PY=/path/to/openmm_dev/bin/python DEV=0 setsid nohup bash smoke_pmi_rep01.sh \
#       > smoke_pmi_rep01.log 2>&1 < /dev/null & echo $!
#
# Afterwards extend the same replica to the P2 length with the identical command
# but --prod-ns 10 (keep --dcd-ps unchanged; it is fixed once the run starts).
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
DEV="${DEV:-0}"
DCD_PS="${DCD_PS:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-11_public_data_execution_packet
cd "$J"

echo "== host $(hostname)  $(date -Is)  PY=$PY  DEV=$DEV  DCD_PS=$DCD_PS"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true

echo "== [1/4] prep $(date -Is)"
"$PY" scripts/prep_peptide_sample.py --sample PMI --model-id model01 --seed 20261011

echo "== [2/4] setup $(date -Is)"
"$PY" scripts/run_peptide_md.py setup --sample PMI --model-id model01 --platform CUDA --device-index "$DEV"

echo "== [3/4] run 1 ns $(date -Is)"
"$PY" scripts/run_peptide_md.py run --run-id PMI_model01_rep01 --platform CUDA --device-index "$DEV" \
    --prod-ns 1 --dcd-ps "$DCD_PS"

echo "== [4/4] strip $(date -Is)"
"$PY" scripts/strip_peptide_replica.py --run-id PMI_model01_rep01 --diag-stride 1 --dry-stride 1

echo "== done $(date -Is)"
ls -la "$P/prepared/PMI/model01" "$P/md/PMI/model01/setup" "$P/md/PMI/model01/rep01" \
       "$P/md/PMI/model01/rep01/prod" "$P/md/PMI/model01/rep01/analysis"
