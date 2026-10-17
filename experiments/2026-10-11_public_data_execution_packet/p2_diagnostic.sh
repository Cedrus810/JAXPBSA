#!/usr/bin/env bash
# P2 structure/equilibration diagnostic: PMI, PMI-L10A, P12Q_L10A x 3 replicas, 10 ns each.
# Sequential on one GPU.  Resumable: re-run the same command after any failure;
# finished prep/setup/stages/chunks are skipped.  The log is always written next to this script.
#
# model01 (prep seed 20261011; done 2026-10-12):
#   PY="$(command -v python)" DEV=0 setsid nohup bash p2_diagnostic.sh > /dev/null 2>&1 < /dev/null &
# model02 = start sensitivity: same protocol, only the PDBFixer prep seed differs (20261012);
# after each prep the rebuilt residues A:25 / B:12 are compared with model01 and the run
# stops if the start is identical (nothing to test):
#   PY="$(command -v python)" DEV=0 MODEL=model02 PREP_SEED=20261012 setsid nohup bash p2_diagnostic.sh > /dev/null 2>&1 < /dev/null &
#
# The shell prints "Done" at once and $! is only setsid's parent; the real PID is on
# the "== start ... pid N" line of the log.
set -euo pipefail
: "${PY:?set PY to the openmm_dev python on this machine}"
DEV="${DEV:-0}"
MODEL="${MODEL:-model01}"
PREP_SEED="${PREP_SEED:-20261011}"
PROD_NS=10
DCD_PS=10          # must match the smoke run of PMI_model01_rep01
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-11_public_data_execution_packet
if [ "$MODEL" = model01 ]; then LOG=$P/p2_diagnostic.log; else LOG=$P/p2_diagnostic_$MODEL.log; fi
exec >> "$LOG" 2>&1
cd "$J"

echo "== start host $(hostname) $(date -Is) PY=$PY DEV=$DEV MODEL=$MODEL PREP_SEED=$PREP_SEED PROD_NS=$PROD_NS DCD_PS=$DCD_PS pid $$"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true

for S in PMI PMI-L10A P12Q_L10A; do
  echo "== [$S/$MODEL] prep $(date -Is)"
  "$PY" scripts/prep_peptide_sample.py --sample "$S" --model-id "$MODEL" --seed "$PREP_SEED"
  if [ "$MODEL" != model01 ]; then
    echo "== [$S/$MODEL] compare rebuilt residues with model01 $(date -Is)"
    "$PY" - "$P/prepared/$S/model01/complex_dry.pdb" "$P/prepared/$S/$MODEL/complex_dry.pdb" <<'EOF'
import sys, json, math
def heavy(path):
    out = {}
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")) and ln[76:78].strip() != "H":
            key = (ln[21], int(ln[22:26]), ln[12:16].strip())
            if key[:2] in (("A", 25), ("B", 12)):
                out[key] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return out
a, b = heavy(sys.argv[1]), heavy(sys.argv[2])
if set(a) != set(b):
    raise SystemExit(f"rebuilt-residue atoms differ between models: {sorted(set(a) ^ set(b))}")
dev = {f"{k[0]}:{k[1]}.{k[2]}": round(math.dist(a[k], b[k]), 3) for k in a}
per_res = {r: max(v for k, v in dev.items() if k.startswith(r)) for r in ("A:25.", "B:12.")}
print(json.dumps({"max_heavy_atom_shift_vs_model01_A": per_res}))
# PDB coordinates carry 0.001 A; a smaller shift means the same start
if max(dev.values()) < 0.001:
    raise SystemExit("start identical to model01: prep seed does not change the rebuild; nothing to test")
EOF
  fi
  echo "== [$S/$MODEL] setup $(date -Is)"
  "$PY" scripts/run_peptide_md.py setup --sample "$S" --model-id "$MODEL" --platform CUDA --device-index "$DEV"
  for R in 01 02 03; do
    RID="${S}_${MODEL}_rep${R}"
    echo "== [$RID] run $(date -Is)"
    "$PY" scripts/run_peptide_md.py run --run-id "$RID" --platform CUDA --device-index "$DEV" \
        --prod-ns "$PROD_NS" --dcd-ps "$DCD_PS"
    echo "== [$RID] strip $(date -Is)"
    "$PY" scripts/strip_peptide_replica.py --run-id "$RID" --diag-stride 1 --dry-stride 1
  done
done
echo "== all done $(date -Is)"
