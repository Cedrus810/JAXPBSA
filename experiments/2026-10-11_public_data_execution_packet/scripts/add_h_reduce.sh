#!/usr/bin/env bash
# Prep protocol v2.1, stage 3: hydrogens by reduce (MolProbity) on the MODELLER heavy-atom models.
# docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md §11.  Runs on 183 (openmm_dev's reduce; not on 181).
#   -NOFLIP     no Asn/Gln/His flips: heavy atoms stay exactly as MODELLER wrote them
#   -HIS        add ring NH to His; reduce picks HD1/HE2/both from its H-bond scoring
#   -NUClear    nuclear X-H distances (ff14SB C-H 1.09 A; the default electron-cloud ~0.97 A)
#   -Nterm25    residues numbered <= 25 may be N-terminal: receptor starts at A:Glu25, peptide at B:1
# Output: inputs/modeller_v21/<sample>/reduced_<v2mN>.pdb (+ .log).  Re-running overwrites.
#
#   bash experiments/2026-10-11_public_data_execution_packet/scripts/add_h_reduce.sh            # all samples, v2m1 v2m2
set -euo pipefail
REDUCE="${REDUCE:-/home/ruigengji/miniforge3/envs/openmm_dev/bin/reduce}"
P=/home/ruigengji/JAXPBSA/experiments/2026-10-11_public_data_execution_packet
for d in "$P"/inputs/modeller_v21/*/; do
  for m in ${STARTS:-v2m1 v2m2}; do
    h="$d/heavy_$m.pdb"
    [ -f "$h" ] || continue
    # reduce exits 1 when it changed the model (e.g. added H); only >1 is an error
    rc=0; "$REDUCE" -NOFLIP -HIS -NUClear -Nterm25 "$h" > "$d/reduced_$m.pdb" 2> "$d/reduced_$m.log" || rc=$?
    if [ "$rc" -gt 1 ]; then echo "reduce failed ($rc) on $h" >&2; exit 1; fi
    echo "$(basename "$d")/$m: $(grep -c '^ATOM' "$d/reduced_$m.pdb") atoms, $(grep -m1 'USER  MOD reduce' "$d/reduced_$m.pdb" | cut -c11-)"
  done
done
