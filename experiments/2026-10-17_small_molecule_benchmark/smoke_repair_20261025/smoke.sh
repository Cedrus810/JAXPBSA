#!/usr/bin/env bash
# One smoke of the repaired paths on 183 (2080 Ti), Hou a1, own package (not the real replicas):
#   A  online MD (AsyncPBSAReporter + MPS 30%) killed during chunk 2, resumed with the same command,
#      then the same command again (must skip as complete).
#   B  strip -> offline score killed after >=5 frames -> strip again (new DCD header time)
#      -> score resume with the same --out (rejected before the 2026-10-25 fix) -> score again (0 new).
# Log smoke.log.
set -euo pipefail
J=/home/ruigengji/JAXPBSA
P=$J/experiments/2026-10-17_small_molecule_benchmark
S=$P/smoke_repair_20261025
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
RID=a1_ph74_gaff2_smoke2
REP=$S/md/hou/a1/ph74_gaff2/smoke2
OUT=$S/scores/a1_smoke2
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_kasuga_repair_smoke CUDA_MPS_LOG_DIRECTORY=/tmp/mps_kasuga_repair_smoke_log
exec >> "$S/smoke.log" 2>&1
echo "== start host $(hostname) $(date -Is) pid $$"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control || true' EXIT
cd "$J"
C=(--package "$S" --inputs "$P/manifests/hou_inputs.csv")
R=(run "${C[@]}" --run-id "$RID" --platform CUDA --device-index 0 --nvt-ps 10 --npt-ps 100 --prod-ns 0.3
   --online --mps-percentage 30)
SC=(scripts/score_smallmol_replica.py "${C[@]}" --run-id "$RID" --start-ns 0 --end-ns 0.3 --block-ns 0.1 --out "$OUT")
chunks() { "$PY" -c "import json; print(json.load(open('$REP/status.json')).get('chunks_done', 0))" 2>/dev/null || echo 0; }

echo "== [A1] setup $(date -Is)"
"$PY" scripts/run_smallmol_md.py setup "${C[@]}" --sample a1 --target hou --model-id ph74_gaff2 \
    --platform CUDA --device-index 0

echo "== [A2] run 0.3 ns, to be killed during chunk 2 $(date -Is)"
"$PY" scripts/run_smallmol_md.py "${R[@]}" &
MD=$!
echo "md pid $MD"
until [[ -f "$REP/status.json" ]] && (( $(chunks) >= 1 )); do
  kill -0 "$MD" 2>/dev/null || { echo "== MD exited before chunk 1"; wait "$MD" || true; exit 1; }
  sleep 5
done
sleep 20
echo "== [A2] kill -TERM $MD at chunks_done=$(chunks) $(date -Is)"
ps -o pid,etime,cmd -p "$MD" || true
kill -TERM "$MD"
wait "$MD" || echo "md rc=$?"
sleep 15   # the PBSA worker exits on EOF from the closed socket
ls -la "$REP/online"

echo "== [A3] resume with the same command $(date -Is)"
"$PY" scripts/run_smallmol_md.py "${R[@]}"
echo "== [A4] same command again (expect skip, complete) $(date -Is)"
"$PY" scripts/run_smallmol_md.py "${R[@]}"

echo "== [B1] strip $(date -Is)"
"$PY" scripts/run_smallmol_md.py strip "${C[@]}" --run-id "$RID"
head -c 300 "$REP/analysis/solute_dry.dcd" | strings | grep Created
echo "== [B2] score, to be killed after >=5 frames $(date -Is)"
"$PY" "${SC[@]}" &
SP=$!
echo "score pid $SP"
until [[ -f "$OUT/frames.csv" ]] && (( $(wc -l < "$OUT/frames.csv") >= 6 )); do
  kill -0 "$SP" 2>/dev/null || { echo "== score exited before 5 frames"; wait "$SP" || true; exit 1; }
  sleep 1
done
ps -o pid,etime,cmd -p "$SP" || true
kill -TERM "$SP"
wait "$SP" || echo "score rc=$?"
echo "frames.csv lines after kill: $(wc -l < "$OUT/frames.csv")"
sleep 61   # next strip must land in a different minute so the DCD header changes

echo "== [B3] strip again $(date -Is)"
"$PY" scripts/run_smallmol_md.py strip "${C[@]}" --run-id "$RID"
head -c 300 "$REP/analysis/solute_dry.dcd" | strings | grep Created
echo "== [B4] score resume, same --out $(date -Is)"
"$PY" "${SC[@]}"
echo "frames.csv lines: $(wc -l < "$OUT/frames.csv")"
echo "== [B5] score again (expect 0 new) $(date -Is)"
"$PY" "${SC[@]}"
echo "== all done $(date -Is)"
