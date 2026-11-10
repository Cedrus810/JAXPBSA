set -x
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for s in 1YCR S4; do
  python scripts/grid_error.py --name $s --frames 20 --stride 500 --phases 3 \
    --h-cr 0.5 0.4 0.35 0.3 --h-lig 0.25 0.2 0.15 --out grid_err_$s.npz 2>&1 | tee grid_err_$s.log
done
