set -x
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 1. 标准基准，对齐 2080ti_v2.csv / 5080_v2.csv
python scripts/benchmark.py --csv 4090_v2.csv 2>&1 | tee bench_4090.log
# 2. 开 MPS，全部在线实验都在同一个 daemon 下跑（与 §18.22 口径一致）
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
python scripts/online_overhead.py --mode both --steps 200000 2>&1 | tee ovh_4090_both.log
for p in 20 30 40 50; do
  python scripts/online_overhead.py --mode async --steps 200000 --mps-percentage $p --mps-priority 1 2>&1 | tee ovh_4090_mps$p.log
done
echo quit | nvidia-cuda-mps-control
