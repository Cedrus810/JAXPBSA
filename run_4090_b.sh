set -x
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 1. 分阶段耗时：看 4090 上 solver / surface / DST 各占多少，对照 5090 的 82%/8%/5%
python scripts/profile_stages.py 0.5 32 2>&1 | tee prof_4090_h05_fp32.log
# 2. PERFORMANCE_REVIEW P3：V-cycle 平滑次数按 ms/帧 重扫（默认 nu=2）
for nu in 1 2 3; do
  python scripts/benchmark.py --quick --mg-nu $nu --csv 4090_mgnu$nu.csv 2>&1 | tee bench_4090_mgnu$nu.log
done
# 3. 弄清 --order jax 为什么系统性慢 0.8%：只跑 5000 间隔，不开 MPS，两个 order 各重复 3 次
for i in 1 2 3; do
  python scripts/online_overhead.py --config 5000 --mode sync --steps 200000 2>&1 | tee ovh_4090_order_rep$i.log
done
