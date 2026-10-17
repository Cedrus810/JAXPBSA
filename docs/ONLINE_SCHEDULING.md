# 单 GPU 的 OpenMM / PBSA 在线调度

`PBSAReporter` 是同步入口：MD 回调完成整帧分析后才返回。
`online_async.AsyncPBSAReporter` 把分析移到同卡常驻进程，回调只复制坐标、步数和
模拟时间并入队。结果允许稍晚到达，以 `step` / `time_ps` 对齐；它适用于在线监测，
不适用于必须等当前 PBSA 结果才能决定下一步的积分器。

## 启动

在 Linux 的 NVIDIA GPU 环境中，用同一个用户启动 MPS，随后运行 MD：

```bash
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d
python your_md_script.py
```

MPS 要在 OpenMM CUDA Context 创建前启用。已有 MPS 服务时使用它，不要重复启动。
如果使用自定义 `CUDA_MPS_PIPE_DIRECTORY`，daemon、MD 和 worker 必须继承同一路径。
库不自动启动/关闭 MPS 服务，避免影响机器上其他运行中的任务。

```python
from jaxpbsa.online_async import AsyncPBSAReporter

# system/topology 来自溶剂化体系；ref/pilot 已切为溶质，单位 Å。
# sim 的 CUDA Context 使用当前可见的同一张 GPU。
with AsyncPBSAReporter.from_system(
    system, topology, solute_idx, ligand_local_idx, ref_coords_A,
    interval_steps=25000,
    analyzer_kwargs={"pilot_coords_A": pilot_coords_A},
    out_csv="pbsa.csv",
    max_pending=2,
    mps_thread_percentage=30,
    mps_priority=1,
    cpu_threads=1,
) as reporter:
    sim.reporters.append(reporter)
    sim.step(production_steps)
```

不要先在 MD 进程创建 `OnlineMMPBSA` 再传给 worker。`from_system` 只序列化主机参数；
JAX 网格、编译和预热全部留在 worker。父进程选择的 fp32/fp64 数组精度会显式传递。
`analyzer_kwargs` 接受现有 `OnlineMMPBSA` 参数，网格、求解精度和科学计算公式不变。

子进程环境在 Python 启动前设置，不 fork 已有 CUDA Context。它关闭 JAX 预分配，并
将 OMP/OpenBLAS/MKL 的线程数限制为 `cpu_threads`。CPU 线程设置不等于 CPU 核绑定。
MPS 线程限额和优先级仅写入 worker 环境，父进程的设置不会被修改。

`mps_thread_percentage` / `mps_priority` 默认不设置，可运行普通同卡异步对照。
显式请求 MPS 时，库检查 control 程序和 daemon 是否可联系，失败就在预热前报错，
不会把不存在的 MPS 限额当成已经生效。`reporter.worker_config` 留存 worker 接收的
环境设置；daemon 全局限额还可能进一步限制它。
daemon 可联系不证明 OpenMM Context 已接入 MPS。带 MPS 参数的基准会在计时前
查询 `get_server_list` / `get_client_list <server PID>`，核对 MD 进程和异步 worker
的 PID，输出 `MPS_VERIFIED`；任一 PID 缺失则失败。该检查要求与 daemon 使用相同
PID namespace，确认的是进程 client 身份，不能量化每个 Context 的实际并发或资源占用。
命令接口见 [NVIDIA MPS 工具参考](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)。

worker 的 `PYTHONPATH` 保留父进程的 `sys.path`，包括可能指向 `scripts/` 的首项。
新增脚本请避免与库或标准库模块重名，否则可能遮蔽 worker 所需的导入。

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` 限制可用的执行线程规模，不是 GPU 时间或
显存带宽的 20% 配额，也不预留独占 SM。低优先级是调度提示，不抢占已经运行的
kernel。这些设置必须在实际 GPU 上扫描验证，不能据此承诺 MD 不减速。
参考 [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)。

## 积压、结果和退出

- `max_pending` 包括正在算的帧；默认最多 2 个未收集结果的帧。队列满时立即报错，
  不等待 GPU、不无声丢帧。说明分析速度跟不上采样，需要增加间隔或 worker 资源。
- 纯监测可以显式选择 `on_backlog="skip"`。首次跳帧会警告，累计计数在 `n_skipped`，
  退出时打印。跳过的样本不适合直接混入固定间隔的系综均值。
- worker 复用同步入口的 margin / convergence / SA 检查。`on_violation="flag"`
  写标记并继续；`"raise"` 和计算错误在下次 `poll()`、reporter 调用或 `close()` 抛出。
  异步错误不能在帧提交时立即知道。
- CSV 每个完成结果及时 flush，写盘不在 MD 回调中。额外字段 `analysis_s` 是分析器
  耗时，`queue_wait_s` 是从快照提交到分析开始的等待，`latency_s` 是提交到分析完成的
  时间（不包含其后 CSV flush 和结果回传）。
  父子进程统一用 `time.clock_gettime(time.CLOCK_MONOTONIC)` 计这些时间，依赖 POSIX
  同一主机共享的单调时钟；不依赖 `perf_counter()` 的跨进程实现。
- `poll()` 非阻塞地收集新结果，返回列表并更新 `latest_result`、`n_reports`、
  `total_pbsa_s`、`max_latency_s`。`pending` 是提交数减去已收集结果数；查询前可 poll。
  不 poll 时 CSV 仍会及时写，OpenMM 每次询问下一次报告时也会自动 poll。
- 用 context manager 或显式 `close()` 排空所有已提交帧，回收进程。一次关闭之后，
  若继续用同一个 Simulation，请先从 `sim.reporters` 移除该 reporter。
  `close(timeout=...)` 超时后可重试，不强杀仍在运行的 CUDA/MPS client。
  context manager 退出默认最多等 60 秒，可通过 `close_timeout=...` 修改，`None` 表示
  无限等待；显式 `close()` 不设 timeout 时仍等待排空。超时会抛出 `TimeoutError`，
  保留 reporter 后可再次调用 `close(timeout=...)`。基准也使用这一关闭超时。
  忘记关闭且 `pending > 0` 时，`weakref.finalize` 发出 `ResourceWarning` 并尝试请求
  正常排空，不阻塞垃圾回收；解释器退出时不能保证排空。Python 默认隐藏
  `ResourceWarning`，可用 `-W default::ResourceWarning` 查看。

## 验证同卡调度

保持体系、采样间隔、GPU 和 MPS daemon 相同，对照独占 MD、同步 reporter 和异步 worker：

```bash
# 普通同卡异步，与同步入口对照。
python scripts/online_overhead.py --mode both --steps 200000

# 已启动 MPS；限额只给后台 PBSA，baseline/sync 在同一个 daemon 下测。
python scripts/online_overhead.py --mode both --steps 200000 \
    --mps-percentage 20 --mps-priority 1
```

可分别试 10/20/30 的线程限额。每个配置使用独立进程，并且支持先建 OpenMM 或先预热
JAX 两种初始化顺序。`RESULT` 行的 `ns_day` 是 MD 循环吞吐，`drain_s` 是 MD 结束后
排空最后帧的时间，`total_ns_day` 把排空计入端到端耗时。`max_latency_s` 记录最大分析
延迟，`submit_s` 记录回调中的提交耗时（不包含 OpenMM 获取 State 的同步成本）。
所有已提交帧必须完成；任一配置失败时基准以非零状态退出。

先看 MD 吞吐是否改善，再看延迟是否小于采样帧的墙钟间隔、有无积压；PBSA 单帧变慢
本身不代表方案失败。profiling 要在真实 MD + PBSA 重叠期间采集，CUDA Graph 的 trace
使用 `--cuda-graph-trace=node`。历史 RESULTS §16/§18.15 是同步分时实测，不能用它
排除 MPS 并发的收益。

当前测试覆盖真实进程通信、OpenMM 与更短间隔 reporter 共存、MD 在 worker 忙时继续
步进、坐标快照、结果顺序、及时落盘、失败传播和正常排空。真实分析器与同步入口的
数值对照按运行时 JAX 后端执行。

2080 Ti 实测见 RESULTS §18.22（5 万步单次）：不限额异步只比同步快 ~1.5%；MPS 限额后
5000 步间隔的 MD overhead 由 15% 降到 ~10%，30% 与 20% 吞吐相当、延迟低一半。
4090 复测见 RESULTS §18.23（20 万步，MPS 20–50% 四档）：5000 步间隔 11.8% → 3.9%（30%），
20/30/40% 在噪声内持平，50% 变差，20% 只增加延迟。推荐 `mps_thread_percentage=30,
mps_priority=1`。每配置各跑一次；干扰来源未 nsys profile。
