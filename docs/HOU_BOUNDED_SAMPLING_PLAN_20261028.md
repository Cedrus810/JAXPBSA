# Hou 有界补采样：计划（2026-10-28，Claude 起草，未执行）

## 必读（开工前全部读完）

| 路径 | 为什么要读 |
|---|---|
| `/home/ruigengji/JAXPBSA/CLAUDE.md` | 冻结基线、生产环境、PBS 选卡写法、4090/5090 的使用限制。 |
| `/home/ruigengji/JAXPBSA/docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md` §7、§9、§10 | 采样诊断与"同起点不同速度不等于多个结构状态"；统计口径（副本先取时间均值，再等权平均）；"副本/模式差异主导次序：给出有界采样诊断预算或报告不可分辨"，以及不能无限延长。 |
| `/home/ruigengji/JAXPBSA/RESULTS.md` §20.2、§20.5–§20.8 | 现有 3 副本 × 5 ns 的分数；副本差异诊断（漂移、姿态、关键相互作用）；数值和在线成本的结论。 |
| `/home/ruigengji/JAXPBSA/scripts/run_smallmol_md.py` 文首说明 | 续跑与延长（调大 `--prod-ns` 就从已提交的 chunk 接着跑），在线分段 CSV 的规则。 |

## 0. 来由与用户决定

- 内容来自用户转贴的他人 comment（2026-10-28）第 4 步：根据前两步决定有限补采样；若主要是持续漂移，就延长相关副本；若是不同姿态长期不交换，就考虑增加独立起点；提前固定预算和停止条件；补完仍分不开的样本，就报告不可分辨，保留完整结果。
- 用户原话（2026-10-28）："试试"（回答"要我写第 4 步的计划吗？"）。
- 报告用默认网格还是 h0.4，仍由用户定。

## 1. 依据（已有结果）

- RESULTS §20.6：
  - a1 rep01、a3 rep01 在同一姿态里持续漂移，5 ns 结束时还没停。
  - a3 rep02、a4 rep02/rep03 跳到了新姿态，5 ns 内没有回来。
  - a6 rep01 在平衡阶段就丢掉了 Asp189 的相互作用。
  - 所以两种情况都有：有的副本是漂移，有的是不同状态不交换。
- RESULTS §20.7：网格修正带来的分差变化（≤1.1）比各对样本的合并副本标准误小，Hou 的次序主要受采样限制。
- RESULTS §20.8：生产入口在 3090 上加在线评分是 298 s/ns；NVT 100 ps 用 21 s，NPT 1 ns 用 261 s。

## 2. 方案（对 7 个样本用同一条规则，在任何新运行之前定死）

不按哪个样本和实验不符来挑样本补采样（设计稿 §9、§10），7 个样本一律补到 **6 个副本 × 10 ns**：

| 段 | 内容 | 回答什么 |
|---|---|---|
| S1 延长 | 现有 21 个副本原地从 5 ns 延长到 10 ns（`run --prod-ns 10 --online --mps-percentage 30`，从 chunk 50 接着跑） | 漂移的副本会不会走到平台；跳到新姿态的副本会不会回来 |
| S2 加副本 | 每个样本加 rep04–rep06（新种子写进 `manifests/runs.csv`），与原副本同一协议：同一晶体起点、新速度、NVT 100 ps + NPT 1 ns、生产 10 ns、在线评分 | 副本均值的标准误；每种姿态或状态在 6 个副本里出现几次 |

- **"独立起点"的取法**：Hou 每个配体只有一个晶体结构，所以 S2 仍从晶体起点出发、用新速度。不用 MD 里跑出来的姿态当新起点：设计稿 §7 写明，不能在没观测到转换的情况下，把几个起点等权混合当作热力学布居。各副本落在哪种状态，分开报告。
- **评分**：在线、默认网格（生产入口没有 `--grid`）。h0.4 不在本批；如果用户选 h0.4，再对新帧离线重评一次（按 §20.5 的速度，约 3–4 GPU-h）。
- **主分析窗口**：(0, 10] ns，不丢弃任何前段，3 + 3 个副本等权。另外单独报告 (0, 5] 和 (5, 10] 两半，以及原来 3 副本 × (0, 5] 的结果，便于对照。窗口现在就定死，看到结果后不改。
- **不改**：网格、MD 协议、评分间隔、MPS 30%、实验数据。

## 3. 预算与停止条件（预先写定）

- **预算**：S1 是 21 × 5 ns = 105 ns；S2 是 21 × (1.1 + 10) ns = 233 ns。按 §20.8 实测的 298 s/ns（3090，含在线评分），NVT/NPT 每个副本另加约 4.7 min，合计约 28 GPU-h。用一个 2 × 3090 节点约 14 h，用两个节点约 7 h。
- **停止条件**：
  - 跑完 6 × 10 ns 就停，不管结果如何都不再延长或加副本。
  - 补完仍分不开的样本对，报告"不可分辨"，并公开全部副本和姿态记录。
  - 某个副本失败时，原命令续跑一次；还失败就报告缺失，不换种子重来。
  - 不按实验相关性删副本、删帧或改窗口。
- **预期**（据 §20.7 的数字推算，只作说明，不作门）：如果副本间的离散不变，6 个副本的标准误约是 3 个时的 1/√2。a1/a3（分差 10.3）可能从合并标准误约 6.2 降到约 4.4；a2/a4 现在就能分开。a7/a1、a3/a2、a4/a5/a6 的分差都 <1，补完大概率仍分不开。

## 4. 冒烟（一次，不占 PBS，不碰生产副本）

在线延长这条路还没在真实运行里走过。用独立小包里已经跑完的 `acconline`（1 ns，在线，MPS 30%）在 183 的 2080 Ti 上做一次：
- `run --prod-ns 1.2 --online --mps-percentage 30`：应从 chunk 10 接着跑，写新文件 `pbsa_from_chunk_0010.csv`，`complete: true`，120 个 step 无缺无重。
- 再跑 `strip`：`frame_times.csv` 应有 120 行，前 100 行与延长前逐行相同。

S2 加副本走的是原 Hou 生产的同一条路（§20.1），不另外冒烟。

## 5. 计划表（正式运行，节点由用户指定，qsub 由用户执行）

| 项 | 内容 |
|---|---|
| 命令 | 新 wrapper `hou_sampling.sh`：每张卡一个 worker，各自起 MPS daemon（照 `hou_online.sh`）。每个 worker 按轮转领一部分副本，依次 `run --prod-ns 10 --online --mps-percentage 30`，然后 `strip`；先排 S1 的 21 个延长，再排 S2 的 21 个新副本（新副本先 `setup`，已有的会跳过）。PBS 脚本照模板 |
| 节点 | 2 × 3090 节点一个（约 14 h）或两个（约 7 h）。walltime 24 h |
| 能否续跑 | 能：原命令重跑，已提交的 chunk 和在线段会跳过 |
| 输出 | 原副本目录里追加 chunk 50–99 与新的在线段；新副本 `md/hou/<s>/ph74_gaff2/rep04–06/`；汇总脚本运行后写 `reports/hou_sampling_<日期>_*.csv` |
| 改动 | `manifests/runs.csv` 加 21 行（rep04–06，新种子，先跑 seed_collisions 检查）；现有汇总脚本 `summarize_hou_online.py` 会把新行算进去，没跑完时会报错，跑完前不要运行它 |
| 现有产物 | 0–5 ns 的 chunk、在线 CSV、`scores/h0.4`、`reference/` 都不改。`strip` 会重写 `solute_dry.dcd` 和 `frame_times.csv`（追加后 500 帧，前 500 帧帧号不变）；已有评分按 step/time 绑定，不受影响。延长前先把 21 个 `status.json` 复制到 `status_before_extend_20261028.json` |
| 这批结果决定什么 | 每个样本对在 6 × 10 ns 下能否分辨，作为 Hou 部分的最终采样结论；各副本的状态和姿态布居（例如 a4 的三种姿态、a6 的 Asp189 相互作用）出现多少次；漂移的副本有没有走到平台 |

## 6. 结果怎么读

- 每个样本报告：6 个副本各自的 (0, 10] 均值、两半的均值、等权均值 ± 副本标准误；用 §20.6 的方法给每个副本标出姿态和关键接触。
- 相邻样本对：列出分差、合并副本标准误、§20.7 的数值修正量。"可分辨"用什么读法还没定（见 HOU_H04_NUMERICAL_ACCEPTANCE_PLAN 的"未确认"一节）。
- 不算实验相关性；做不做实验值统计仍由用户定。

## 7. 执行记录

### 7.1 准备与冒烟（2026-10-28，183 2080 Ti，19:28–19:32）

- `manifests/runs.csv` 加了 21 行（a1–a7 的 rep04–06）：种子 velocity 20261069–20261089，integrator 基数 61000001–81000001（每行相差 1e6），barostat 为 integrator 基数 + 500000。沿用原文件的 CRLF 行尾。原文件备份为 `manifests/runs.before_rep04_20261028.csv`。dry-run 的 seed_collisions 检查：新行和原副本延长到 100 个 chunk 时都为空。
- wrapper `hou_sampling.sh`、作业脚本 `hou_sampling.pbs`（host 待填；两个 worker 各分到 21 个任务，按 §5 的速度估约 14 h）。
- 冒烟 `acceptance_online_20261028/smoke_extend.sh`（日志 `smoke_extend.log`）：把小包里已跑完的 `acconline` 从 1 ns 原地延长到 1.2 ns，在线评分 + MPS 30%。
  - 从 chunk 10 接着跑，新开一段 `pbsa_from_chunk_0010.csv`：20/20 帧，0 标记，最大延迟 0.68 s。
  - 延长后 `complete: true`、`online_valid: true`。120 个已提交 step 无缺无重；原来两段（0–2 被中断、2–10）保留。
  - `strip` 后 `frame_times.csv` 有 120 行，前 100 行与延长前逐字节相同。
  - 延长段 worker 的预热是 121 s：pilot 换成了 100 帧，编译缓存没命中。每 100 ps 用 43–44 s（2080 Ti）。

### 7.2 正式运行（2026-10-29 起，fulda）

- 用户 2026-10-29 原话："欸？？？/完了忘记提交了。。稍等 你这样  cyanus@192.168.4.199上面跑"。fulda（RTX 5090 32 GB，24 核）只有一张卡，所以用 1 个 worker，42 个任务串行：`MODE=worker W=0 NW=1 GPU=0 TAG=fulda1029 bash hou_sampling.sh`。02:24 启动，PID 记在 `pids_fulda.txt`，日志 `hou_sampling_w0.log`。没有用 PBS，`hou_sampling.pbs` 的 host 仍是占位。
- 启动前核对过：fulda 的 NFS 挂载属性缓存是 6 h（`acregmin=21600`），它读到的 `runs.csv`、`hou_sampling.sh`、`run_smallmol_md.py` 与本机 md5 相同；JAX 设备是 `CudaDevice(id=0)`。
- 实测：a1 rep01 从 chunk 50 接着跑，在线 worker 预热 32 s；每 100 ps 用 14.9–15.7 s（含在线评分），约 150 s/ns，是 3090（298 s/ns）的一半。GPU 利用率 92%。
- 预计：S1 21 × 5 ns × 150 s ≈ 4.4 h；S2 21 × 10 ns × 150 s ≈ 8.8 h，加上 NVT/NPT（5090 上没测，按 3090 的一半估约 0.8 h）和每个任务的启动与 strip（约 0.5 h），合计约 14.5 h，预计 10-29 17 时前后结束。能续跑：原命令重跑会跳过已提交的部分。

### 7.3 运行结果与修复（2026-10-29）

- worker 在 18:44 结束，42 个任务都记为 FAIL：
  - 41 个：MD 和在线评分完整，收尾的在线覆盖核对报 "wrong time"。OpenMM 的时间是每步累加 dt，到 9.91 ns（step 4 955 000）偏差 1.0003e-6 ps，超过了 `smallmol_records.online_rows` 的 `abs_tol=1e-6`；0–5 ns 内最大偏差是 1.0e-7。
  - 1 个（a5 rep05）：约 5.5 ns 处 PBSA 积压（max_pending=2），进程报错退出，原因没查，可以续跑。
  - 被标记的帧全部是配体盒 margin < 0（−0.0…−0.42 Å；C/R margin 10–15 Å）：a3 rep01 延长段 38/500，a3 rep02 延长段 53/500，a1 rep05 6/1000。怎么处理待用户定。
- 修复（用户 2026-10-29："改容差、跑相关测试 … 跑"）：`scripts/smallmol_records.py` 的 `online_rows` 容差改为 `0.25 * dt`。依据：偏差只要小于 dt/2，就仍能唯一对应一个 step；真实的错时（例如续跑后时钟归零）至少差一整步。到 10 ns 时浮点累加偏差约 1e-6 ps，比 0.0005 ps 小两个多数量级。另外两处 `abs_tol=1e-6`（`scored_frames`、`sensitivity_frames`）比较的是 strip 写出的同一个 `round(step*dt, 6)`，不受影响，没有改。
- 验证：相关回归 99 项通过（183，只作回归检查）；在 fulda 上用真实数据重跑 `online_rows`，42/42 个副本通过，a1 rep01 是 1000 行（原来报错的那一处）。还没有对这些副本重跑收尾（`run` 核对 + `strip`）。

### 7.4 收尾运行（2026-10-29 22:21 起，183，未完成时交接）

- 用户 2026-10-29 原话："这样 你丢183 跑"。命令：在 183 上运行 `MODE=worker W=0 NW=1 GPU=0 TAG=k183_1029 bash hou_sampling.sh`，用 setsid 启动，自成会话（SID 59833），不依赖 Claude 窗口。PID 59832/59833 记在 `pids_183.txt`，日志追加在 `hou_sampling_w0.log`（从 "worker 0/1 start host kasuga02 2026-10-29T22:21" 那一行起）。
- 启动前核对：183 与 fulda 读到的 `status.json`、日志、改后的 `smallmol_records.py` md5 相同；183 的 GPU 空闲。
- 预期：41 个副本 `run` 秒级判定 `complete`，然后 strip；a5 rep05 从约 chunk 55 续跑到 100（2080 Ti 约 30 多 min）。合计约 1 h。
- 下一个窗口接手时先核对：`ps -o pid,etime -p 59833`（应已退出）；日志里有没有 "worker 0 done"；从 22:21 那一行起有没有 FAIL；每个副本的 `status.json` 是否 `complete: true`；`analysis/frame_times.csv` 是否 1000 行。之后再处理那 97 个被标记的帧（§7.3，待用户定），然后写汇总。

### 7.5a 收尾核实（2026-10-30，Codex，恢复前快照）

- 183 收尾日志从 10-29 22:21:56 到 23:14:12 `worker 0 done`，该段无 FAIL；10-30 SSH 核实原 PID 59833 已退出。此前未完成的 a5 rep05 已补齐。
- stdlib-only 完整性审计：42/42 个副本 MD 与在线覆盖完成，均有 1000 个已提交 step，strip frame/step/time 逐行匹配，共 42,000 帧。
- 39/42 副本逐帧有效；a1 rep05、a3 rep01、a3 rep02 分别有 6/38/53 个配体盒越界帧。运行完成不代表所有分数有效；a1/a3 的 6 × 10 ns 完整窗口均值暂不输出。
- 新报告在 `reports/hou_sampling_20261030_online/`，越界恢复脚本和计划见 [HOU_SAMPLING_RECOVERY_HANDOFF_20261030.md](HOU_SAMPLING_RECOVERY_HANDOFF_20261030.md)。只准备了对 3 个完整副本离线重评的入口，尚未启动新的 GPU 作业；原始 MD/在线评分保留，停止追加采样。
- **结果（2026-10-30 核对）**：worker 于 2026-10-29 23:14 结束（"worker 0 done"）。从 22:21 那一行起，run 42 次、strip 42 次、FAIL 0 次。a5 rep05 从断点续跑到 chunk 100，续跑段 0 标记。42/42 个副本 `complete: true`、`chunks_done` 100、`frame_times.csv` 1000 行。被标记的 97 帧（§7.3）仍待用户定，之后写汇总。

### 7.5 被标记帧的离线重评与汇总（2026-10-30，183）

- 用户 2026-10-30 原话："是的"（回答"要我按这个办法离线重评这 3 个副本吗？"）。`rescore_flagged_w10.sh`：a3 rep01、a3 rep02、a1 rep05，默认网格，pilot 用 (0, 10] ns 的全部帧；18:29–18:49，3000 帧全部有效、0 标记；输出在 `scores/window10_default/`。
- 汇总：包内 `scripts/summarize_hou_sampling.py`，结果见 RESULTS §20.9。

### 7.6 后续严格复核与实验统计（2026-10-30）

恢复产物经过 `scripts/summarize_hou_sampling.py` 的输入绑定和完整覆盖复核，42,000 帧可用，来源明确分为 39 个在线副本和 3 个完整副本离线恢复。后续排序分析按原设计 §9、冻结默认网格执行，结果见 RESULTS §20.10 和 `reports/hou_ranking_20261030/`。预算不再扩大；不依据相关性更换网格、窗口或副本。

## 未确认（Claude 推断）

- 第 3 节的标准误预期假设副本之间的离散不变；如果新副本落到新的状态，离散可能更大。
- 6 × 10 ns 是按"标准误降到约 1/√2、预算在一两个节点日以内"选的，不是由什么推导出来的。更便宜的做法（比如只做 S1，约 8.7 GPU-h）只能回答漂移问题，回答不了布居问题。
