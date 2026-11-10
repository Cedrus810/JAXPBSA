# JAXPBSA 工程修复交接（2026-10-24，JST）

**最新状态：第二轮工程修复已完成，完整工程回归 131 passed / 2 deselected。**
第一轮记录保留在 §1–§5，第二轮新增修复、兼容性与验证见 §6。

用户范围：修复整体 review 发现的工程问题，并交付可接续的报告。工作区为
`/home/ruigengji/JAXPBSA`；保留进入本轮前已有的未提交改动。没有提交 Git、提交 PBS、
运行 MD、参数化、Amber 或真实 PB/SA 求解，也没有调整冻结的数值模型或网格默认值。

## 1. 修复清单

| 问题 | 修复后的行为 | 位置 |
|---|---|---|
| MD chunks 完成后，在线 worker 尾部失败；再次运行会直接写 complete=true | `complete` 同时要求 MD 完成和已提交区间的在线 step 完整覆盖。启动时先核对已有区间；缺帧、重复、错误时间标签或残缺行会报错。worker 排空失败也保存 session、已提交范围、计数和错误。 | `scripts/run_smallmol_md.py`、`scripts/smallmol_records.py` |
| 评分续跑只检查窗口/stride，崩溃后甚至没有 summary 可检查 | 第一个结果写出前保存 `protocol.json`，绑定 run、输入文件实际内容、窗口、stride、分块和显式分析参数。续跑还检查 frame/step/time、重复行、CSV 表头和尾部完整性。MD 续跑增加身份、种子、输入文件和动力学参数核对。 | 同上，及 `scripts/score_smallmol_replica.py` |
| Amber comparison.csv 的时间标签循环错位 | 按同一个零起始序号配对能量与时间，`used[j-1]` 改为 `used[j]`；提供独立的静态元数据修复脚本。 | `scripts/amber_reference_smallmol.py`、`scripts/repair_smallmol_reference_times.py` |
| 收敛标记正常时，NaN/Inf 的 MM 或总能量可能不被 flag | reporter 检查能量、面积和 C/R margin 的有限性；预热阶段拒绝非有限能量。共享网格特意使用的 NaN ligand margin 仍允许；非法 flag 阈值被拒绝。 | `jaxpbsa/online.py` |
| warm trajectory 把失败帧的势传给下一帧 | 未收敛或含 NaN/Inf 的势不进入下一帧 carry，而是清零；失败帧本身的诊断保留。 | `jaxpbsa/pb/energy.py` |
| 不支持的非键表示可能被部分读取并静默评分 | 要求恰好一个 NonbondedForce；拒绝额外 CustomNonbonded/GB/Amoeba 表示及参数偏移。检查参数有限性、R/L 跨边界 exceptions/共价键、重复或非整数索引。 | `jaxpbsa/openmm_io/system.py`、`jaxpbsa/mm/cross.py`、`jaxpbsa/online.py`、`jaxpbsa/online_async.py` |

续跑修复时另处理了两处相关问题：

- `grid_sensitivity_smallmol.py` 改为按 `(config, frame, rotation)` 续跑，配置中有一行不再等同于整配置完成；输入、参考帧、旋转种子和网格参数也写入 protocol。
- `summarize_hou_online.py` 使用同一套覆盖检查和逐帧有效性规则，坏帧不进入排序；1 ns 分块使用 step 推导的时间，避免浮点累计时间使边界帧重复进入两个块。

## 2. 存量产物处理

### BRD4 同帧对照

已修正 q3 的 **30 份 comparison.csv / 90 帧**，以及重新生成的
`reports/reference_q3_frames.csv` 时间标签。frame 124/249/374 现在分别对应
1250/2500/3750 ps；step 为 625000/1250000/1875000。

每份原 CSV 备份为同目录 `comparison.before_timefix_20261024.csv`；合并表备份为
`reports/reference_q3_frames.before_timefix_20261024.csv`。修复脚本逐字段断言：
**除 step/time_ps 外，frame、能量、flag 等全部字段不变。** 修复后又逐帧核对了
全部 90 帧的 step/time 关系。记录在：

`experiments/2026-10-17_small_molecule_benchmark/reports/reference_timefix_q3_20261024.json`

该记录保留每个文件的修复前/后 SHA256 和备份路径。此修复不需要重跑 Amber。

### Hou 在线汇总

按更严格的覆盖与有效性检查重新汇总：**21 条副本、10,500 帧、0 坏帧**。
`hou_online_samples.csv` 和 `hou_online_ranking.json` 与本轮前逐字节一致；
修正分块边界不改变样本均分和排序。原三个汇总文件保存在
`reports/hou_online_*.before_contractfix_20261024.*`。

没有修改 MD 状态文件、原始在线 CSV、轨迹或主评分能量。

## 3. 后续运行时需要知道的兼容性

1. **历史评分文件仍保留。** 历史 `frames.csv` / `sensitivity.csv` 没有 protocol.json，
   无法从现存 summary 反推当时的完整输入。因此不会自动给旧结果补一个看似可信的绑定，
   也不会删除或改判旧结果；新的评分操作需要显式给一个新的 `--out`。
   新建输出目录之后，同输入、同协议可继续使用原命令续跑。
2. **MD 历史来源不能追溯补证。** 历史 status 首次在新入口续跑时，先核对已有身份、
   种子、步数设置，再绑定当前输入，标记 `legacy_input_history_unverified=true`。
   这个标记不表示旧 MD 作废，也不代表已证明历史输入从未改变。新运行从开始就记录绑定。
3. **缺在线评分时保留已完成 MD。** 入口会明确报错；不能把补算伪称为当时完成的在线分析。
   可以在指定节点对已有轨迹执行 strip，再用 `score_smallmol_replica.py --out <新目录>`
   恢复离线评分。该补算不会自动把历史在线运行改成完整。
4. **CSV 名称以 status 为准。** 正常仍为 `pbsa_from_chunk_NNNN.csv`；同一 chunk 重试遇到已有
   文件时添加唯一后缀，避免覆盖失败 session。汇总按 status 中的路径及提交范围读取。
5. **complete 是运行完整性。** `online_valid` 另表示已保存帧是否通过现有规则；完整性、
   网格精度、采样稳健性和实验排序是分别需要验收的事项。

新离线输出示例（在用户指定的计算节点运行；本轮没有执行）：

```bash
IN=experiments/2026-10-17_small_molecule_benchmark/manifests/hou_inputs.csv
python scripts/score_smallmol_replica.py --inputs "$IN" --run-id a1_ph74_gaff2_rep01 \
  --out experiments/2026-10-17_small_molecule_benchmark/scores/replay_offline/hou/a1/ph74_gaff2/rep01
```

输入内容绑定用于阻止混写不同输入；没有以源码摘要自动作废既有结果或编译缓存。

## 4. 验证与边界

新增测试：`tests/test_engineering_contracts.py`、`tests/test_smallmol_records.py`。
覆盖末尾缺分后的实际 cmd_run 重入、完整 CSV 重入、worker 失败落盘、段重叠/缺失/错误时间、
输入及种子漂移、无 summary 的评分续跑、残缺 CSV、部分网格配置、非有限能量、
静态 OpenMM 不支持表示、索引错误、合成 warm-state 失败与恢复、参考时间修复幂等性。

验证命令：

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_engineering_contracts.py tests/test_smallmol_records.py \
  tests/test_online_async.py tests/test_online_overhead.py tests/test_validate_mmpbsa.py \
  -k 'not real_analyzer and not openmm_keeps_stepping'
```

工程回归结果：**94 passed，2 deselected，16.17 s**。最后补齐 `md_complete` 在重入及
chunk 提交时的状态更新后，续跑/记录测试单独复验 **20 passed，0.08 s**。
额外做了 14 个改动 Python 文件的语法解析、CLI dry-run、
真实存量 CSV 的只读/汇总检查和 `git diff --check`。测试使用静态 System、桩 worker 和
合成数组；未建立 MD Context 或调用真实 PB/SA。排除的两项涉及真实分析器或 OpenMM 步进。

另外对真实保存的 BRD4-3、BRD4-6、Hou a1、Hou a6 System 做了纯静态参数读取及跨 R/L
exceptions 检查，4/4 通过，没有建立 Context。q3 时间修复再次 dry-run 显示 0 个待修复文件，
确认幂等。新增测试及修复脚本均为本轮文件；原工作区本来就有未提交的小分子脚本和结果，
不能仅用 `git diff --stat` 的已跟踪文件数作为本轮交付文件数。

**仍需指定节点验证：** 修复后真实体系的短在线运行、一次中断/续跑、相同 pilot/网格下
在线与离线逐帧对照，以及真实求解器 warm-start 故障恢复。工程测试不代替这些验收。

## 5. 数值与采样工作的交接

Hou 当前完成 7 × 3 × 5 ns 在线运行。实验次序大体一致，但 a1/a6/a3 的副本极差约
14.73/11.34/8.12 kcal/mol，a3/a4 有待核实的姿态变化；Hou 尚未完成同帧网格/Amber 对照。
本轮工程修复不解决这些数值与采样问题。

本轮只读取已有 BRD4 sensitivity.csv，另确认了偏差所在的分项。每个样本 9 帧，
默认网格与 h0.35/约0.167 的均值差如下（kcal/mol，四舍五入）：

| 样本 | Δ(C−R) | ΔL | ΔPB = Δ(C−R)−ΔL |
|---|---:|---:|---:|
| BRD4-3 | +2.81 | +0.09 | +2.73 |
| BRD4-6 | +2.70 | +0.08 | +2.62 |

这组快照上的主要变化来自 **C/R 差分**，不能继续用“误差都在孤立配体”解释。
下一步建议顺序：

1. 在指定节点验收修复后的短在线/中断续跑及同帧回放一致性。
2. 固定配体网格，单独检查 C/R 网格、表面和取向敏感性；给 Hou 做预定同帧数值对照。
3. 联合看 Hou a1/a6/a3 的逐副本分项和 a3/a4 姿态，之后再决定有界的补采样。
4. 在冻结精度配置下补同卡 baseline/同步/异步/MPS 的 MD 开销、延迟、积压和排空成本。

工程修复和后续路线均不以提高实验相关性来选择参数、删帧或调整时间窗。

## 6. 第二轮工程修复与验证（2026-10-24）

用户要求继续修理工程问题。本轮在第一轮基础上修复了以下五类问题：

| 问题 | 修复及接手注意事项 |
|---|---|
| 缓存 setup 仅凭 setup_record.json 存在便跳过 | 新 setup 保存来源文件内容与 R/L 选择的 input_contract。setup 跳过、MD 运行、离线评分、网格敏感性入口均先核对 setup 与当前 manifest。来源路径、原子数、样本身份或 R/L 范围不匹配时，在分子运行之前报错。 |
| 在线续跑未绑定 NPT pilot | 新在线运行在生产前绑定 pilot 内容、实际分析器默认参数、PBParams、精度与评分步频。续跑时先核对；pilot 被替换或数值配置改变会在创建生产 Context 前报错。MPS 限额等调度选项不作为数值模型绑定。 |
| 部分 CSV 可被误认为完整结果 | 统一检查所需列、重复表头、额外/缺失单元格、数值和布尔字段。评分与网格敏感性入口要求完整能量字段。在线审计拒绝越过已提交 MD 的窗口、非法段范围、缺失/非有限的 margin 阈值，不再把未知阈值默认为零。 |
| MD 时间参数被 round 静默取整，在线 pilot 可能不足 | 时间请求必须对应整数个积分步；不可表示的 dcd/chunk/NVT/NPT/production 时间直接拒绝。在线 NPT 至少须产生两帧 pilot，dry-run 也检查该条件。默认时间设置不变。 |
| 离线生产续跑或延长后，Context 的 step/time 从零重启 | 所有生产模式都从已提交 chunk 恢复生产时钟，日志与报告帧保持同一 step/time；不改变坐标、速度或随机流的原有恢复方式。 |

相关源文件：

- `scripts/smallmol_records.py`：setup/pilot 绑定、统一 CSV 校验、整数步检查与提交范围检查。
- `scripts/run_smallmol_md.py`：setup 输入记录、生产前检查、在线绑定和离线生产时钟恢复。
- `scripts/score_smallmol_replica.py`、`scripts/grid_sensitivity_smallmol.py`：使用统一 setup/完整能量字段校验，并记录 legacy setup 的来源限制。
- `tests/test_smallmol_preflight.py`：新增 37 项预检、损坏记录和桩运行回归；已有 `test_smallmol_records.py` 相应更新。

### 兼容性与失败恢复

- 17 套现有 BRD4/Hou setup 的身份、路径和原子选择均通过只读检查。本轮没有重建或修改这些 setup。
- 历史 setup 没有 input_contract，无法补证当时的源文件内容。继续使用时显式记录
  `legacy_setup_input_history_unverified=true`；不会据此删除、自动重建或改判既有结果。
- 历史在线运行没有 pilot/分析参数绑定时，首次继续生产只能绑定当前输入，并记录
  `legacy_online_input_history_unverified=true`；不声称已追溯证明过去的 pilot 完全一致。
- 在线文件的最后一条未换行残缺记录只作为未完成写入忽略。之后仍核对全部已提交 step：
  残缺的是未提交尾帧时不阻止恢复，残缺的是已提交帧时仍明确报缺分。文件本身不截断、不覆盖。
- 历史段若没有 `margin_flag` 或 `sizing.margin_flag`，现在必须补明其真实配置再作有效性验收，
  不会按零阈值降低要求；现有 Hou 21 条副本均包含该配置。
- 第一轮关于历史无 protocol.json 的评分目录须使用新 `--out` 的规则继续适用。

### 第二轮验证结果

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_engineering_contracts.py tests/test_smallmol_records.py tests/test_smallmol_preflight.py \
  tests/test_online_async.py tests/test_online_overhead.py tests/test_validate_mmpbsa.py \
  -k 'not real_analyzer and not openmm_keeps_stepping'
```

结果：**131 passed，2 deselected，15.78 s**。两个排除项仍涉及真实分析器或 OpenMM 步进。
离线延长时钟回归驱动了实际 cmd_run 控制流程，使用无物理计算的桩引擎，不是分子 MD。
另验证了 6 个变更 Python 文件的语法、正常 CLI dry-run、非整步时间的 dry-run 拒绝和
`git diff --check`。

存量数据只读检查：BRD4 10 套、Hou 7 套 setup 兼容；Hou 21 条副本的 10,500 帧通过更严格的
CSV/step/阈值检查，0 坏帧。本轮没有改写原始 MD、评分、实验排序或上一轮修正的参考时间表。
机器可读验证记录保存在 `reports/engineering_repair_validation_20261024.json` 的 `round2` 字段。

尚未执行真实 GPU 在线中断/续跑、真实 PB warm-start 故障恢复或新的网格/采样实验；
这些仍按 §4–§5 交由指定节点验收。本轮确认的五类工程问题已修复，不再作为待实现项。

## 7. 补修：评分续跑不再按 DCD 内容摘要绑定（2026-10-25，Claude）

问题：`score_smallmol_replica.py` 和 `grid_sensitivity_smallmol.py` 的 protocol.json 记录了
`solute_dry.dcd`（以及 `frame_times.csv`）的 SHA256。可是 `strip` 每次调用都会从头重写 DCD，
文件头里还带着创建时间（实测 `REMARKS Created 18 October, 2026 at 08:45`）。所以作业在评分途中
中断后重新 qsub（`brd4_full.sh` 依次执行 strip→评分），或者 MD 延长之后，坐标虽然没变，评分也会被
`scoring inputs/protocol changed` 拒绝。

修复：两个入口的 protocol 升到 schema 2，轨迹改为按路径和原子数绑定。评分续跑时，已保存帧的
step/time 原本就由 `scored_frames` 逐帧和 `frame_times.csv` 核对；网格敏感性额外把参考帧的
(frame, step, time_ps) 写进 protocol。system.xml、setup_record.json、prmtop 仍按原样做摘要绑定。
已有的 schema 1 protocol.json（如果有的话）和新版本不一致，仍按原规则换一个新的 `--out`。

验证：拿真实的 BRD4-1 rep01 DCD 复制一份，只改文件头里的时间（模拟重跑 strip）：旧的 SHA256
绑定不再相等，新的绑定相等，坐标逐位相同。相关测试 99 passed（只是回归检查，这三个测试文件都
没有覆盖这两个入口的 protocol）。**还没有在真实运行里走过"strip→评分→再 strip→评分续跑"。**

### 7.1 冒烟（2026-10-25 09:20–09:31，183 2080 Ti，Hou a1，单独的包 `experiments/2026-10-17_small_molecule_benchmark/smoke_repair_20261025/`）

脚本 `smoke.sh`，日志 `smoke.log`，PID 记在 `pids_183.txt`。不影响正式副本。

- **在线中断/续跑**：setup 13 s → NVT 10 ps → NPT 100 ps（10 帧 pilot）→ 生产 0.3 ns，`--online --mps-percentage 30`。
  chunks_done=1 时对 MD 进程发 SIGTERM（rc 143）；原样重跑同一条命令，从 chunk 1 接着跑，写新的
  `pbsa_from_chunk_0001.csv`，结束时 `complete: true, chunks_done: 3`；再跑一次，`skip_md_done: true, online_complete: true`。
  被杀的 session 在 0000 文件里多写了 step 55000–80000 共 6 行未提交的帧，`online_rows` 只取已提交范围，
  合起来 30 个 step（5000…150000，每 5000 一个，无缺无重），全部收敛，0 帧被标记，`online_valid: true`。
- **评分续跑（§7 的修复）**：strip（DCD 头 `Created … 09:27`）→ 评分到第 8 帧时 SIGTERM → 跨过一分钟后重新 strip
  （头变为 `09:30`）→ 同一个 `--out` 续跑：没有被拒，补齐 30/30 帧，0 帧被标记 → 再跑一次，0 帧新算。
  修复前，这一步会因为 DCD 的 SHA256 不同被拒。
- **同帧在线与离线**（30 帧，在线减离线）：E_coul 和 E_LJ 的差 ≤3e-4，ΔG_PB +0.004 ± 0.001（最大 +0.0065），ΔG_SA 0。
  两边的网格尺寸是否完全相同没有逐一核对（在线的 pilot 是 NPT 帧，离线的是窗口帧）。
- 耗时：生产 35–36 s / 100 ps（MD 加在线评分，约 355 s/ns）；在线 PBSA 0.52 s/帧，最大延迟 0.59 s；离线 284 ms/帧；
  worker 冷启动 131 s，有缓存时约 10 s。
- 另外看到一点：MD 进程被杀之后，worker 打出 `ConnectionResetError` 和 `BrokenPipeError` 两段 traceback
  （`jaxpbsa/_online_worker.py:54` 只捕获了 `EOFError` 和 `BrokenPipeError`）。worker 照样退出了，续跑不受影响，
  只是日志里多了这些输出。没有改。

还没做：真实求解器 warm-start 故障恢复；在节点上对正式副本做中断续跑。
