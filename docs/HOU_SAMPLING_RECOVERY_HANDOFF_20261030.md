# Hou 6 × 10 ns 收尾与网格越界恢复（2026-10-30，Codex）

**当前状态更新：** §1–§4 保留恢复前快照。实际重评已由 `rescore_flagged_w10.sh` 在 183 完成，输出路径是 `scores/window10_default/`，无需再启动 §3 的替代 wrapper。最新严格复核、报告网格与实验统计见 §5，以及 RESULTS §20.9–20.10。

## 必读（开工前全部读完）

| 路径 | 用途 |
|---|---|
| `CLAUDE.md` | 冻结默认、生产环境和节点分工。 |
| `docs/HOU_BOUNDED_SAMPLING_PLAN_20261028.md` | 固定的 6 × 10 ns 预算、完整窗口、停止条件与 10-29 收尾来源。 |
| `docs/HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md` §8 | 已完成的数值检查；h0.4 尚未证明收敛，不再加密。 |
| `docs/HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md` §8 | 已完成的在线成本、中断恢复与同 pilot 回放；不能重复列为尚未执行。 |
| `RESULTS.md` §20.6–20.9 | 当前实测及本次 CSV 完整性审计。 |

## 1. 当前核实的事实

本次用户指示是“好了继续推进了”。以下审计和恢复方案由 Codex 实现；不把方案写成用户已经选择的科学口径。

- 10-29 在 183 的收尾已结束：`hou_sampling_w0.log` 最后一段从 22:21:56 开始，到 23:14:12 `worker 0 done`，该段没有 FAIL。
- 10-30 通过 SSH 对 183 实查：PID 59833 已不存在；GPU 0 是 RTX 2080 Ti，显存占用 1 MiB，利用率 0%，无计算进程。这是检查时快照，不保证以后仍空闲。
- 42 个副本 `complete/md_complete/online_complete` 全为 true，每个 100 个 chunk、1000 个已提交在线 step；strip 的 1000 行 frame/step/time 逐行对应，无缺无重。合计 420 ns、42,000 帧。
- `complete` 只说明运行与覆盖完成。39 个副本的 `online_valid` 为 true，另外 3 个有 97 个越界帧；这 3 个仍不能作为完整有效评分使用。

| 副本 | 越界帧 | 最早时间 ps | 最低配体 margin Å |
|---|---:|---:|---:|
| a1 rep05 | 6 | 6080 | −0.1134 |
| a3 rep01 | 38 | 6230 | −1.3838 |
| a3 rep02 | 53 | 6450 | −0.5275 |

这里的最低 margin 是对全部已提交帧重新读取所得，取代旧交接中未完整覆盖的范围估计。未删帧、未放宽 margin 守卫、未修改原始在线 CSV 或 status。

## 2. 已完成的分析交付

入口：`scripts/summarize_hou_sampling.py`，stdlib only，不加载 OpenMM、JAX、DCD，不做新的分子计算。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
  scripts/summarize_hou_sampling.py \
  --out experiments/2026-10-17_small_molecule_benchmark/reports/hou_sampling_20261030_online
```

输出目录下：

- `audit.json`：输入清单、源码和 status/CSV/frame_times 的 SHA256，覆盖、有效性、限制与待恢复副本。
- `replicas.csv`：42 个副本的完整窗口状态、分数与 margin；有坏帧的副本不输出完整窗口均值。
- `blocks.csv`：420 个 1 ns 块的分项；块内有坏帧就留空该块均值，不丢坏帧后重算。
- `samples.csv`：原始 3 × (0,5] 基线、原 3 × (0,10]、全 6 × 两个半段、全 6 × (0,10]。每个副本先取时间均值，再等权平均；所需副本任一窗口不完整有效，则该样本结果留空。
- `pairs.csv`：全部 21 对的分差与合并副本标准误；涉及缺失样本的行留空，不自动判断显著或可分辨。
- `original_online_flagged.csv`：97 帧的完整索引与 margin，全部保留。

当前默认网格的条件性采样诊断如下，单位 kcal/mol。标准误来自独立副本的时间均值，不把 1 ns 块当成独立副本：

| 样本 | 原 3 × 5 ns 均值 ± 副本 SE | 全 6 × 10 ns 均值 ± 副本 SE | 全 6 副本极差 |
|---|---:|---:|---:|
| a1 | −32.28 ± 4.29 | 待恢复 | — |
| a2 | −21.83 ± 1.30 | −22.58 ± 0.59 | 4.04 |
| a3 | −22.57 ± 2.40 | 待恢复 | — |
| a4 | −19.99 ± 0.39 | −19.33 ± 0.87 | 5.62 |
| a5 | −16.57 ± 0.58 | −15.88 ± 1.05 | 7.76 |
| a6 | −21.66 ± 3.39 | −18.40 ± 2.00 | 11.61 |
| a7 | −32.51 ± 1.01 | −33.39 ± 0.80 | 5.87 |

可确认的采样现象：a2 的副本 SE 缩小；a4、a5 的副本差异扩大；a6 的极差仍约 11.6。新副本可能访问了先前未充分覆盖的状态，具体姿态原因尚待分析。预算已经跑满，按原计划停止追加 MD。

## 3. 待启动的有界恢复方案（未执行）

目的：恢复 a1/a3 的完整评分覆盖，并量化重新定尺影响；不提高实验相关性，不补采样，不改变默认网格间距或 PB 模型。

使用已跑通过的 `score_smallmol_replica.py --grid default --pilot window`，对上述 3 个副本分别重评完整 (0,10]，每个 1000 帧。pilot 包含该副本完整窗口，用于一次性固定 C/R 和 L 网格。原来的 NPT pilot 在线结果不改；新输出是**离线恢复**。

采用完整副本重评而不是只补 97 个坏帧，是因为全窗口重新定尺会改变网格尺寸/相位，原先有效帧的分数也可能变化。后续汇总将整副本选为离线来源，并分别报告原有效帧、原坏帧、全部帧的配对分项变化。若仍有坏帧或绑定不一致，拒绝纳入恢复结果。

| 项 | 具体内容 |
|---|---|
| wrapper | `experiments/2026-10-17_small_molecule_benchmark/hou_recover_margin_20261030.sh` |
| 命令 | 见下；默认 DRY_RUN=1 只打印计划，无 GPU 初始化 |
| 资源建议 | 183 / GPU 0 RTX 2080 Ti；1 GPU、8 CPU 线程；启动时再次检查 GPU 无计算进程，有则退出 |
| 规模 | 3 × 1000 = 3000 帧；零新增 MD |
| 耗时估计 | 旧 183 离线实测约 284 ms/帧，加每副本约 2 min 初始化，约 20–30 min；扩盒可能增加成本，此估计不是新实测 |
| 续跑 | 同命令重跑按协议和已保存 frame 跳过；不删除历史产物 |
| 新分数 | `scores/margin_recovery_20261030/default/hou/<sample>/ph74_gaff2/repNN/` |
| 新汇总 | `reports/hou_sampling_20261030_recovered/`，含 `recovery_paired_differences.csv` |
| 验证 | 全 1000 frame/step/time、逐帧有效性、MD 输入摘要、模型与精度、边界阈值全部匹配；保持历史原 3 × 5 ns 基线原值 |

```bash
# 只读 dry-run：已通过。
bash /home/ruigengji/JAXPBSA/experiments/2026-10-17_small_molecule_benchmark/hou_recover_margin_20261030.sh

# 在授权节点上执行；本次尚未启动。
GPU=0 DRY_RUN=0 bash /home/ruigengji/JAXPBSA/experiments/2026-10-17_small_molecule_benchmark/hou_recover_margin_20261030.sh
```

## 4. 验证与下一步

- 新增分析保护回归：`tests/test_hou_sampling_summary.py`，覆盖坏帧/缺副本不能生成 cohort 均值、等权副本统计、strip 重复帧、恢复 CSV 缺帧/重复/错时/坏帧/NaN margin/错网格、输入变化拒绝。
- wrapper `bash -n` 和 3 个真实副本的 CLI dry-run 通过。尚未声称扩盒重评在 GPU 上成功。
- 恢复后首先审阅原有效帧的配对变化；以明确来源报告混合了在线与离线恢复的分数，不能把它叫作 42 个副本在线全部有效。
- 之后只做固定预算内的分项/半段/姿态诊断与报告。不同速度都来自同一晶体起点，状态出现次数不是平衡布居；无自动的 2SE 显著性门；不运行实验相关性优化。
- Reason cloud 账号状态为 `login_required`，本次未创建云端任务，未上传项目数据，未发生云端计算。

## 5. 恢复完成与实验排序（2026-10-30 更新）

- 另一个执行入口 `rescore_flagged_w10.sh` 已于 18:29–18:49 在 183 对三个副本各重评完整 10 ns，3000 帧全部有效，结果在 `scores/window10_default/hou/`。不重复启动本交接 §3 的替代重评批次。
- 本轮重新使用严格审计器，`--recovery-root experiments/2026-10-17_small_molecule_benchmark/scores/window10_default/hou`，输出 `reports/hou_sampling_20261030_verified/`：42/42 个副本的完整评分可用，输入/默认模型/精度和 frame/step/time 绑定通过。原在线评分仍只有 39/42 副本有效，三个离线来源明确保留，原 97 个标记不抹掉。
- 报告配置：沿用冻结默认 C/R 0.5 / L 0.25；h0.4 作为数值敏感性对照，不按实验相关性选择。完整 6 × 10 ns 的 h0.4 分数没有运行过，不混称为同预算比较或已收敛。
- 按设计 §9 完成统计：同系列 a1–a4 ρ=1.00、τ-b=1.00、6/6 对一致；全七条 ρ=0.75、τ-b=0.619、17/21 对一致；a5–a7 的补充结果为 3/3 一致。双侧精确 p 分别为 0.0833、0.0663、0.3333（Spearman）。不把小集合中的完美次序写成稳健泛化证据。
- 10,000 次副本＋连续时间块（1/2/5 ns）重采样：ρ 百分位范围 a1–a4 为 0.80…1.00，全七条为 0.571…0.929。候选组成重采样独立计算，全七条为 −0.059…1.00。未知实验误差不补造，同化学系列不自动等于同测定条件组。
- 来源、全部成对次序、图和复现方法在 `reports/hou_ranking_20261030/` 与 `scripts/rank_hou_sampling.py`。新排序回归和完整性回归共 20 项通过。没有新跑 MD/PBSA。

复现（输出使用新目录，保留本次结果）：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/jaxpbsa-ranking-matplotlib \
  /home/ruigengji/miniforge3/envs/openmm_dev/bin/python scripts/rank_hou_sampling.py \
  --analysis-dir experiments/2026-10-17_small_molecule_benchmark/reports/hou_sampling_20261030_verified \
  --out /tmp/jaxpbsa-hou-ranking-reproduction --bootstrap-draws 10000 --seed 20261030
```
