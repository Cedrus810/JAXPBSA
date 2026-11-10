# Hou 验收线交接：诊断 → 数值 → 在线成本 → 6 × 10 ns 采样 → 实验排序（2026-10-28 – 10-30，Claude）

## 必读（开工前全部读完）

| 路径 | 为什么要读 |
|---|---|
| `/home/ruigengji/JAXPBSA/CLAUDE.md` | 冻结基线、生产环境 `openmm_dev`、机器分工、PBS 选卡写法、4090/5090 的使用限制。 |
| `/home/ruigengji/JAXPBSA/docs/BRD4_HOU_EXECUTION_HANDOFF.md` 文首 | 当前优先执行入口；10-30 停点那一行是 Codex 写的总括。 |
| `/home/ruigengji/JAXPBSA/RESULTS.md` §20.6–§20.10（含 §20.9a） | 这条线的全部实测结论，结论以这里为准。 |
| `/home/ruigengji/JAXPBSA/docs/HOU_SAMPLING_RECOVERY_HANDOFF_20261030.md` §1、§5 | Codex 的收尾审计、严格复核与排序统计的复现命令；它的 §3 已作废（见下文 §4）。 |
| `/home/ruigengji/JAXPBSA/docs/HOU_BOUNDED_SAMPLING_PLAN_20261028.md` §2、§3、§7 | 6 × 10 ns 的预定规则、预算、停止条件与执行记录（含时间容差修复、fulda / 183 运行）。 |
| `/home/ruigengji/JAXPBSA/docs/HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md` §6、§8 与"未确认" | 数值验收的读法与结果；"可分辨"的读法建议还在"未确认"里，没定。 |
| `/home/ruigengji/JAXPBSA/docs/HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md` §2、§8 | 在线成本 / 中断续跑 / 同 pilot 回放的结果；为什么不做 warm-start 恢复。 |

## 1. 现状（2026-10-30）

这条线来自用户 10-28 转贴的一条他人 comment，提出 4 步。4 步和其后的实验排序都已完成，没有在跑的作业。

| 步 | 结果 | 记录 |
|---|---|---|
| 1 副本差异诊断 | a1/a3 是同姿态漂移，a3 rep02、a4 是姿态切换，a6 rep01 在平衡期丢了 Asp189；都与网格无关 | RESULTS §20.6 |
| 2 h0.4 数值验收 | h0.35 − h0.4 为 −0.48…+0.61；h0.3 − h0.35 在 a1/a2 为 −1.4/−1.3，没看到收敛；网格修正带来的分差变化 ≤1.1，比副本误差小；pbsa 加密后"我们 − pbsa"的跨度不变 | RESULTS §20.7 |
| 3 同卡在线成本与恢复 | 3090 上默认网格的在线 overhead 是 3–5%，h0.4 约是它的 2 倍；全部帧都跟上了，没有积压；生产入口中断后续跑无缺无重；同 pilot 回放 174/175 帧差 ≤3.8e-3 | RESULTS §20.8 |
| 4 有界补采样 | 每个样本 6 副本 × 10 ns，42/42 完成；3 个配体盒越界的副本整段离线重评。{a7, a1} 与其余五个分得开，a2–a5 之间分不开 | RESULTS §20.9、§20.9a |
| 实验排序（Codex 算） | a1–a4 ρ = 1.00（6/6），全七条 ρ = 0.75（17/21），精确 p 0.083 / 0.066；采样重采样下全七条 ρ 0.57–0.93 | RESULTS §20.10，`reports/hou_ranking_20261030/` |

Claude 已逐项手算核对 §20.10 的 ρ、τ-b、成对一致数和 a1–a4 的精确 p，与 §20.9 的样本分数一致；重采样范围没有复算。

## 2. 用户原话（逐字，带日期）

（节点状态行里的终端颜色转义码已删掉。）

- 2026-10-28：转贴了一条他人 comment（4 步方案，原文见本对话，要点已写进各计划文档 §0）。
- 2026-10-28："可以"（回答"要我来做第 1 步的诊断表吗？"）；"是的 走起"（回答"要我据此写第 2 步（h0.4 数值验收）的小批计划表吗？"）；"好的"（回答是否改代码并冒烟）。
- 2026-10-28："yayoi36   : [----------------]               ( 0 / 32) free GroupF  AMD EPYC 9354/3.25GHz + RTX 5080, 128GB"；"跑完了"。
- 2026-10-28："是的"（回答"要我接着写第 3 步…的计划表吗？"）；"可以"（回答是否改三处脚本并冒烟）；"yayoi10   : [----------------]               ( 0 / 32) free GroupA  AMD EPYC 7452/2.35GHz + RTX 3090 x 2, 128GB"；"跑完了"。
- 2026-10-28："试试"（回答"要我写第 4 步的计划吗？"）；"可以"（回答是否写 wrapper 和 PBS 并冒烟）。
- 2026-10-29："欸？？？/完了忘记提交了。。稍等 你这样  cyanus@192.168.4.199上面跑"。
- 2026-10-29："│ 改容差、跑相关测试  │ 约 5 分钟  │ —  跑"；"这样 你丢183 跑"；"然后 你脱离窗口 我准备下班"。
- 2026-10-30："是的"（回答"要我按这个办法离线重评这 3 个副本吗？"）。
- 2026-10-30："好了 帮你整理完了"，同时转贴了一段排序统计结果，其中写有"主报告采用冻结默认网格，h0.4 作为数值敏感性对照"。这句出自转贴内容，用户本人没有另外单独说；RESULTS §20.10 和执行交接文首都按这个口径写了。
- 2026-10-30："写handoff"。

## 3. 文件地图

**代码改动**（都在脚本层，没碰 `jaxpbsa/`）：
- `scripts/smallmol_records.py`：`online_rows` 的时间容差改为 0.25 × dt（CHANGELOG 10-29）。
- `scripts/grid_sensitivity_smallmol.py`：加 `h0.3` 配置。
- `scripts/online_overhead.py`：加 `--hou-run-id`、`--grid`、`--cache-dir`，RESULT 加计数字段。
- `scripts/score_smallmol_replica.py`：加 `--pilot npt`。

**两个同名脚本，别弄混**：
- `scripts/summarize_hou_sampling.py`（仓库根，Codex）：严格审计器，只用标准库，带 `--recovery-root`；§20.9a 和 §20.10 的输入就是它出的（`reports/hou_sampling_20261030_{online,verified}/`）。
- `experiments/2026-10-17_small_molecule_benchmark/scripts/summarize_hou_sampling.py`（包内，Claude）：出 §20.9 的表，包括姿态 / Asp189 距离 / 被替换副本的配对差（`reports/hou_sampling_{replicas,samples,pairs,replaced}_20261030.csv`）。
- 排序：`scripts/rank_hou_sampling.py`（Codex）。

**包内其它汇总脚本**（`experiments/2026-10-17_small_molecule_benchmark/scripts/`，都在 183 上只读运行）：`diag_hou_replicas.py`（§20.6）、`summarize_hou_acceptance.py`（§20.7）、`summarize_hou_online_acc.py`（§20.8）。

**作业脚本**（包根目录）：
- `hou_acc.{sh,pbs}`（§20.7）、`hou_online_acc.{sh,pbs}`（§20.8）、`hou_sampling.{sh,pbs}`（§20.9）、`rescore_flagged_w10.sh`（§20.9 的离线重评）。
- `hou_sampling.pbs` 实际没用上，host 仍是占位：采样是在 fulda 上直接跑、收尾在 183 上跑的。

**产物**：
- 分数：`scores/h0.4/`、`scores/replay_npt/`、`scores/window10_default/`、`reference/grid_sensitivity/s25/`、`reference/amber_mmpbsa/q3_pbsa{0.35,0.25}/`。
- 在线验收的小包：`acceptance_online_20261028/`（独立的 `runs.csv`，setup 是软链接）。
- 42 个副本：`md/hou/<s>/ph74_gaff2/rep01–06/`；rep01–03 里有 `status_before_extend_20261028.json`。
- `manifests/runs.csv` 加了 rep04–06 共 21 行；原文件备份为 `runs.before_rep04_20261028.csv`。

**PID 记录**：`pids_183.txt`、`pids_fulda.txt`。所列进程都已退出（10-30 核对）。

**commit 草稿**（都没提交，按时间顺序提交）：`.commit_drafts/2026-10-28_1135_…`、`_1640_…`、`_1710_…`、`_1910_…`、`_1935_…`、`2026-10-30_1900_…`。Codex 的改动（`scripts/summarize_hou_sampling.py`、`scripts/rank_hou_sampling.py`、`tests/test_hou_sampling_summary.py`、`hou_recover_margin_20261030.sh`、§20.9a、§20.10、`reports/hou_sampling_20261030_*`、`reports/hou_ranking_20261030/`）不在这些草稿里。

## 4. 操作要点（本轮实测）

- **Codex 交接 §3 的 `hou_recover_margin_20261030.sh` 不要再跑**：同样的重评已经由 `rescore_flagged_w10.sh` 完成（Codex 交接 §5 也写明了）。
- **NFS 属性缓存 6 h**：fulda 和 183 的 `/home/ruigengji` 挂载都是 `acregmin=21600`。跨机器启动前，先用 md5 比对关键输入；查进度时 ssh 到写文件的那台机器上看。
- **ssh 启动会挂住**：用 `run_in_background` 跑 `setsid nohup … & echo $!`，拿到 PID 后停掉本地的 ssh。被 setsid 的进程不受影响。
- **fulda（cyanus@192.168.4.199）**：1 × RTX 5090，JAX 能用 GPU。Hou 生产 MD 加在线评分是 150 s/ns，是 3090（298 s/ns）的一半。
- **在线 MPS 的编译缓存**：MPS 30% 下的 worker 不命中不限额进程写的缓存，第一次会再冷启动一次（§20.8）；pilot 帧数变了也不命中。
- **不要再运行包内首轮的 `summarize_hou_online.py`**：它会把 `runs.csv` 里的 rep04–06 一起读进来（按 5 ns 截取），然后覆盖首轮的 `reports/hou_online_{replicas,samples}.csv` 和 `hou_online_ranking.json`。

## 5. 还没定 / 还没做

由用户定：
1. "可分辨"用什么读法。现在只列分差和合并标准误的比值；2 倍的建议在数值验收计划的"未确认"一节，没有写进正文。
2. BRD4 的报告网格和实验值统计。RESULTS §20.10 只覆盖 Hou；BRD4 仍是 §19.2 的描述性 ρ ≈ 0.43。

没查清的现象（都已记在 RESULTS，不影响现有结论）：
- a4 rep01 第 140 帧，同 pilot 回放的 G_R 差 0.10（§20.8）。
- a5 rep05 在 5.5 ns 处的 PBSA 积压（§20.9；之前最大延迟约 0.13 s，帧间隔约 1.8 s）。
- h0.3 − h0.35 比 h0.35 − h0.4 大且反号（§20.7）。
- 已报告、没修：MD 进程被杀后，`jaxpbsa/_online_worker.py:54` 会打印 `ConnectionResetError`（工程交接 §7.1）。

## 未确认（Claude 推断）

- a6 的两种状态（和 Asp189 形成 / 不形成相互作用）、a3/a4 的多姿态，可能需要不同起点或更长采样才能估出布居；设计稿 §7 不允许把从同一个起点跑出来的状态比例当作布居。预算已经用完，没有新计划。
- h0.3 − h0.35 不收敛，可能是因为配体盒步长（0.167 → 0.15）和 C/R 步长同时变；没有单独检查。
- 被标记帧的根源是配体盒按 1 ns NPT pilot 定尺，10 ns 内配体伸展得超出了这个范围。如果以后跑更长的在线 MD，可能需要更大的配体盒余量；这一条没有验证过，也没改任何默认值。
