# 小分子基准：PB 网格诊断与 h0.4 重评交接（2026-10-26，Claude）

**2026-10-30 更新：** 最新入口是 [HOU_SAMPLING_RECOVERY_HANDOFF_20261030.md](HOU_SAMPLING_RECOVERY_HANDOFF_20261030.md) §5。副本诊断、数值/成本验收、6 × 10 ns 补采样和三个越界副本的完整离线恢复均已完成；严格复核确认 42,000 帧完整可用。报告沿用冻结默认网格，h0.4 保留为敏感性；实验排序已按设计 §9 完成，同系列 a1–a4 ρ=1.00，全七条 ρ=0.75，分别报告采样和候选组成不确定性。不重复提交已完成的运行，详见 RESULTS §20.9–20.10。

## 必读（开工前全部读完）

| 路径 | 为什么要读 |
|---|---|
| `/home/ruigengji/JAXPBSA/CLAUDE.md` | 冻结基线（main 3737870，不改方法和默认参数）、生产环境 `openmm_dev`、机器分工、PBS 选卡写法。 |
| `/home/ruigengji/JAXPBSA/docs/BRD4_HOU_EXECUTION_HANDOFF.md`（文首 10-25、10-26 两行，§7.11） | 当前优先执行入口；(a) 加密重评 / (b) 先诊断，这个待定事项的来历。 |
| `/home/ruigengji/JAXPBSA/docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md` §8、§9、§10 | 数值敏感性设计、评分统计口径；"数值变化足以改变次序就先做数值诊断"。 |
| `/home/ruigengji/JAXPBSA/RESULTS.md` §19.4–§19.8、§20.2–§20.5 | 本轮全部实测结论，结论以这里为准。 |
| `/home/ruigengji/JAXPBSA/docs/BRD4_GRID_DIAGNOSIS_PLAN_20261025.md` §3、§5、§8 | D0/D1/D3 的设计、结果怎么对应选项，以及执行记录。 |
| `/home/ruigengji/JAXPBSA/docs/BRD4_H04_RESCORE_PLAN_20261025.md` §2、§6 | `--grid h0.4` 的代码改动和 BRD4 重评记录。 |
| `/home/ruigengji/JAXPBSA/docs/ENGINEERING_REPAIR_HANDOFF_20261024.md` §5、§7 | 剩余的数值和采样工作清单；评分续跑修复（不按 DCD 的 SHA256 绑定）。 |

## 1. 现状（2026-10-26 07:10）

已完成，全部有实测：
- **工程修复**：评分和网格敏感性脚本的续跑不再用 DCD 的 SHA256 绑定（strip 会重写文件头里的时间）。183 上冒烟通过，见工程交接 §7。
- **BRD4 默认网格偏差诊断**：
  - D0：偏差全在 C−R，9 帧全部同号；转动平均消不掉，所以不是相位冻结（RESULTS §19.6）。
  - D1：亚格点平移同样消不掉。最大的一项是配体电荷在 C/R 0.5 网格上的自项 LL_C（BRD4-3/6/8 为 +2.1～2.6）；受体项随样本变号（RESULTS §19.7）。
- **BRD4 15 000 帧 h0.4 重评**：次序和默认网格完全相同（Spearman 1.000）；修正量 −0.35…+2.30，BRD4-3 和 BRD4-6 最大（RESULTS §19.8）。
- **Hou 同帧对照（D3）**：
  - 63 帧，E_coul / E_LJ / SA 和 Amber 对齐（差 ≤0.06）。
  - 逐帧的网格效应约是 BRD4 的 2 倍。
  - 加密后，"我们 − pbsa"在样本之间的跨度没有缩小（6.5 → 5.9），和 BRD4 不同（RESULTS §20.4）。
- **Hou 21 × 500 帧 h0.4 重评**：
  - 修正量 −1.64…+3.48，a6 最大（+3.48，配体是中性的）。
  - 次序 Spearman 0.893，a6 降到最末。
  - 次序有变动的 a2–a6 这一段，本来就被副本之间的差异掩盖（RESULTS §20.5）。
  - 9 帧估计在 Hou 上和 500 帧最多差 1.9，不能只靠 9 帧下判断。

没有做的：
- 实验值统计。BRD4 和 Hou 用 h0.4 时的实验相关性都没有算。
- 报告用哪套网格（默认还是 h0.4），没有定。

## 2. 用户原话与决定（逐字，带日期）

（节点状态行里的终端颜色转义码 `^[34m`、`^[0m` 已删掉，其余逐字照录。）

- 2026-10-25（这是回答"要我写这个诊断的计划吗？"）："可以"。
- 2026-10-25："你跑完了之后脱离窗口 。。。nfs 更新需要手动"。
- 2026-10-25："可以做 D0 D1 都做了"。
- 2026-10-25（回答"要我改成直接在 183 上跑 D1 正式那一轮吗？"）："跑起来"。
- 2026-10-25（回答"要我按选项 ① 写 h0.4 重评全部 15 000 帧的计划表吗？"）："可以"。
- 2026-10-25："yayoi04   : [----------------]               ( 0 / 32) free GroupA  AMD EPYC 7452/2.35GHz + RTX 3090 x 2, 128GB 到时候跑这个   好了你先冒烟去"。
- 2026-10-25："跑完了"（BRD4 h0.4 重评）。
- 2026-10-26："183正忙 181 也在忙。。。"；"可以"（在 180 上跑 D3 冒烟）。
- 2026-10-26："yayoi10   : [----------------]               ( 0 / 32) free GroupA  AMD EPYC 7452/2.35GHz + RTX 3090 x 2, 128GB 整个节点丢给你"。
- 2026-10-26："好了"（D3 作业跑完）；"可"（写 Hou h0.4 的 PBS 脚本）；"跑完了"（Hou h0.4 重评）。
- 2026-10-26："写从的脚本然后写handoff"。Claude 理解为"写 commit 草稿，然后写交接"，没有和用户确认过。

**仍由用户定**：(a)/(b) 的最终选择，也就是 BRD4 和 Hou 报告用默认网格还是 h0.4；以及是否做实验值统计。Claude 没有改任何默认参数。

## 3. 文件地图

**代码**：
- `scripts/score_smallmol_replica.py`：`--grid {default,h0.4}`。
- `scripts/grid_diag_smallmol.py`：D1。
- `scripts/amber_reference_smallmol.py`、`scripts/grid_sensitivity_smallmol.py`：`--ours online`，供 Hou 用。
- `scripts/smallmol_records.py`：`online_rows_by_frame`。

**包内汇总脚本**（`experiments/2026-10-17_small_molecule_benchmark/scripts/`，都在 183 上只读运行）：
- `grid_diag_d0.py`、`grid_diag_d1.py`；
- `summarize_brd4_round1.py [h0.4]`；
- `summarize_reference_q3.py [hou]`；
- `compare_brd4_grids.py`、`compare_hou_grids.py`。

**作业脚本**（包根目录）：
- `brd4_h04.{sh,pbs}`、`hou_ref.{sh,pbs}`、`hou_h04.{sh,pbs}`；
- `grid_diag.pbs`：D1 实际是在 183 上直接跑的，这个文件只留作记录。

**产物**：
- 原始结果：`scores/h0.4/{brd4,hou}/`、`reference/grid_diag/q3/brd4/`、`reference/{amber_mmpbsa,grid_sensitivity}/q3/hou/`。
- 汇总表：`reports/grid_diag_d{0,1}_*20261025.csv`、`reports/brd4_h0.4_{replicas,samples}.csv`、`reports/brd4_h04_vs_default_20261025.csv`、`reports/reference_q3_hou_{frames,samples}.csv`、`reports/hou_h04_vs_default_20261026.csv`。

**冒烟目录**：`smoke_repair_20261025/`、`smoke_grid_diag_20261025/`、`smoke_h04_20261025/`、`smoke_d3_20261026/`。

**commit 草稿**（都还没提交）：`.commit_drafts/2026-10-25_0935_…`、`2026-10-25_2344_…`、`2026-10-26_0710_…`。三份 add 的文件有重叠，建议用最后一份里的"合并提交"命令一次提交。

## 4. 操作要点（本轮实测）

- **181 的 NFS 视图有滞后**：一次汇总在 181 上报错，换到 183 上读到的就是完整文件。日志和汇总一律在 183 上通过 ssh 运行，用 `/home/ruigengji/miniforge3/envs/openmm_dev/bin/python`。
- **ssh 启动会卡住**：按全局模板用 `ssh … 'bash -c "… setsid nohup … & echo $!"'` 启动时，外层 `bash -c` 不退出，本地的 ssh 会一直挂着。做法是把这次 ssh 放到后台运行，进程起来后再用 ssh 查出 python 的 PID，追加到 `pids_*.txt`，然后停掉本地那个 ssh。被 setsid 的进程不受影响。
- **180 的 JAX 不能用 GPU**：180 的 `openmm_dev`（`/home/canna/miniforge3`）里 jaxlib 不带 CUDA，JAX 会静默退回 CPU。180 只能验证代码路径和跑 AmberTools，GPU 耗时和显存测不出来。
- **yayoi 节点**：PBS 不设 `CUDA_VISIBLE_DEVICES`，作业脚本自己挑没有计算进程的卡（写法见各个 `.pbs`）。
- **实测耗时**（3090）：
  - BRD4 h0.4 每帧 239 ms，每个副本初始化约 2 min。
  - Hou h0.4 每帧 382 ms（来自 q3 的网格敏感性测量）；21 副本 × 500 帧在 2 卡上实际用了 40 min。
  - Hou 同帧对照：Amber 3 min；网格每卡约 1.2 h。

## 5. 下一步候选（来自工程交接 §5 和本轮结果，Claude 整理，由用户选）

1. 报告网格的决定。默认和 h0.4 的两套 500 帧分数都已经齐了，见 RESULTS §19.8 和 §20.5。
2. Hou a1/a6/a3 的逐副本分项，以及 a3/a4 的姿态。只读分析已有 CSV 和轨迹诊断，不占 GPU。a1 的副本间极差在 h0.4 下是 18.9，次序由采样主导。**（2026-10-28 已做，见 RESULTS §20.6。）**第 2 步（h0.4 有界数值验收）的计划：[HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md](HOU_H04_NUMERICAL_ACCEPTANCE_PLAN_20261028.md)，2026-10-28 已在 yayoi36 上跑完，见 RESULTS §20.7。第 4 步（有界补采样）：[HOU_BOUNDED_SAMPLING_PLAN_20261028.md](HOU_BOUNDED_SAMPLING_PLAN_20261028.md)，2026-10-30 完成，见 RESULTS §20.9。
3. 同一张卡上 MD 开销的基准：baseline、同步、异步、MPS 四种对比（工程交接 §5 第 4 条），要占 GPU。计划：[HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md](HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md)（2026-10-28 已在 yayoi10 上跑完，见 RESULTS §20.8）。
4. 工程上还没验证的两件事：真实求解器 warm start 失败后的恢复；生产副本在节点中断后的续跑。
5. 已报告、没修的小问题：MD 进程被杀后，在线 worker 会打印 `ConnectionResetError` 和 `BrokenPipeError`。`jaxpbsa/_online_worker.py:54` 只捕获了 `EOFError` 和 `BrokenPipeError`。不影响结果。

## 未确认（Claude 推断）

- LL_C 在 h=0.5 下偏大的物理原因：可能是配体电荷离受体的介电界面很近，也可能是配体内部偶极在粗网格上欠解析。没有单独算过，例如没有算 G_L 在 h=0.5 与 0.25 下的差。
- Hou 上"我们 − pbsa"的样本间跨度加密后不缩小：可能来自 pbsa 自身的离散误差（pbsa 用的是 h 0.5，没有加密），也可能是表面定义不同。这批数据分不开。（2026-10-28 已测：pbsa 加密到 0.25 后，跨度仍是 6.0，所以不是 pbsa 的离散误差，见 RESULTS §20.7。）
- Hou 的网格修正比 BRD4 大，可能和配体带电（+2、+1）有关；但修正最大的 a6 配体是中性的，所以这个解释不完整。
