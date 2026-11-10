# BRD4 全部 15 000 帧按 h0.4 重评：计划（2026-10-25，Claude 起草；已完成，结果见 RESULTS §19.8）

## 必读（开工前全部读完）

| 路径 | 为什么要读 |
|---|---|
| `/home/ruigengji/JAXPBSA/CLAUDE.md` | 冻结基线（不改默认参数）、生产环境、PBS 的 GPU 选择写法。 |
| `/home/ruigengji/JAXPBSA/docs/BRD4_GRID_DIAGNOSIS_PLAN_20261025.md` §5、§8 | 为什么是 h0.4：诊断结果和对应的选项 ①。 |
| `/home/ruigengji/JAXPBSA/RESULTS.md` §19.4、§19.6、§19.7 | h0.4 下"我们 − pbsa"的样本间跨度 0.57（默认 4.1）；偏差来自 C/R 网格上的离散误差。 |
| `/home/ruigengji/JAXPBSA/docs/BRD4_HOU_EXECUTION_HANDOFF.md` §4.6–4.7、§7.7–7.9 | 首轮评分的窗口、分块和产物位置，以及各卡上的实测耗时。 |
| `/home/ruigengji/JAXPBSA/scripts/score_smallmol_replica.py` | 要改的评分脚本，以及它的续跑与 protocol 绑定。 |

## 0. 范围

- 用户 2026-10-25 的原话："可以"。这是回答我的问题"要我按选项 ① 写 h0.4 重评全部 15 000 帧的计划表吗？"。范围只是写计划，不改代码，也不跑。
- 冻结的方法和默认参数（h 0.5 / 0.25）不改。h0.4 作为**另一套评分**，输出到单独的目录，和首轮结果并列。
- 这个计划只做重评。(a)/(b) 里"是否把 h0.4 作为 BRD4 的报告网格"仍由用户决定。

## 1. 算什么、和什么比

- 量：每帧 ΔG_MMPBSA（以及其中的 ΔG_PB），配置 C/R h 0.4、配体紧盒 h 0.2，其余全部是冻结默认（`PBParams(precond="auto")`、mbondi2、INP=1 SA）。和 `grid_sensitivity_smallmol.py` 的 `h0.4` 配置完全相同。
- 窗口：和首轮相同，每副本 (0, 5] ns，500 帧，1 ns 分块，共 30 副本 × 500 = 15 000 帧。
- 对照：
  1. 首轮默认网格的同帧结果（`scores/default/...`）；
  2. q3 的 90 帧应当复现已有的 h0.4 参照值（`reference/grid_sensitivity/q3/...`，config h0.4）。两者是同一帧、同一配置、同一个 pilot 窗口，这是这批结果自带的核对；§19.4 里新建 analyzer 重算的差异 ≤0.001。
- 不设通过/不通过阈值，结果直接报。

## 2. 要改的代码（脚本层，不碰 `jaxpbsa/`）

1. `scripts/score_smallmol_replica.py`：加 `--grid {default,h0.4}`，默认 `default`，行为和现在完全一样。`h0.4` 时给 `OnlineMMPBSA` 传 `h=0.4, h_lig=0.2`。protocol 的 `analyzer` 字段记录实际的 h / h_lig（现在是写死的 0.5 / 0.25）。默认输出目录改为 `scores/<grid>/...`：`default` 时路径不变，已有结果照常续跑。
2. 包内 `scripts/summarize_brd4_round1.py`：加一个参数，指定读 `scores/<grid>`，输出 `reports/brd4_<grid>_{replicas,samples}.csv`。默认行为不变。
3. 新的批处理脚本 `brd4_h04.sh` / `brd4_h04.pbs`：照抄 `brd4_rescore.sh` / `.pbs` 的结构。每块 GPU 一个 worker，30 个副本轮流分给各 worker，失败的副本记 FAIL 后继续下一个，重新执行同一条命令可以续跑。

## 3. 冒烟（改完代码后一次，183）

- 机器与输入：183 的 2080 Ti，BRD4-3 rep01，`--grid h0.4 --start-ns 1.2 --end-ns 1.3`，共 10 帧，包含 q3 帧 124（t = 1250 ps）。输出写到 `smoke_h04_20261025/`，不进 `scores/h0.4`。
- 测量：预热时间和每帧耗时。
- 核对：帧 124 和已有的 h0.4 参照值对比。冒烟的 pilot 只有 10 帧，网格盒子和参照不同，D0 测得这类差异 ≤0.001。
- 正式运行时 pilot 是整个 (0, 5] ns 窗口，和参照的 pilot 相同，所以 q3 帧应当逐位复现参照值。

## 4. 计划表（正式运行）

| 项 | 内容 |
|---|---|
| 命令 | `qsub brd4_h04.pbs`（内容：每块空闲 GPU 跑一个 `W=<k> NW=<n> GPU=<id> bash brd4_h04.sh`，脚本对每个副本执行 `score_smallmol_replica.py --run-id <rid> --grid h0.4 --start-ns 0 --end-ns 5 --block-ns 1`） |
| 节点 | PBS，用户指定（项目规则）；也可以在 183 上跑，见下面的耗时 |
| 资源 | 每个 worker 1 GPU、8 线程；显存：h0.4 网格最大 225³，h0.35（更大）在 183 实测约 8.4 GB，11 GB 卡够用 |
| 预计耗时 | **3090**：实测（yayoi30，30 个副本的 q3 h0.4，每次调用都含 PB+SA+MM，与评分同一个调用）每帧中位 241 ms（192–284）。15 000 × 0.241 s ≈ 60 min，加上 30 次网格建立和预热，中位 58 s（7.6–300 s，有编译缓存的快）≈ 30 min。单卡约 1.5 h，2 卡约 45 min。**2080 Ti（183）**：实测每帧 458 ms、冷启动预热 248 s，15 000 × 0.458 s ≈ 1.9 h，加 30 × ≤248 s ≤ 2.1 h，约 2–4 h。DCD 读取和 CSV 写入的耗时没有单独测，冒烟后修正。 |
| 续跑 | 可以：`frames.csv` 里已有的帧会跳过（现有机制） |
| 输出 | `scores/h0.4/brd4/<sample>/mobley_amber/rep<NN>/{frames.csv,summary.json,…}`；汇总写 `reports/brd4_h0.4_{replicas,samples}.csv` |
| 这批结果决定什么 | BRD4 在 h0.4 下的样本分数和次序。这是"默认网格的数值误差是否改变结论"的直接回答，之后做实验比较时用哪套分数由用户定 |

## 5. 不做的事

- 不改 `jaxpbsa/` 和默认参数，不改首轮 `scores/default` 的结果。
- 不扩到 5 ns 以外，不换窗口或分块。
- 不做实验值统计（另行决定）。
- 不同时做 h0.35 或别的配置。

## 6. 执行记录

- 用户 2026-10-25 原话："yayoi04 … RTX 3090 x 2, 128GB 到时候跑这个   好了你先冒烟去"。
- 代码已按 §2 改完：
  - `scripts/score_smallmol_replica.py` 加 `--grid {default,h0.4}`。`default` 的 analyzer 记录和输出路径都不变；已有的 `scores/default` 目录本来就没有 protocol.json，不受影响。
  - 包内 `scripts/summarize_brd4_round1.py` 可选参数 `<grid>`。
  - 新增 `brd4_h04.sh` 和 `brd4_h04.pbs`（yayoi04，2×3090，每卡一个 worker，16 线程，walltime 4 h）。
- 冒烟（183，2080 Ti，BRD4-3 rep01，1.2–1.3 ns 共 10 帧，`smoke_h04_20261025/`）：
  - 10/10 帧有效，0 帧被标记。
  - 帧 124 和已有的 h0.4 参照对比：ΔG_MMPBSA 差 −1.2e-4，ΔG_PB 差 −1.2e-4。G_C、G_R 各差 −0.029，在 C−R 里抵消，这是 pilot 不同导致网格盒子不同（193×161×193，参照是 500 帧 pilot）。
  - 耗时：每帧 261 ms（这个较小的网格），冷启动预热 232 s。
- 正式运行：`cd experiments/2026-10-17_small_molecule_benchmark && qsub brd4_h04.pbs`。日志 `brd4_h04_w{0,1}.log`。跑完后在 183 上执行 `scripts/summarize_brd4_round1.py h0.4`。

### 6.1 正式运行（完成）

- 用户 2026-10-25 原话："跑完了"。yayoi04，14:45–15:44。两个 worker 各 15 副本，0 FAIL，15 000 帧全部有效。
- q3 的 90 帧和 h0.4 参照逐位相同。
- 样本次序和默认网格完全相同（Spearman 1.000）。修正量 −0.35…+2.30（BRD4-3 +2.30、BRD4-6 +2.06），详见 `RESULTS.md` §19.8。
- 耗时比计划多约 15 min：每帧 239 ms 和预计一致，多出来的是每副本的初始化（中位 121 s，计划按 58 s 算）。

## 未确认（Claude 推断）

- 预计耗时假定 3090 上评分的每帧耗时和网格敏感性的每次调用相同。评分脚本多了 DCD 读取和 CSV 写入，冒烟后修正。
- 我推测 h0.4 能把 BRD4 的 C/R 离散误差降到样本间跨度 ≤0.6。这个推测只有 q3 的 90 帧做依据（§19.4），500 帧上没有验证过。
