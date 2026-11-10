# P1/P2 会话说明：基线对照、BRD4 统计、BRD4-5 重跑及附带产物（2026-11-01，Claude）

**2026-11-06 更新：** 本地收尾已完成，当前入口 [PAPER_CLOSEOUT_20261106.md](PAPER_CLOSEOUT_20261106.md)。本文保留 11 月 1–2 日会话记录；下面的未完成项是该日期快照。

本文说明 2026-11-01 这一轮做了哪些实验（包括计划外补做的）、产生了哪些文件（包括冒烟目录、失败日志等杂项）、哪些是正式结果、哪些不能用。结论数字以 RESULTS §21 为准。

## 必读（开工前全部读完）

| 文件 | 为什么要读 |
|---|---|
| `CLAUDE.md`（项目根） | 冻结方法、生产环境 `openmm_dev`、机器分工与 PBS 选卡写法 |
| `docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md` | 本轮按它推进 P1、P2；§8 勾选状态 |
| `RESULTS.md` §21.1–§21.4 | 本轮全部实测结论 |
| `docs/BRD4_5_REPREP_PLAN_20261101.md` | BRD4-5 输入错误的修法、冒烟与正式运行记录 |
| 包内 `reports/baseline_20261101_v2/report.md`、`reports/brd4_statistics_20261101_v2/report.md` | P1、P2 的现行版本（v1 保留作对照） |

## 1. 现状

- P1.0、P1.1、P2 已完成并勾选（TODO、计划 §8）。下一步按计划是 P3（证据/成本表、论文图表初稿、复现包），未开始。
- 没有在跑的进程：183 上本轮启动的 PID（见 §5）2026-11-01 核对均已退出；PBS 作业 21818、22051 已结束。
- 本轮新代码没有写 commit 草稿（`.commit_drafts/` 最新一份是 10-30）。

## 2. 用户原话（2026-11-01，逐字）

- "好了 我更新了 一下接下来的 计划你读一下 …… 顺序固定为：基线对照 → BRD4 分组统计 → 复现包与论文收口 → 按需扩展新体系。"
- "可以"（依次回答：开始 P1.0；写 B1 入口并在 183 冒烟；下载 BRD4 配体文件并核对；写 protocol 并出 B0/B1/B2 对照；开始 P2；写 BRD4-5 重跑方案；写准备脚本并冒烟）。
- "yayoi43   : [----------------]               ( 0 / 32) free GroupE  AMD Ryzen 9 5950X + RTX 4080 super, 128GB"；"跑完了"。
- "yayoi28   : [----------------]               ( 0 / 32) free GroupD  AMD Ryzen 5950X/3.40GHz + RTX 3090, 64GB 慢慢跑"；"跑完了吧"。
- "是的"（回答是否把新 BRD4-5 接回 P1/P2 重算）。
- "然后还要补说明"；"补实验等乱七八糟"（本文）。

（节点状态行里的终端颜色转义码已删。）

## 3. 实验清单

### 3.1 计划内

| 实验 | 机器 / 作业 | 产物 | 结论入口 |
|---|---|---|---|
| P1.0 输入与缺口清单（只读） | 181 | `reports/baseline_20261101/inventory.csv` | 计划 §8 |
| B1：17 个最小化起点单帧评分 | yayoi43 RTX 4080 SUPER，作业 21818，19 min | `scores/start_default/`、`reports/baseline_20261101/start_scores.csv` | RESULTS §21.1 |
| P1.1 B0/B1/B2 对照（只读） | 181 | `reports/baseline_20261101/` | RESULTS §21.2 |
| P2 BRD4 标签审计与分组统计（只读） | 181 | `reports/brd4_statistics_20261101/` | RESULTS §21.3 |

### 3.2 计划外补做的（都因为前一步发现了缺口）

| 补做 | 起因 | 产物 | 结论入口 |
|---|---|---|---|
| 下载 BRD4 上游配体 sdf/mol2（commit 5233e0a）并与 prmtop 逐条核对 | 可旋转键需要键级，prmtop 没有 | `sources/brd4/{sdf,mol2}/`、`sources/retrieval_brd4_ligands.json`、`reports/baseline_20261101/brd4_ligand_file_check.csv` | 执行交接 §7.4 补注 |
| 读 6 篇 BRD4 标签原文（PMC/Europe PMC）；10 条配体 InChIKey 对 PubChem/CCD | P2 第 1–2 条要求核对来源与测定条件 | `sources/brd4/papers/`（`retrieval.txt` 记录来源）、`reports/brd4_statistics_20261101/label_audit.csv` | RESULTS §21.3 |
| **BRD4-5 改用阿普唑仑重新准备并重跑 3 × 5 ns** | 审计发现上游 BRD4-5 是 5,6-二氢阿普唑仑（C17H15ClN4），不是实验化合物 | 独立包 `brd4_5_alprazolam_20261101/`（yayoi28 作业 22051） | RESULTS §21.4，方案文档 |
| P1、P2 v2 重算（BRD4-5 换源） | 上一行 | `reports/baseline_20261101_v2/`、`reports/brd4_statistics_20261101_v2/` | RESULTS §21.4 |

### 3.3 只作对照、不能当结果用的

- 冒烟输出：`scores/smoke/start_default/`（B1 冒烟，BRD4-3、a1）、`brd4_5_alprazolam_20261101_smoke/`（BRD4-5 准备 + setup + B1 冒烟，183 2080 Ti）。冒烟的 BRD4-5 B1 = −24.11，正式（yayoi28）= −21.08，见 §6。
- v1 报告 `reports/baseline_20261101/`、`reports/brd4_statistics_20261101/`：其中 BRD4 部分含（或排除）错误的 BRD4-5，已被 v2 取代；保留不动，以便对照与复现。
- 旧的 `md/brd4/BRD4-5/mobley_amber/` 和 `scores/{default,h0.4,start_default}/brd4/BRD4-5/`：二氢形式，保留，不再用于实验统计。

## 4. 文件地图

路径未加前缀的都在包 `experiments/2026-10-17_small_molecule_benchmark/` 下。

**代码（新增/修改）**
- `scripts/score_smallmol_start.py`（仓库根，新增）：B1 单结构评分；不传 pilot、`fluctuation_allowance=0`；后加 `--model-id`、`--inputs`（默认不变）。
- `scripts/build_baseline_inventory.py`：P1.0 清单（只读）。
- `scripts/collect_start_scores.py`：汇总 B1 → `start_scores.csv`。
- `scripts/check_brd4_ligand_files.py`：BRD4 配体文件对 prmtop 核对。
- `scripts/compare_baselines.py`：P1 对照；`--out` + 协议 `sample_sources`。
- `scripts/brd4_statistics.py`：P2 统计与作图；`--out` + `sample_sources`。
- `scripts/prepare_brd4_5_alprazolam.py`：BRD4-5 只换配体的准备脚本。
- 两个分析脚本加选项后，对 v1 输入逐位复现 v1 产物（在 scratchpad 核对）。

**作业脚本**（包根）
- `start_smoke.sh`、`start_full.sh/.pbs`：B1 冒烟与全量。
- `brd4_5_smoke.sh`、`brd4_5_full.sh/.pbs`：BRD4-5 冒烟与正式（PBS host 已填 yayoi28）。

**日志**（包根）
- `start_smoke_try1_pilot1frame.log`：B1 第一次冒烟失败（冻结代码拒收 1 帧 pilot），改用 `fluctuation_allowance=0` 后 `start_smoke.log` 通过。
- `start_full.log`、`start_b1.o21818`：B1 全量。
- `brd4_5_smoke_try1_parmed_slice.log`：BRD4-5 第一次冒烟失败（ParmEd 复制切片后的 AmberParm 报 LJ 表长度错），改为先转 Structure 后 `brd4_5_smoke.log` 通过。
- `brd4_5_fix.o22051`、`brd4_5_alprazolam_20261101/{setup,w0}.log`：BRD4-5 正式运行。

**PID**：`pids_183.txt` 末尾 7 行是本轮（178702、179827、179828、238424、238425、238549、238550），都已退出。

**文档**：新增 `docs/BRD4_5_REPREP_PLAN_20261101.md` 与本文；更新 `TODO.md`、`docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md` §8、`docs/BRD4_HOU_EXECUTION_HANDOFF.md`（文首 BRD4-5 说明、§7.4 补注）、`RESULTS.md` §21.1–§21.4（§21.2 补注 BRD4-5）、`CHANGELOG.md` 2026-11-01 各节。

## 5. 操作要点（本轮实测）

- **不要改主包 `manifests/runs.csv`**：Hou 10-30 的 `reports/hou_sampling_20261030_verified/audit.json` 绑定了它的 sha256，改了 `scripts/rank_hou_sampling.py` 会以"审计过期"拒绝复现。新增体系用独立包（先例 `acceptance_online_20261028/`、`brd4_5_alprazolam_20261101/`），入口都支持 `--package/--inputs`。
- 183 上 JAX 会打印 cuda plugin 0.11.1 与 jaxlib 0.10.2 不兼容的警告，但实际在 `cuda:0` 上跑（summary.json 的 device 字段），可忽略。
- ssh 启动 `setsid nohup … & echo $!` 有时挂住；用 `timeout 20 ssh …` 启动，再用 `ps` 按进程名取 PID 记录。
- BRD4 两套网格的旧逐帧分数在 `scores/{default,h0.4}/brd4/`，default 那批没有 `protocol.json`（早于协议绑定）。

## 6. 还没做 / 待定（2026-11-01 晚处理后的状态）

- P3：W1 证据清单与统计复现、W3 成本表已完成（RESULTS §21.7，`reports/evidence_20261102/`）；软件能力对照、W4 图表与初稿、W5 复现包未做。
- commit 草稿：已写 `.commit_drafts/2026-11-01_1820_p1_p2_baselines_brd4_5.md`（未提交，由用户执行）。
- BRD4 实验值原文核对：已完成（2026-11-02，用户提供的 13 个论文/SI 文件在 `/home/ruigengji/JAXPBSA/` 根目录，未移动）。BRD4-3、5–10 核到原表；BRD4-4 的 −6.36 在所引论文中找不到（RESULTS §21.6，`reports/brd4_statistics_20261101_v2/label_verification_20261102.csv`）。注意：181 的 NFS 目录缓存 6 h，新文件 `ls` 看不到时按文件名直接访问或在 183 上列目录。
- BRD4-8 偏弱：只读分项分解 `reports/brd4_statistics_20261101_v2/brd4_8_components.csv`（包内 `scripts/diag_brd4_8.py`）。默认网格下 BRD4-6 比 BRD4-8 低 3.49，差值几乎全在 elec（E_coul + ΔG_PB：12.83 vs 17.39，差 4.6）；E_LJ 反而 BRD4-8 略好（−32.2 vs −31.1），SA 相同；BRD4-6 多出的羟乙氧基给出 E_coul −19.1 vs −4.3，大半被去溶剂化抵消。BRD4-8 姿态稳定（配体 RMSD p50 1.1–1.4 Å）。这是分项描述，不是原因。
- B1 对最小化的敏感性：已完成（用户给 yayoi37，PBS 22221，`b1_repeat.pbs` → `b1_repeat.sh`，独立包 `b1_minimisation_repeat_20261101/`）。|差| 中位 0.81、最大 5.00（Hou a7）；BRD4 次序不变，Hou 的 B2 增益结论在两次最小化下都成立（RESULTS §21.5）。

## 未确认（Claude 推断）

- 上游 BRD4-5 多出的 2 个 H 可能是 openbabel 加氢时丢了 C6=N5 双键；没有核实上游生成过程。
- B1 在两台机器上差 3 kcal/mol，可能来自 GPU 上最小化路径不同（混合精度、不同卡）落到了不同的局部极小；没有做能区分的检查。
