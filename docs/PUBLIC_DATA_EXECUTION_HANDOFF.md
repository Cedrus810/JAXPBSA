# 公开实验数据复算：执行交接（首轮 MDM2 / PMI 八样本）

日期：2026-10-11 建立，2026-10-17 整理（JST）。

> **2026-11-06 接续入口：** 用户重新提出多肽结构问题，现已整理 [PPI结构诊断与接续计划](PPI_STRUCTURE_RECOVERY_PLAN_20261106.md)。当前下一项为 PMI、PMI-L10A、P12Q_L10A 三个代表、六模型的逐阶段问题定位；本轮只完成记录整理和计划，未进行新几何验收或分子计算。2026-11-07 R0/R1 已完成（16 个模型）：未定位到 v2.1 硬错误，偏离在 B1–B3、B11–B12 和 A25，见 `experiments/2026-11-06_ppi_structure_review/r0r1/structure_diagnosis.md`。下文 P4 停点与 ready 状态保留为原执行快照；后续是否复用或重跑由具体结构证据决定。

本文件是当前的接手入口，按主题写现状。2026-10-11 至 10-17 的逐条执行日志（v1 冒烟、P2 诊断、PDBFixer 问题的排查过程等）保存在 `docs/archive/PUBLIC_DATA_EXECUTION_HANDOFF_full_log_20261017.md`。

> 2026-10-17 后续决定：本文件继续作为**多肽线**证据和交接入口。用户要求先写较简单的蛋白–小分子基准设计，计算留给后续执行；新路线见 [SMALL_MOLECULE_BENCHMARK_DESIGN.md](/home/ruigengji/JAXPBSA/docs/SMALL_MOLECULE_BENCHMARK_DESIGN.md)。旧数据与未完成的数值对照保留，不因此判定采样或 PBSA 模型为已确认根因，也不自动扩大多肽运行。

## 必读（开工前全部读完）

| 绝对路径 | 为什么要读 |
|---|---|
| 全局机器规则（NFS、环境、资源和作业启动；机器本地维护，不入库） | 避免在错误节点跑 prep/MD/MODELLER。 |
| 项目冻结基线与生产环境说明（项目根维护，不入库） | 哪台机器能跑什么。 |
| `/home/ruigengji/JAXPBSA/docs/JCIM_PEPTIDE_PAPER_OUTLINE.md` | 论文要回答的问题，以及统计、基线和成本的口径。 |
| `/home/ruigengji/JAXPBSA/docs/PUBLIC_DATA_STRUCTURE_PLAN.md` | 完整路线 P0–P5，以及评分、对照、统计的原则（§9、§10.3）。 |
| `/home/ruigengji/JAXPBSA/docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md` | **现行准备协议 v2.1**，以及 MD、评分、P4 方案和问题 0 的全部结果。 |
| `/home/ruigengji/JAXPBSA/experiments/2026-10-11_public_data_execution_packet/protocol.md` | 输入化学的决定（§1）、取帧和统计单位（§2.1），在看实验排名之前固定。 |
| `/home/ruigengji/JAXPBSA/experiments/2026-10-11_public_data_execution_packet/run_commands.md` | 实际用过的作业脚本和命令。 |
| `/home/ruigengji/JAXPBSA/docs/PUBLIC_DATA_RECORD_AUDIT_20261011.md` | 其他系列（Bcl-xL、MDMX）的标签问题；以后扩展时避免沿用错误的库标签。 |

## 1. 当前状态（2026-10-17）

计划当前这一步：**P4 评分与对照**（计划 §4、§9）。主评分已完成；**还没有与实验对照**。

| 阶段 | 状态 |
|---|---|
| P0 输入 | 完成。8 条竞争 SPR Kd；末端、GAI、水、altloc 的决定见 §2 |
| P1 入口 | 完成。prep、setup/run、strip、评分、汇总脚本都已写好并实际用过（§5） |
| P2 结构诊断 | 完成。v1、v2 的问题导致换成 v2.1（设计稿 §1） |
| P3 采样 | 完成。v2.1：8 样本 × 2 起点 × 3 副本 × **20 ns**，立体化学类别变化 0，没有解离 |
| P4 评分 | 主评分完成：1–20 ns 每帧打分，91200 帧，0 帧被标记（设计稿 §4）。**数值对照（设计稿 §5）待用户确认** |
| 实验对照 | 没有做。之前要先在 protocol §3 固定终点和方向 |

**1–20 ns 样本分数**（kcal/mol，越负越好）：PMI −61.41、A4L_L10A −60.59、S11M_L10A −60.28、PMI-L10A −59.82、PMI-M4 −59.75、T1L_L10A −58.95、P12Q_L10A −57.57、N8A_L10A −56.85。1–10、10–20、1–20 ns 三个窗口的次序相同；按副本统计只能稳定分出前 5、后 3 两组。

### 待决（等用户）

1. **问题 0：这个体系适不适合 PBSA**（设计稿 §6）。目前判断：粗分可能可以，细排序很可能做不到。五个 L10A 双突变体的实验 Kd 只跨 0.5 kcal/mol。拟做的检查是突变位点的界面接触和埋藏（只读已有轨迹），等用户同意并指定机器。
2. **P4 数值对照方案**（设计稿 §5）：N1 网格、N2 Amber MMPBSA.py、N3 MM/GBSA、N4 不跑 MD 的基线。要用户确认方案，并定 N2/N3 的 CPU 放在哪里。
3. **实验对照前要固定的**（写进 protocol §3）：主要终点（Spearman、Kendall，按副本做 bootstrap）、方向约定、要不要报 top-k。

### 其他待办

- `sample_structure_audit.csv` 的 ready 标记已按 v2.1 更新（§2.3）；PMI-M4 的末端仍是推断。
- `strip_peptide_replica.py` 的 `time_ps` 列有错（设计稿 §7），尚未修。
- 完整成本表（准备、MD、编译、评分、I/O）还没整理，各步的实测耗时散在设计稿 §3、§4。
- 首轮之后的扩展（同系列保留集、MDMX、其他靶点）按计划 §6–§7，等首轮结论。

## 2. 任务与输入

### 2.1 任务

用已发表的实验测量和现有 PDB 结构，做回顾性的 MD / MM-PBSA 分析。首轮固定为 Li X. 2021、MDM2 25–109、竞争 SPR 测得的 8 个样本，复合物模板 **3EQS**。不再扩充数据源（用户决定）。

### 2.2 实验标签（已对原文复核）

| 样本 | 原文位置 | Kd，nM |
|---|---|---:|
| PMI | Table 3 | 2.7 ± 0.5 |
| PMI-L10A | Table 2、Table 3 重复列出；记录 Table 3 | 897 ± 33 |
| T1L / L10A | Table 2 | 123 ± 10 |
| A4L / L10A | Table 2 | 98 ± 5 |
| N8A / L10A | Table 2 | 229 ± 11 |
| S11M / L10A | Table 2 | 198 ± 4 |
| P12Q / L10A | Table 2 | 132 ± 2 |
| PMI-M4 | Table 3 | 31.6 ± 2.3 |

- 条件：25 °C、pH 7.4、10 mM HEPES、150 mM NaCl、0.005% P20；原文写三次独立实验、各有重复。表中的 ± 不擅自定义为 SD 或 SEM。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8463443/)
- 8 条都是同一竞争 SPR 的 Kd，没有混入 FP Ki 或 PMI-M3 的 ITC Kd。
- 包内文件：`pilot_measurements.csv`。
- 来源：原文 `experiments/2026-10-10_peptide_samples/raw/mdm2_article.html`，模板 `…/raw/mdm2_3EQS.pdb`，历史清单 `…/manifests/pilot_mdm2_8.csv`。`raw/` 已加进 `.gitignore`，不推送到远端。
- **使用边界：** Kd 不用于挑起点、调介电或半径、缩短某个样本的采样。主评分的规则都在看实验排名之前固定（protocol §2.1）。

### 2.3 输入化学和结构决定（详见 protocol §1）

- **末端：** 肽和受体都是自由的 NH3+/COO−，不加 ACE/NME。依据：2021 SI Table S1 中，7 条的声明质量与自由末端的理论平均质量相差 ≤ 0.01 Da；C 端酰胺化会差 −0.98 Da，ACE/NME 会差 +55 Da。这些是声明的理论质量，不是实测读数。PMI-M4 不在表中，它的末端是推断。对照表：`inputs/termini_mass_check.csv`。
- **GAI：** 8 条全删。它挨着 His96 ND1（2.86 Å），SPR 缓冲液和结晶条件里都没有胍。
- **晶体水：** 117 个全删，用 TIP3P 重新溶剂化。其中 4 个同时与受体和肽有近接触（A:HOH1、A:HOH185、A:HOH186、B:HOH74），strip 时报告它们的再占据情况，不设门。
- **交替构象：** 15 个受体残基统一取 altloc A。其中 Lys51、Leu57、Met62 挨着肽，以后可能需要做敏感性检查。
- **缺失残基：** 3EQS 缺 A:Glu25 和 B:Pro12。第 12 位按各样本的序列补建（P12Q 补 Gln）。PMI 是需要补缺失区域的实验参考；其余 7 条是同位点模板模型，不能称为实验结构。
- **质子化：** His73、His96 为 HIE（v2.1 由 reduce 选择，用户确认）；复合物净电荷 +4，肽 −1。
- **`sample_structure_audit.csv` 的 ready 标记（10-17 按 v2.1 更新）：** 8 条的 `structure_ready`、`parameter_ready` 都为 true，依据是 C2、C3 通过、48 个副本运行正常。`chemistry_ready`：7 条为 true；PMI-M4 为 false，因为末端是推断，按 protocol §1.1 作为工作假设使用。

## 3. 准备协议

现行协议是 **v2.1**：181 上用 MODELLER 补建、做突变（只出重原子），183 上用 reduce 加氢，然后过 C1–C4 检查。流程、规则、阈值和为什么从 v1、v2 换过来，都在设计稿 §1–§2。

旧数据：
- v1（PDBFixer）的 15 个副本：`md/<样本>/model01|model02/`，只用于 PMI、PMI-L10A、P12Q_L10A；
- v2 在 yayoi40 上的 prep/setup：`v2m1|v2m2/`。

这两批保留为诊断数据，不进主分析。v1 P2 诊断（10-12，fulda RTX 5090）的要点：9 个副本、10 ns，类别变化 0，没有解离，补建的 Pro12 动得最厉害（第 12 位 CA 偏移中位数 1.7–6.2 Å）。完整表格见存档日志。

## 4. 运行与结果

全部结果在设计稿 §3（MD）和 §4（评分），这里只列入口：

- MD 和结构诊断：`md/<样本>/v21m{1,2}/rep0{1,2,3}/analysis/`（`diagnostics_summary.json`、`diagnostics.csv`）。
- 主评分：`scores_1_20/<样本>/<模型>/<副本>/frames.csv`、`summary.json`。
- 主汇总：`scores_1_20_summary/`（`summary.json`、`window_samples.csv`、`block_means.csv`、`structure.csv`、`sizing_paired.csv`）。
- 敏感性窗口：`scores/`（1–10 ns）、`scores_10_20/`（10–20 ns）。这两段的网格是分开定的，只作敏感性分析。

## 5. 文件地图（包 `experiments/2026-10-11_public_data_execution_packet/`）

| 路径 | 内容 |
|---|---|
| `pilot_measurements.csv`、`sample_structure_audit.csv`、`provenance.json`、`packet_check.json` | 标签、逐样本审计、来源、表格一致性检查 |
| `protocol.md`、`run_commands.md` | 协议和命令 |
| `proposed_runs.csv` | 全部副本的身份、种子和输出目录（v1、v2、v2.1 各段，phase 字段区分） |
| `inputs/3eqs_audit/`、`inputs/modeller_v2/`、`inputs/modeller_v21/` | 3EQS 静态审计；MODELLER 第一段；第二段、reduce 和最终起点 |
| `prepared/<样本>/<模型>/` | `parameters.npz`、`system_vacuum.xml`、`complex_dry.cif`/`.pdb`、`prep_record.json` |
| `md/<样本>/<模型>/setup/`、`…/rep0N/` | setup 记录、`status.json`、生产块（`prod/`）、`analysis/` |
| `scores*/`、`scores_1_20_summary/`、`scores_ext20/` | 评分与汇总（`scores_ext20/` 是用分段定尺拼接的旧汇总，已被 `scores_1_20_summary/` 取代） |
| `scripts/` | 包内脚本：MODELLER、reduce、finalize、汇总、诊断 |
| 项目 `scripts/` | `prep_peptide_sample.py`、`run_peptide_md.py`、`strip_peptide_replica.py`、`score_peptide_replica.py`、`peptide_geometry.py` |
| `*.pbs`、`*.sh`、`*.log` | 作业脚本和日志 |
| `diagnostics/` | v1/v2 立体化学诊断的 trace |
| `archive/` | 旧版清单、v1 冒烟产物、整理前的 protocol 和 run_commands |

`.gitignore`（10-17）排除了 MD 的大文件（`*.dcd`、`md/**` 下的 xml、cif、pdb、log 和 `diagnostics.csv`）、`system_vacuum.xml`、敏感性窗口的逐帧 CSV、第三方原文和下载的 SI、PPIKB 派生大表，以及 PBS 的 stdout 文件。推送到远端的只有记录、脚本、主评分和汇总。

## 6. 操作要点

- **机器分工（用户规则）：**
  - 181 只跑 MODELLER（系统 python 3.14）；
  - reduce 在 183 上跑；
  - prep、setup、MD 和评分在用户指定的 PBS 节点上跑，由用户 `qsub`；
  - 评分汇总脚本是我在 183 上跑的，只读 CSV，用户没有另作规定。
- **PBS 模板：** zsh；`-q default`；`select=1:ncpus=N:ngpus=1:mem=60gb:host=<节点>`；激活 `openmm_dev`（生产环境，只读）；日志写进包内。
- **用过的节点：**
  - yayoi42（4080）：MD 104.5 s/ns；
  - yayoi43（4080 SUPER）：MD 117 s/ns，评分 115 ms/帧；
  - yayoi31（4080 SUPER）：只跑过评分，91200 帧连同初始化共 3 h 37 min；
  - yayoi28（3090）：评分 120–128 ms/帧；
  - yayoi26 的 GPU 在 10-15 不可见，作业一启动就失败。
- **续跑：** 所有作业重新 `qsub` 同一个脚本即可，已完成的块和已打的帧会跳过。评分换窗口要换输出目录。
- **NFS：** 181 上的文件属性和目录列表会滞后。核对结果时在 183 上读，或者直接读文件内容。
- **git：** 工作区在 NFS 上，我不执行 git 命令；提交由用户在实际机器上做，commit 草稿在 `.commit_drafts/`。
