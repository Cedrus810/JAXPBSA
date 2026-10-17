# WP1 数据审计：MDM2 多肽的公开实验数据

日期：2026-10-10。对应 `docs/JCIM_PEPTIDE_PAPER_OUTLINE.md` §9 的 WP1，交付"实验标注清单、序列/修饰、构建体、条件、结构来源、出处"。

**核实状态约定**：标【已核实】的，是本次直接读原文页面确认、并在本文件里按原文数值抄录的；标【未核实】的，来自子 agent 的文献检索报告，没有逐项回到原文核对，不能当事实使用，引用前必须自己核一遍。

## 0. 结论

| 问题 | 结论 |
|---|---|
| 靶点 | **MDM2 继续**，不换 |
| WP2 第一批数据源 | **Li C. et al., J Mol Biol 2010, 398:200–213 的 Table 3**（p53 17–28 单点丙氨酸扫描） |
| BindCraft 设计候选 | **不进 WP2**，移到 WP3 做分类/富集检验（理由见 §2） |
| 关键优势 | 该表的 MDM2 构建体是 **25–109**，与 1YCR chain A 完全一致；候选都是单点突变，共用一套制备流程 |
| 动态范围 | Kd 51 nM → 27 µM（单点突变内），折合 ΔG 跨度 **约 3.7 kcal/mol**；另有 2 条弱到测不出的 |

## 1. 主数据源：JMB 2010 Table 3【已核实】

- 出处：Li C., Pazgier M., et al. *J Mol Biol* 2010, 398:200–213。DOI 10.1016/j.jmb.2010.03.005，PMID 20226197。
- 开放获取全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC2856455/
- Table 3 原标题：*Dissociation equilibrium constants (Kd ± SD, M) of 17–28 p53 and 15 analogs for MDM2 and MDMX*。

### 1.1 实验条件（原文引述）

| 项 | 内容 |
|---|---|
| MDM2 构建体 | "Synthetic 25–109MDM2 and 24–108MDMX were freshly folded as described." → **MDM2 25–109**，化学合成，无标签 |
| 方法 | SPR **竞争法**（Biacore，p53 固定在芯片上），不是直接结合 |
| 缓冲液 | "10 mM HEPES buffer containing 150 mM NaCl, 0.005% surfactant P20, pH 7.4" |
| 温度 | "performed at room temperature" —— **没有给具体数值** |
| 末端修饰 | "all our peptides were neither N-acetylated nor C-amidated" → **自由 N 端 / 自由羧基 C 端** |
| 测不出的处理 | Table 3 脚注：F19A 与 W23A 的结合 "too weak to reliably measure, thus listed as 'not determined.'" |

### 1.2 Table 3 全表（MDM2 列，按原文抄录）

p53 17–28 = E17 T18 F19 S20 D21 L22 W23 K24 L25 L26 P27 E28。pKd 与 ΔG 由 Kd 换算（ΔG = RT ln Kd，T = 298.15 K），不是原文数字。

| # | 突变 | 序列 | Kd (M) 原文 | pKd | ΔG (kcal/mol) |
|---|---|---|---|---|---|
| 1 | P27A | ETFSDLWKLLAE | 5.1±0.3E-08 | 7.29 | −9.95 |
| 2 | S20A | ETFADLWKLLPE | 2.1±0.1E-07 | 6.68 | −9.11 |
| 3 | K24A | ETFSDLWALLPE | 2.3±0.2E-07 | 6.64 | −9.06 |
| 4 | E28A | ETFSDLWKLLPA | 2.4±0.2E-07 | 6.62 | −9.03 |
| 5 | **WT** | ETFSDLWKLLPE | 4.4±0.4E-07 | 6.36 | −8.67 |
| 6 | E17A | ATFSDLWKLLPE | 5.6±0.2E-07 | 6.25 | −8.53 |
| 7 | L25A | ETFSDLWKALPE | 7.3±0.1E-07 | 6.14 | −8.37 |
| 8 | D21A | ETFSALWKLLPE | 8.3±0.2E-07 | 6.08 | −8.30 |
| 9 | T18A | EAFSDLWKLLPE | 1.2±0.1E-06 | 5.92 | −8.08 |
| 10 | L22A | ETFSDAWKLLPE | 5.0±0.4E-06 | 5.30 | −7.23 |
| 11 | L26A | ETFSDLWKLAPE | 2.7±0.1E-05 | 4.57 | −6.23 |
| 12 | F19A | ETASDLWKLLPE | **n.d.** | — | 删失 |
| 13 | W23A | ETFSDLAKLLPE | **n.d.** | — | 删失 |

同表还有 5 条**截短**肽（ETFSDLWKLL 7.5±0.2E-08；TFSDLWKLL 1.0±0.1E-06；FSDLWKLL 3.5±0.4E-05；FSDLWKLLP 1.4±0.3E-04；FSDLWKLLPE 1.2±0.2E-04）。长度变了，网格定尺与封端决策都不同，**不进第一批**，留给 WP3 扩展。

### 1.3 与现有 1YCR 制备的差异

| 项 | 1YCR 现状（`scripts/prep_1ycr.py`） | JMB 2010 实验 | 处理 |
|---|---|---|---|
| 受体 | chain A，MDM2 25–109 | MDM2 25–109 | **一致，不动** |
| 配体长度 | chain B，p53 **17–29** ETFSDLWKLLPEN | p53 **17–28** ETFSDLWKLLPE | **截掉 N29** |
| 配体末端 | ACE / NME 帽 | 自由 N 端 / 自由 C 端 | **待定，见下** |
| 受体末端 | ACE / NME 帽 | 合成全长片段 | 保持加帽 |
| 力场 | ff14SB | — | 不变 |

**封端决策（需要定）**：实验用的是自由末端，现有脚本两端都加 ACE/NME。两个选择——

- **(a) 保留 ACE/NME**：与实验的化学形式不符，但避免在一个静电本来就弱的体系里凭空多出 ±1 电荷（这正是 `prep_1ycr.py` 开头写加帽理由的地方）。若封端带来的是各候选一致的偏移，靶点内排序不受影响，而 WP2 的主指标正是靶点内 Spearman。
- **(b) 改自由末端**：与实验一致，但 12 条候选全部要重新验证净电荷与质子化态。

倾向 (a)，并在论文里写明这是模型选择、不是实验复现。【未核实】子 agent 称原文有一句"封端会让结合一致减弱约 2 倍（data not shown）"，我没在原文里核到这句，**不能作为 (a) 的依据写进论文**，要用就先核。

## 2. BindCraft 设计候选：不进 WP2

- 出处：Filius et al., *ACS Chem Biol* 2025, 20(12):2991–2998，DOI 10.1021/acschembio.5c00774，PMC12723675。

### 2.1 已核实的内容

直接读 PMC 全文确认（原文引述）：

- "Fifteen peptides were successfully purified, and screened for binding to MDM2 on the BLI"
- "Seven of the peptides displayed clear association and dissociation kinetics with high-quality fits (R 2 > 0.95), indicative of specific binding"
- 设计肽的 Kd："yielding dissociation constants (KD) in the low nanomolar range (**65–165 nM**"
- p53 参照肽："obtaining a KD of **123 nM**"
- **MDM2 构建体：原文完全没写**。Methods 只描述了 BLI 流程和 WDR5 的表达纯化，没有 MDM2 的残基范围、标签或来源。
- 原文内部矛盾：先说 seven，后文又提到 "the eight peptides with reliable kinetics"。

### 2.2 四条不适合做 WP2 第一批的理由

1. **动态范围太窄**：7 条结合者挤在 65–165 nM，折合 ΔG 跨度仅约 0.55 kcal/mol。MM/PBSA 的 replica 间抖动通常就在这个量级甚至更大，分不开。对照 JMB 2010 的 3.7 kcal/mol。
2. **没有基线排序**：大纲 §5「基线与公平比较」要求有"原始设计/结构评分"。该文没有公开每条候选的 AF2/BindCraft 分数表，设计结构也没有存库（数据可用性声明只说"available from the corresponding author upon request"）。要有这个基线，得自己重跑 BindCraft，那是另一个工作量。
3. **标签是分类不是亲和力**：binder/non-binder 来自单浓度 1 µM 下的 R²>0.95。【未核实】子 agent 从 SI Table 1 抄出的拟合 Kd 显示，被判 non-binder 的若干条拟合值反而比 binder 更小 —— 若属实，这套标签只能做分类/富集，不能做排序。这条若要写进论文，必须自己核 SI。
4. **逐条建模成本**：序列长度 12–20 不等，每条都要独立准备、独立定网格，与 §2 的窄范围不成比例。

### 2.3 保留用途

按大纲 §4B，这套数据适合做**检测候选的分类/富集检验**，放 WP3。届时需要先解决：MDM2 构建体未知（只能假定 1YCR 的 25–109）、设计结构需向作者索取或重建、基线分数需重跑。

【未核实】子 agent 还报告了 16 条候选序列（来自 SI 的 LC-MS 图）、SI Table 1 的逐条动力学参数，以及一处 ID 对不上的问题（Table 1 有 B16 行，但序列图里只有 B15/B19、没有 B16）。这些都没有回到 SI PDF 核对，用之前要核。

## 3. 其他来源：本轮未核实

以下全部来自子 agent 报告，**我没有回到原文核对任何一项**，列在这里只作为后续线索。

| 来源 | 子 agent 的说法 | 若要使用需先做 |
|---|---|---|
| JMB 2010 **Table 1**（PMI 系列 17 条，Kd 3.2 nM–160 µM） | 同一篇、同一条件，可做第二层 | 核 Table 1 原文；PMI 需要另一个晶体骨架 |
| PDB **3EQS / 3LNZ** | MDM2 链与 1YCR 同为 25–109，可做 PMI 的骨架 | 核 PDB 条目的实际残基范围 |
| Ye et al., *Commun Chem* 2022, 5:128（组合丙氨酸扫描） | 多数 Kd 只存在于图里；有订书钉和非天然化学；无弱端数据 | 若确认如此，则不适合，排除 |
| Pazgier et al., *PNAS* 2009 | 同实验室，但表格少，只能交叉检查 | 核 SI 表 |
| Schon et al., *JMB* 2002 | 付费墙，仅摘要可见 | 取得 PDF |

大纲 §10 提到的"第二靶点"：【未核实】子 agent 称同篇 BindCraft 论文的 WDR5-MYC 位点有 9 条测试、6 条结合、带 R²，可作候选；WIN 位点、PD-1、PD-L1 全阴性无动力学。需要自己核 SI 后再定。

## 4. WP2 候选集提案

从 §1.2 取 8 条定量 + 2 条删失 = **10 个体系**。都是 1YCR chain B 的单点突变（先截 N29），受体不变，一套拓扑流程通吃。

| 组 | 候选 | Kd | 作用 |
|---|---|---|---|
| 强 | P27A (51 nM)、S20A (210 nM)、K24A (230 nM) | 51–230 nM | 比野生型强的一端 |
| 中 | **WT** (440 nM)、D21A (830 nM)、T18A (1.2 µM) | 参照点及其附近 | 细粒度排序的主要考验 |
| 弱 | L22A (5.0 µM)、L26A (27 µM) | 微摩尔 | 热点破坏 |
| 删失 | F19A、W23A | 测不出 | 必须排在最后，否则评分不可信 |

选择依据：覆盖 §1.2 的全部量级；同时包含 §4 大纲要求的"强结合与弱结合变体"；F19A/W23A 是公认的疏水热点，作为"连这个都排不对就不用往下做了"的检验。未入选的 E17A、L25A、E28A 留给 WP3 扩到全表。

**风险（大纲 §9 已预先写明）**：如果结果只能把 L26A/F19A/W23A 这类严重破坏热点的分出来，而 210–830 nM 之间分不开，那就不能宣称细粒度亲和力排序，按大纲退回更窄的 Application Note。这是 WP2 要回答的问题本身，不是失败。

## 5. WP2 开工前的软件缺口

| # | 缺口 | 现状 | 需要 |
|---|---|---|---|
| 1 | 制备脚本不支持突变 | `scripts/prep_1ycr.py` 把 1YCR 写死：源文件、链、加帽、输出名全是常量 | 一个接受"序列/突变 + 骨架"的制备入口，保留现有的幂等性保证（固定种子、Reference 平台、单线程最小化、sha256 校验） |
| 2 | MD 脚本无 replica 支持 | `scripts/run_s4_md.py` 有 `--name`，但**没有 `--seed`**，输出名不区分 replica；写 checkpoint 但不会从 checkpoint 续跑 | `--seed` / `--replica`，输出名带 replica 号，支持重跑同一条命令时跳过已完成的 |
| 3 | 无多候选评分汇总 | 只有单体系的 `crl.py` / `validate_mmpbsa.py` | 按候选 × replica 汇总：分项、score、有效帧数、数值 flag、replica 间不确定性（大纲 §6 第 3 条） |

缺口 1 和 2 不补就没法跑 WP2；缺口 3 可以在 MD 跑着的时候写。

## 6. 未解决、需要决定的事

1. **封端**：按 §1.3 选 (a) 还是 (b)。
2. **采样预算**：大纲 §5 写"每条候选至少三条独立 replica，长度由试验阶段的统计决定，不预设 10 ns 便足够"。1YCR 现有轨迹是 10 ns，可作为起点，但需要先用一两条候选测构象与评分的 replica 间抖动，再定正式长度。
3. **温度**：实验只写"room temperature"，没有数值；我们的 MD 与 PB 都在 298.15 K。论文里要写明这是假定。
4. **JMB 2010 Table 1（PMI 系列）**是否纳入：取决于是否要引入第二个晶体骨架。
