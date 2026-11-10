# R3 变体结合姿态的结构依据（MDM2–PMI，16 个模型）

日期：2026-11-07（JST）。计划：[PPI 结构接续计划](../../../docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md) §4 R3。
逐模型的用途建议见 [`model_use_decisions.csv`](model_use_decisions.csv)：**这是 Claude 拟的建议，`user_decision` 一列待用户审阅**。新增问题已记入 `../r0r1/issue_ledger.csv` L11–L14。

## 找到的实验参照

2026-11-07 用 RCSB 的序列模式检索了 `F.EYW..L` 和 `F.EYW..A` 两种模式。与 MDM2（25–109，与 3EQS 受体序列相同）结合、且和本批相关的条目：

| PDB | 肽 | 分辨率 | 拷贝数 | 与本批的关系 |
|---|---|---|---|---|
| 3EQS | PMI TSFAEYWNLLSP | 1.65 Å | 1 | 现有模板；B12 未解析 |
| 3LNZ | PMI-N8A TSFAEYWALLSP | 1.95 Å | 8 | N8A 的实验姿态；肽链 H、L 解析到 Pro12 |
| 5UMM | M3 LTFLEYWAQLMQ | 1.65 Å | 2 | 第 1、2、4、8、11、12 位与 T1L、S2T、A4L、N8A、S11M、P12Q 相同；肽链 B 解析到 Gln12 |

**没有任何含 L10A 的复合物。** 7 个变体都带 L10A；用 `F.EYW..A` 检索只命中 5 个无关的大蛋白。3EQY、4RXZ、5UML 的受体是 MDMX，不作为参照。原始文件从 `files.rcsb.org` 下载，存在 `raw_pdb/`，只作为文本解析。

## 实验结构说明了什么

做法：每条肽链先配上与它接触的受体链，按受体 CA 27–108 叠合到 3EQS 上，再逐残基比较。结果在 `reference_vs_3eqs.csv` 和 `reference_copy_spread.csv`。

- **N8A 与 M3 的结合姿态都与 PMI 相同。** 肽骨架相对 3EQS 偏 0.47–0.93 Å。1–9 位的 CA 偏移：3LNZ ≤0.83 Å，5UMM ≤1.03 Å。10–11 位偏得更多（最多 1.5 和 2.4 Å），但 3LNZ 自己的 8 份拷贝在 10、11 位之间也差到 1.5 和 3.0 Å。所以 C 端这一段在晶体中本来就不唯一。
- 因此，除 L10A 以外，本批涉及的突变位点（1、2、4、8、11、12 位）都有实验结构支持"沿用 PMI 骨架"。L10A 对结合姿态有什么影响，现有 PDB 无法检验。

## 模型与实验参照的比较

比较对象是每个模型的 prepared、minimized 和全部 60 个 1 ns 检查点，结果在 `model_vs_reference.csv` 和 `model_rotamers_vs_reference.csv`。

- **核心骨架**：MD 中 3–7 位相对最近参照拷贝的 CA 偏移中位数为 0.5–1.0 Å；PMI-L10A 只能和 3EQS 比，最大 1.1 Å。
- **C 端 8–11 位（L13）**：MD 中这几位离开 3EQS 的位置 1.2–2.8 Å，但离 3LNZ/5UMM 的位置多数只有 0.6–1.3 Å，比起点离得更近。例外是 PMI/v21m1，因为 r02、r03 的 C 端漂移，对 3LNZ 最大到 2.3 Å。旧诊断里"相对 3EQS 的偏移"中，这一段有相当部分是在移向其他晶体构象。
- **第 12 位（L12）**：MODELLER 补建的 Pro12 起点，距 3LNZ 中的 Pro12 3.4 Å。在 MD 中，PMI v21m1 r01、v21m2 r01/r03 以及 N8A_L10A 全部 6 个副本移近到 1.5–2.4 Å；PMI/v21m1 的 r02、r03 反而远离到 5.5 和 6.9 Å，也就是 R0/R1 账目中的 L06。P12Q 的 Gln12 起点距 5UMM 2.8–2.9 Å，rotamer 完全不同，MD 中 χ1 几乎从不落在晶体的那个区间。
- **突变侧链的 rotamer（L14）**：MODELLER 搭出的 Leu1、Leu4、Met11（χ1）和 Thr2，起点 rotamer 都与 5UMM 不同。注意 5UMM 两份拷贝的 Leu4 本身就不一致，起点恰好与其中 C/D 拷贝相同。MD 中，Leu1、Leu4 和 Met11 的 χ1 在 65–95% 的检查点落在晶体的区间；PMI-M4 的 Thr2 仍然偏离（两个模型分别只有 7% 和 47% 落在晶体区间）。5UMM 的邻近残基与本批变体不同（例如 M3 有 Leu10、Gln9），所以 rotamer 只能作为有上下文差异的参照。

## 用途建议（待用户定）

| 建议 | 模型 | 理由 |
|---|---|---|
| 继续使用，注明假设 | PMI、PMI-L10A、T1L、A4L、N8A、S11M、PMI-M4，各 2 个模型 | 核心骨架有实验依据，或与 PMI 同骨架；没有硬错误。假设是：B12 的位置来自建模，L10A 的姿态沿用 PMI；侧链起点 rotamer 的差异大多在 MD 中得到纠正 |
| 姿态问题待解决 | P12Q_L10A 的 v21m1、v21m2 | 决定这个样本的第 12 位是补建出来的，起点和 MD 都与唯一的实验参照（5UMM 中 M3 的 Gln12）不符 |

本轮没有发现需要重建或重跑的硬错误。

**用户决定（2026-11-08）**：14 个模型注明模板、末端和侧链假设，进入 R4；P12Q 两个模型保留诊断和结构敏感性分析，暂缓精细姿态解释，不以转到 5UMM 的 rotamer 为目标。原话见计划 §12，`model_use_decisions.csv` 的 `user_decision` 一列已填。

**可以考虑的结构对照（未做，需要用户定）**：用实验坐标作为第 12 位的起点重建一组 PMI 和 P12Q 模型——PMI 取 3LNZ 中的 Pro12，P12Q 取 5UMM 中的 Gln12——再与现有模型比较。这属于计划 R2/R3 中"有问题指向的少量结构对照"。它需要新的 MODELLER、prep、setup、MD 和评分，计算由用户安排；规模在定之前不做估计。

## 未确认（Claude 推断）

- 第 8–11 位在 MD 中移向 3LNZ/5UMM，说明这段的溶液构象更接近这些晶体中的状态。但 5UMM 和 3LNZ 的序列与本批不同，这一点没有对照验证。
- L10A 的姿态沿用 PMI 是合理的工作假设，依据是 MD 中核心残基稳定。不过 MD 稳定本身并不能证明姿态正确（计划 R3 也这样规定）。

## 复现

```bash
cd /home/ruigengji/JAXPBSA
curl -sSf -o experiments/2026-11-06_ppi_structure_review/r3/raw_pdb/5UMM.pdb https://files.rcsb.org/download/5UMM.pdb
curl -sSf -o experiments/2026-11-06_ppi_structure_review/r3/raw_pdb/3LNZ.pdb https://files.rcsb.org/download/3LNZ.pdb
OMP_NUM_THREADS=1 /home/ruigengji/miniforge3/envs/openmm_dev/bin/python experiments/2026-11-06_ppi_structure_review/scripts/r3_pose_evidence.py   # 181，约 2 分钟，只用 numpy 读文本
```
