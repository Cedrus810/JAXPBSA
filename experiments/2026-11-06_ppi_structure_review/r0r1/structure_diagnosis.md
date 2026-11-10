# R0/R1 结构诊断（MDM2–PMI，v2.1 全部 16 个模型）

日期：2026-11-07（JST）。计划：[PPI 结构接续计划](../../../docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md) §4 R0/R1。问题逐条见 [`issue_ledger.csv`](issue_ledger.csv)。

## 结论

**v2.1 的构建、加氢、prep、setup 和 MD 中，没有定位到具体的硬错误。** 先按计划做了 3 个代表、共 6 个模型（PMI、PMI-L10A、P12Q_L10A × v21m1/v21m2），再用同一个脚本、同一套口径核对了其余 10 个模型，结果一致。

已经排除的环节（本轮重算）：

| 环节 | 结果 | 账目 |
|---|---|---|
| 序列、原子映射、各构建阶段 | 实验观测到的重原子从 3EQS 到 prepared 始终不变（0.000 Å）。新增的只有 A25、B12 和氢；删除的只有各突变侧链。prepared 的 PDB 与 `parameters.npz` 一致 | L01 |
| 立体化学与几何 | 1184 个阶段快照中：没有 D 型 CA；没有顺式肽键（含 Ser11–Pro12）；Thr/Ile 的 CB 手性与 3EQS 相同；HA 偏角 ≤22.2°；没有 C2 定义的冲突；共价键最长 1.993 Å | L02 |
| 周期成像与诊断代码 | 用 960 个检查点重算，与 `diagnostics.csv` 相差 ≤0.001 Å，肽的周期平移量都是 0。91,200 个评分帧中，受体–肽 LJ ≤ −40.8 kcal/mol，库仑 ≤ −95.3 kcal/mol | L05 |

**最早偏离晶体坐标的是 setup 最小化**（L03）：肽 1–11 位骨架相对 3EQS 偏 0.65–1.05 Å，其中肽自身叠合后的 RMSD 只有 0.30–0.80 Å，质心平移 0.42–0.79 Å；受体 CA 偏 0.46–0.72 Å。这一步没有任何类别变化或 4σ 越界。偏多少算正常，没有现成判据，本轮也不新设。

**MD 中较大的变化都在末端，核心残基没有大变化。** 48 个副本中，B3–B7 各副本 CA 偏移的中位数为 0.52–1.69 Å，B8–B10 为 0.74–2.95 Å。大偏移集中在三处：

- **C 端（L06）**：PMI/v21m1 的 r02、r03 中，B12 中位偏移 7.6 和 8.3 Å，B11 约 4.1 Å。从 NVT 或生产第 1 ns 就开始。这就是旧记录里"第 12 位 CA 偏移 7.6–9.1 Å"的来源。
- **N 端（L07）**：A4L/v21m2/r03 中 B1 中位偏移 8.4 Å，最大 11.7 Å；S11M/v21m1/r01 中 B1 4.5 Å、B2 3.1 Å，B3 在 17–20 ns 升到 2.9 Å。旧记录里的"肽 RMSD 上升"副本主要就是这些。
- **受体 A25（L08）**：补建的受体 N 端在 3 个副本中漂移 4.6–9.0 Å。

## 口径问题（不影响结构结论）

- **L09**：原诊断里的"肽 CA RMSD"和"CA12 偏移"，参考的是真空 prepared 模型，并且把 B1、补建的 B12 和核心残基放在一起算。本轮按区域拆开：B1–11 对 3EQS，B12 对它自己的建模坐标，核心残基逐个报告。
- **L04**：最小化后受力最大的原子都是 O–H 成对的约束分量（约 860–1280 kJ/mol/nm），力完全沿键方向。把重原子和它的 H 合起来算，最大净力是 133–651。撤约束那一刻的 top-5 里，O–H 对和真实的 CA 受力（Asp46、Thr49 CA，1057–1303）混在一起。撤约束时的力没有保存，所以这里的区分只能按原子名推断。

## 没做 / 仍未定位

- 检查点之间的帧没有重读 DCD。每帧的手性和顺反类别变化，仍以已有 `diagnostics.csv` 的记录（0 次）为准。
- R3 的变体结合姿态有没有结构依据，没有检查（L11）。
- 用户原先看到的"结构不对"具体指哪里，本轮没有对上。如果它指上面的末端运动或旧 RMSD 数字，对应 L06–L09；如果是别的现象，需要样本、副本和文件位置才能复查。

## 按计划 R2 表，下一步落在哪一行

证据对应 R2 表中"展示/诊断口径"一行（L04、L09），和"只有端部运动较大，核心几何正常"一行（L06–L08）。按计划，这两行都是先完成分区诊断，不自动追加计算。本轮没有证据支持重建、重新 setup 或重跑。需要注意：L07 中的 B1–B3 是实验观测到的残基，不完全符合"未观测端部"的说法，这一点如实列出。

计划的下一项是 R3：逐个变体核对有没有对应的实验复合物、哪些部分继承自模板，然后填 `model_use_decisions.csv`。

## 未确认（Claude 推断）

- 最小化带来的 0.3–0.8 Å 肽内部变化和约 0.6 Å 的整体平移，是无约束最小化中常见的松弛；本轮没有做对照来核实。
- B11/B12 和 B1 的运动反映的是末端真实的柔性，而不是某个准备错误留在 MD 里的后果。依据是：各构建阶段都没有偏差，几何也全部正常；但我没有用不同的末端建模做对照来区分这两种可能。

## 复现

在 181 上用 `openmm_dev` 的 python 运行，只做 numpy 计算和文本/XML 解析：不导入 OpenMM，不读 DCD，不改原包。脚本在 `../scripts/`：

```bash
cd /home/ruigengji/JAXPBSA
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
OMP_NUM_THREADS=1 $PY experiments/2026-11-06_ppi_structure_review/scripts/stage_trace.py --samples PMI PMI-L10A P12Q_L10A --tag rep6   # 1 min 40 s
OMP_NUM_THREADS=1 $PY experiments/2026-11-06_ppi_structure_review/scripts/stage_trace.py --samples T1L_L10A A4L_L10A N8A_L10A S11M_L10A PMI-M4 --tag rest10
$PY experiments/2026-11-06_ppi_structure_review/scripts/scan_checks.py
python3 experiments/2026-11-06_ppi_structure_review/scripts/summarize_r0r1.py
```

产物：`stage_atom_map_*.csv`（各阶段的原子映射与增删）、`stage_geometry_*.csv`（逐阶段、逐检查点的分区几何）、`stage_residue_*.csv`（逐残基）、`diag_crosscheck_*.csv`、`imaging_scan_48.csv`、`minimized_forces_all16.csv`、`model_summary.csv`、`replica_summary.csv`。`views/<样本>_<模型>_stages.pdb` 是叠合到 3EQS 受体 CA 坐标系后的多 MODEL 结构，16 个模型各一个文件、每个 15 个 MODEL，依次为 3EQS、stage1、heavy、start、prepared、minimized，以及每个副本的 npt_end、10 ns、20 ns。表名后缀 `rep6` 是 6 个代表模型，`rest10` 是其余 10 个。
