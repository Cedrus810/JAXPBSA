# JAXPBSA 当前后续任务

更新：2026-11-06（JST）。详细定义、输入、交付与完成条件以 [后续执行计划](docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md) 为准；论文组织以 [在线方法论文框架](docs/ONLINE_MMPBSA_PAPER_OUTLINE.md) 为准。

当前状态：**BRD4 / Hou 本地方法论文收尾已落实：正文/SI v0.2、五张主图、图源表、软件能力核查及源码/报告审阅包。** 入口：[收尾记录](docs/PAPER_CLOSEOUT_20261106.md)。

- [x] Hou 有界 6 × 10 ns 采样、越界恢复、严格审计与实验排序；见 [RESULTS §20.9–20.10](RESULTS.md)。
- [x] 后续路线成文，明确执行顺序与计算边界。
- [x] P1.0：静态核查已有起点、描述符来源及评分，形成输入/缺口清单（2026-11-01，包内 `reports/baseline_20261101/inventory.csv`，由包内 `scripts/build_baseline_inventory.py` 只读生成）。缺口：B1 起点分数与单结构评分入口（17 条）；BRD4 配体 sdf/mol2（可旋转键）。
- [x] P1.1：完成描述符、最小化起点评分与现有 MD 平均 PBSA 的基线对照（2026-11-01，包内 `reports/baseline_20261101/report.md`，RESULTS §21.2）。
  - B1 入口 `scripts/score_smallmol_start.py` 已写；2026-11-01 在 183（2080 Ti）冒烟通过（BRD4-3、a1 各 1 帧有效，原命令重跑跳过；`scores/smoke/start_default/`）。全量 17/17 已于 yayoi43 完成（作业 21818），`reports/baseline_20261101/start_scores.csv`，RESULTS §21.1。
  - BRD4 配体 sdf/mol2 已从 commit 5233e0a 补取并与 prmtop 逐条核对一致（`reports/baseline_20261101/brd4_ligand_file_check.csv`）；B0/B1/B2 输入已无缺口。
  - 协议 `protocol.json` 统计前写定；B0 描述符、对照表、成对表与报告已生成（`scripts/compare_baselines.py`）。
- [x] P2：补齐 BRD4 来源/测定分组统计，并列默认与 h0.4 结果（2026-11-01，包内 `reports/brd4_statistics_20261101/report.md`，RESULTS §21.3）。
  - 发现 BRD4-5 上游输入是 5,6-二氢阿普唑仑，不是实验化合物；已移出主统计。重新准备与重跑方案：`docs/BRD4_5_REPREP_PLAN_20261101.md`；已于 yayoi28 重跑完成（作业 22051），P1/P2 v2 在 `reports/{baseline,brd4_statistics}_20261101_v2/`，RESULTS §21.4。
- 2026-11-01 本轮全部实验（含补做）、杂项文件与冒烟/失败日志的说明：[P1/P2 会话说明](docs/P1_P2_SESSION_HANDOFF_20261101.md)。
- [x] P3 本地收尾：统一证据与成本、正文/SI、论文图表、源码与报告复现包。
  - W1/W3 既有证据和成本：`reports/evidence_20261102/`（RESULTS §21.7）。
  - W3 软件能力核查：`docs/PAPER_SOFTWARE_COMPARISON.md`；认可已有 GPU PB，不作跨工具优越性主张。
  - W4：正文/SI v0.2、五张 PDF/PNG 主图及图源 CSV（`docs/paper_assets/`）；采用修正 BRD4-5 的 v2，保留历史数值诊断范围与负结果。
  - W5 本地交付：`docs/PAPER_REPRODUCIBILITY.md`、`scripts/reproduce_paper.py`、`scripts/build_review_bundle.py`；只复现报告与图表，完整分子原始材料另存。
- [ ] 外部发布/投稿：作者署名与单位、公开存档地址/DOI、完整原始输入和轨迹的分发及许可归属。尚未执行，不影响本地审阅。
- [ ] P4：收口后决定是否扩展独立靶点/系列；目标与预算另定。

Hou 当前采样预算已结束。本轮只做稿件、报告检查和绘图；未追加 MD/PBSA、PPI 计算或 GPU 基准，未发布或投稿。

## PPI：结构问题接续（2026-11-06）

计划入口：[PPI结构诊断与接续计划](docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md)。范围为现有 MDM2–PMI；先解决结构解释，再接回数值/实验对照。

- [x] 整理已知错误、姿态证据缺口和后续步骤；按明确路径汇总16模型的历史检查记录与阶段文件存在性（`experiments/2026-11-06_ppi_structure_review/`）。
- [x] R0/R1（2026-11-07，6 个代表模型 + 其余 10 个）：未定位到 v2.1 硬错误；最早偏离晶体坐标的是 setup 最小化；MD 中的大变化在 B1–B3、B11–B12 和 A25，核心残基正常。见 `experiments/2026-11-06_ppi_structure_review/r0r1/structure_diagnosis.md`、`issue_ledger.csv`。
- [x] R2/R3（2026-11-07）：R2 只涉及诊断口径和分区报告。R3 用 3LNZ（N8A）和 5UMM（M3）作为实验参照，核对了 16 个模型；没有含 L10A 的复合物。建议见 `experiments/2026-11-06_ppi_structure_review/r3/pose_evidence.md`、`model_use_decisions.csv`。
- [x] 用途审阅（2026-11-08，本chat）：14个模型条件性继续，P12Q_L10A的2个保留姿态疑问。审阅人和适用范围见 `experiments/2026-11-06_ppi_structure_review/r3/model_use_decisions_reviewed_20261108.csv`、`review_20261108.md`；原提案保留，不改写为人类用户逐项确认。
- [x] C 端起点对照（2026-11-08 完成）：现有副本下暂未分辨出起点效应，尚未建立起点等效性（PMI −1.02 ± 1.05，P12Q +1.56 ± 1.83 kcal/mol）；新旧起点分开报告。
- [x] 用途审阅（用户 2026-11-08 确认）：14 个模型注明假设后进入 R4；P12Q 两个模型保留诊断和结构敏感性分析，暂缓精细姿态解释。
- [ ] R4：对 14 个模型做数值与基线验收；原 8 候选的完整结果保留，排除 P12Q 的结果只作结构敏感性分析。口径草案和冒烟（2026-11-08）见 `experiments/2026-11-08_ppi_r4/r4_protocol.md`；N1–N4 已在 yayoi30 跑完（作业 25019），结果见 `experiments/2026-11-08_ppi_r4/numeric_report.md`：MM 项与 Amber 一致；默认网格的 ΔG_PB 相对 h0.35 约 −10 kcal/mol，在样本之间相差约 1.1，与相邻样本的分差同一量级；h0.35 还不是收敛的参照。用户 2026-11-08 确认口径（"可以"），口径在读标签之前冻结（protocol §8）。实验对照结果见 `experiments/2026-11-08_ppi_r4/experiment_report.md`：8 条集合上 ρ = 0.48（只计副本抽样的区间 0.24–0.74），28 对中 20 对排序一致；没有胜过简单基线（起点单帧评分 0.57、起点 ΔSASA 0.50）。见 `experiments/2026-11-07_ppi_cterm_control/README.md`。
- [ ] R4：结构解释清楚后补数值/基线和实验评价；计算批次、资源和预算按实际缺口另定。

2026-11-08 当前停点：其他7个样本可准备R4；P12Q保留诊断/结构敏感性，原8候选结果完整保留。v21mx1单列，不与旧6副本合并；本次对照收口，不追加以追到5UMM rotamer为目标的MD。项目CLAUDE.md中计划入口和yayoi08选卡实测例外保留。

此处未完成项都是后续工作。本轮没有重新建模、参数化、最小化、读取整批轨迹或运行MD/PBSA；原48条轨迹和历史协议保持原样。 2026-11-07 的 R0/R1 只读了 PDB 和 state XML 检查点，在 181 上用 numpy 计算，没有读 DCD，也没有改原包。
