# PPI结构问题：本轮记录整理

日期：2026-11-06。计划入口：[PPI_STRUCTURE_RECOVERY_PLAN_20261106.md](../../docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md)。

`record_inventory.csv` 按原样本表的8个名字与固定的v21m1/v21m2读取明确路径，列出16个模型的构建、finalize、prepared和setup记录，以及所列坐标文件的存在性。`summary.json` 给出记录计数。

本轮16个模型均有finalize/C2/C3通过记录，所列阶段文件存在。字段 `*_recorded` 是历史记录中的值；本轮没有重算坐标几何、局部受力或MD诊断，没有新做结合姿态验收。`bound_pose_review=pending` 表示当前待查工作，不判定模型已错或已正确。

本目录未扫描轨迹、运行建模/参数化/最小化/MD/PBSA，也未修改原包、ready标记、历史分数或协议。后续问题定位与重跑决定应另写带来源的表格，按计划推进。

## 2026-11-07：R0/R1

`r0r1/` 存放 16 个模型的逐阶段重算（脚本在 `scripts/`）。结论和复现命令见 `r0r1/structure_diagnosis.md`，问题逐条见 `r0r1/issue_ledger.csv`。`record_inventory.csv` 里的 `atom_mapping_rechecked`、`bound_pose_review` 字段是 11-06 的记录，保持原样不改：映射已在 `r0r1/stage_atom_map_*.csv` 中重查过，结合姿态（R3）仍待查。

## 2026-11-07：R3

`r3/` 存放结合姿态的结构依据核对：从 RCSB 下载的 3LNZ、5UMM 在 `raw_pdb/`；比较表；`pose_evidence.md`；`model_use_decisions.csv`（建议，待用户审阅）。

