# C 端起点对照：肽骨架取自 3LNZ（model v21mx1）

日期：2026-11-07（JST）。依据：[PPI 结构计划](../../docs/PPI_STRUCTURE_RECOVERY_PLAN_20261106.md) §11；R3 说明见 [pose_evidence.md](../2026-11-06_ppi_structure_review/r3/pose_evidence.md)。

## 要回答的问题

v21m1/v21m2 的肽 C 端起点，是 3EQS 中 1–11 位的构象加上 MODELLER 补建的第 12 位。这套对照要回答：如果改从实验观测到的 C 端构象出发，结构结果和评分是否不同？对照中只改变这一个因素：

- PMI 与 P12Q_L10A 的肽 1–12 位重原子，取自 3LNZ（MDM2 + PMI-N8A）的肽链 H。叠合时用的是与它配对的受体链 G 的 CA 27–108 位，这份拷贝解析到了 Pro12。
- 受体（含 A25）逐字节复制自对应样本 v21m1 的第一段起点，原包中的 `inputs/modeller_v2/<PRO|GLN>/start_v2m1.pdb`。
- 加氢、prep、setup、MD 和评分的协议与 v21m1 完全相同；入口脚本没有改动，只是都通过 `--packet` 指向本目录。

**用户决定（2026-11-07）**：第 12 位对照回答"可"；起点方案选"整条肽骨架取自 3LNZ（推荐）"。这一方案是在实测发现"只换第 12 位接不上 3EQS 的 Ser11"之后提出的，因为在第 8–11 位的任何一处拼接，接口肽键都长 1.5–2.4 Å。

## 构建过程（全部已完成）

| 步骤 | 机器 | 脚本 / 产物 | 结果 |
|---|---|---|---|
| 1x 拼接起点 | 181，numpy | `scripts/build_xstart.py` → `inputs/modeller_v2x/<样本>/start_v2mx1.pdb` | 通过 C1 定义的检查：受体位移 0.000 Å；188 个立体化学项无异常；骨架项最大 2.0σ；受体–肽最近 2.72 Å |
| 2x 突变 | 181，MODELLER 10.8 | `scripts/build_xmutant_modeller.py` → `inputs/modeller_v21/<样本>/heavy_v2mx1.pdb` | PMI：A8N；P12Q_L10A：A8N、L10A、P12Q |
| 3 加氢 | 183，reduce 4.10，参数与 v2.1 相同 | `reduced_v2mx1.pdb` | 原子数为 1644 / 1638，与 v21m1 相同；His73、His96 都是 HIE |
| 4 finalize | 181 / 183，原 `finalize_allh_models.py` 的 `finalize()` | `scripts/finalize_x.py` → `start_v21mx1.pdb` | 通过；HA 偏角 0°，重原子位移 0.000 Å |
| 5 prep（C2） | 183（冒烟） | `prepared/<样本>/v21mx1/` | 通过 |
| 6 setup（C3） | 183 的 2080 Ti（冒烟） | `md/<样本>/v21mx1/setup/` | 通过；类别变化 0；PMI 撤约束时最大受力 1127（Thr49 OG1），P12Q 为 1481（Gln12 CA） |

**一次失败与修正**：第一次构建 P12Q 时，C2 检查失败——Gln12 的 CB–CG 键长 1.71 Å、CG–CD 1.74 Å。原因是 MODELLER 把模板中 Pro12 的 CG、CD 坐标沿用给了 Gln 的同名原子。修正办法是建模前从模板里删掉突变残基 CB 以外的侧链原子（脚本中标为 change 4）。修正后键长为 1.523 和 1.522 Å，molpdf 从 80.4 降到 6.6。失败的产物保留在 `failed_20261107_p12q_c2/`。

## 正式运行（待用户 qsub）

```bash
cd /home/ruigengji/JAXPBSA/experiments/2026-11-07_ppi_cterm_control
qsub ppi_cterm.pbs        # yayoi08（2 × RTX 3090，用户 2026-11-07 贴出该节点空闲）
```

PMI 和 P12Q_L10A 各占一块卡（`GPU_SLOT=1/2`，各用 16 线程），日志分别为 `logs/run_control_PMI.log` 和 `logs/run_control_P12Q_L10A.log`。

- 运行内容（`run_control.sh`）：2 个样本 × 3 个副本 × 20 ns，然后 strip，再对 1–20 ns 每一帧打分，输出到 `scores_1_20/`。中断后重新 qsub 即可续跑。
- 种子：沿用 v2 设计稿 §2.5 的公式，模型号取 9，与原包中的种子不冲突。

## 产物对应关系

`proposed_runs.csv`（6 行，phase 为 `R3_cterm_control`）、`sample_structure_audit.csv`（从原包摘出 2 行，只改了 `new_prepared_directory` 和 `remaining_tasks` 两列）。对照组是原包中的 `md/<样本>/v21m{1,2}/` 和 `scores_1_20/`。

## 运行记录

- 作业 24803（yayoi08，2026-11-07 17:35 起）：yayoi08 的 PBS 设置了 `CUDA_VISIBLE_DEVICES=0,1`，而脚本只在这个变量未设置时才挑卡，结果两个进程都用 device 0，同挤在 GPU 0 上，GPU 1 空着。生产段约 388 s/ns（4080 单独跑是 117 s/ns）。18:32 时两个样本都只完成 rep01 的 7/20 ns。随后把 `run_control.sh` 改为从已设的列表里取第 `GPU_SLOT` 块卡，由用户 qdel 后重新提交；已完成的 ns 块会被跳过。
- 作业 24808（yayoi08，2026-11-07 18:36–21:29）：修正后 PMI 用 GPU 0，P12Q_L10A 用 GPU 1，3090 单卡实测 166–169 s/ns。6 个副本都完成 20 ns；strip 各 2000 帧，没有手性或顺反类别变化；1–20 ns 各打分 1900 帧，0 帧被标记。rep01 的前 7 ns 是在 24803 中两个进程共用一块卡时跑的，只是更慢，计算协议不变。

## 结果（2026-11-08，`scripts/compare_control.py`，在 183 上只读运行，输出在 `results/`）

比较前定好的量：每个副本 1–20 ns 的 ΔG_MM/PBSA 均值，以副本为统计单位；各残基 CA 在 3EQS 受体坐标系中的偏移，取每个副本 20 个 1 ns 检查点的中位数，定义与 R3 相同。

| 样本 | 旧起点 v21m1+v21m2（6 个副本） | v21mx1（3 个副本） | 差值 ± SE | 旧数据中两个起点之差 |
|---|---|---|---|---|
| PMI | −61.41（SD 0.74） | −62.43（SD 1.74） | −1.02 ± 1.05 | 1.12 |
| P12Q_L10A | −57.58（SD 0.77） | −56.02（SD 3.13） | +1.56 ± 1.83 | −0.32 |

- **评分**：现有副本下暂未分辨出起点效应，尚未建立起点等效性（措辞按用户 2026-11-08 决定）。两个样本的差值都在约 1 个 SE 以内，与旧数据中两个起点之间的差同一量级；没有预先设定等效区间。新旧起点分开报告，不合并成 9 个副本。P12Q 的差主要来自 rep03（−52.5），这个副本出现了 N 端松开：B1 中位偏移 5.9 Å，B2 3.5 Å。
- **PMI 的 C 端**：v21mx1 的 3 个副本中，B12 始终在 3LNZ Pro12 附近（中位 1.5–1.75 Å），B11 中位 2.1–2.4 Å，与起点（2.26 Å）持平。旧模型中，v21m1 的 r02、r03 远离到 5.4、6.9 Å，其余 4 个副本是 1.5–3.8 Å。漂移在旧模型中是 2/6，在 v21mx1 中是 0/3，样本太少，不能据此说漂移是补建起点造成的。
- **P12Q 的 Gln12**：起点 rotamer 为 t;t;g+，5UMM 中是 g+;t;t。MD 中 χ1 有 3% 的检查点落在 5UMM 的区间（旧模型为 0% 和 15%）；CA12 距 5UMM 的 Gln12 中位 2.9–3.5 Å。5UMM 是 M3 复合物，第 9–12 位的局部序列不同，所以这只是一个带环境差异的参照。按用户 2026-11-08 的决定，后续不以转到它的 rotamer 为目标；P12Q 两个模型保留诊断和结构敏感性分析，暂缓精细姿态解释。

