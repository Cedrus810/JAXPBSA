# 本轮改动记录

对应两份 code review 的回应 + 后续实测。所有性能数字见 [RESULTS.md](./RESULTS.md)。

## 2026-11-06 · 在线方法论文本地收尾

- 正文/SI 更新至 v0.2：采用修正阿普唑仑的 BRD4 v2；更正实验值对应、逐帧残差范围、网格收敛和 worker/cache 表述；保留 Hou 改善与 BRD4 无增益的结果。
- `scripts/build_paper_assets.py`：从存量报告生成五张 PDF/PNG 主图、图源 CSV、标签/成本/42 副本附表，拒绝缺副本、标记丢失与旧输入混入。
- `scripts/reproduce_paper.py`、`scripts/build_review_bundle.py`：当前源码/报告快照、文件摘要、元数据与独立解包报告/作图复现；不冒充完整原始分子数据公开包。
- 软件能力/文献核查、复现说明、收尾记录、README 中英与任务索引已同步。25 项相关测试通过；解包核对及 22 个 CSV 复现通过，见 RESULTS §21.8。
- 未修改核心方法、环境和历史科学产物；未启动分子计算或 PPI 线。提交仅写草稿。

## 2026-11-02 · P3 W1/W3：证据清单、统计复现、成本表

- 包内 `scripts/reproduce_reports.py`：在仓库外镜像包里重跑全部已记录的汇总/统计命令并逐文件比对；`scripts/cost_table.py`：从 setup/status/summary/prep 记录抽取按 GPU 分组的成本。产物 `reports/evidence_20261102/`（RESULTS §21.7）。
- 修 `scripts/diag_hou_replicas.py`（固定 rep01–03、(0, 5] ns）与 `scripts/summarize_hou_online_acc.py`（只取验收的 10 个 chunk），使其在后来延长过的数据上仍复现原表。

## 2026-11-02 · BRD4 标签原文核对

- 用用户提供的论文与 SI 逐条核对 BRD4 十条实验标签，写 `reports/brd4_statistics_20261101_v2/label_verification_20261102.csv` 与报告末节（RESULTS §21.6）。BRD4-4 的值在所引论文中找不到；统计产物未改。

## 2026-11-01 · B1 最小化敏感性

- 包内 `b1_repeat.sh/.pbs`（只用现有入口，独立包 `b1_minimisation_repeat_20261101/`）在 yayoi37 重做 17 条最小化并评 B1；包内 `scripts/compare_b1_repeat.py` 对比正式 B1。另加只读分项脚本 `scripts/diag_brd4_8.py`。结果 RESULTS §21.4–§21.5。

## 2026-11-01 · BRD4-5 重跑完成，P1/P2 v2

- yayoi28 作业 22051 完成 BRD4-5（阿普唑仑）准备、3 × 5 ns、默认与 h0.4 评分和 B1（RESULTS §21.4）。
- 包内 `scripts/brd4_statistics.py`、`scripts/compare_baselines.py` 加 `--out` 和协议 `sample_sources`（指定样本改读独立包）；默认行为不变，对 v1 输入逐位复现 v1。新产物 `reports/{baseline,brd4_statistics}_20261101_v2/`，v1 保留。

## 2026-11-01 · BRD4-5 改用阿普唑仑重新准备（冒烟）

- 包内 `scripts/prepare_brd4_5_alprazolam.py`：上游 BRD4-5 配体删 H8/H9、NAL=CAO 双键，InChIKey 须为阿普唑仑；AM1-BCC/GAFF + parmchk2 + tleap；ParmEd 只替换配体，其余与上游 prmtop 逐项核对并记录。输出到独立包，不改主包 `manifests/runs.csv`。
- `scripts/score_smallmol_start.py` 加 `--model-id`、`--inputs`（默认不变）。
- 新增包内 `brd4_5_smoke.sh`、`brd4_5_full.sh/.pbs`。183 冒烟通过（`docs/BRD4_5_REPREP_PLAN_20261101.md` §8）；正式作业未提交。

## 2026-11-01 · P2 BRD4 标签审计与分组统计

- 读取 6 篇原文（PMC/Europe PMC，存 `sources/brd4/papers/`），写 `reports/brd4_statistics_20261101/label_audit.csv`；10 条配体的 InChIKey 与 PubChem/CCD 逐条比对，发现 BRD4-5 上游输入为 5,6-二氢阿普唑仑。
- 包内 `scripts/brd4_statistics.py`：按审计后写定的 `protocol.json` 对默认与 h0.4 网格分别算分组 ρ/τ-b/成对/精确 p、副本＋连续时间块重采样与候选重采样（复用 `rank_hou_sampling.py` 的函数），并作图。结果 RESULTS §21.3。

## 2026-11-01 · P1 基线：输入清单与最小化起点评分（B1）

- 包内 `scripts/build_baseline_inventory.py`：只读生成 `reports/baseline_20261101/inventory.csv`（17 条候选的身份、来源、R/L 范围、标签、起点、键级来源、已有 B2 分数和缺口）。
- 新增 `scripts/score_smallmol_start.py`：用冻结默认 `OnlineMMPBSA` 给 `setup/solute_reference.npz`（无约束最小化后的溶质）打一帧分，分析器构造与 `score_smallmol_replica.py` 相同。冻结代码拒收单帧 pilot（在线帧会超出单帧的涨落），这里只评这一帧，所以不传 pilot、传 `fluctuation_allowance=0`，网格按该结构定尺、边界余量仍为默认 `margin_min`。未改 `jaxpbsa/`。
- 新增包内 `start_full.sh/.pbs`、`scripts/collect_start_scores.py`。183 冒烟后在 yayoi43 跑完 17/17（RESULTS §21.1）。
- 补取 BRD4 上游配体 `sources/brd4/{sdf,mol2}/`（commit 5233e0a，`sources/retrieval_brd4_ligands.json`）；包内 `scripts/check_brd4_ligand_files.py` 核对与 prmtop 一致（`reports/baseline_20261101/brd4_ligand_file_check.csv`）。inventory 重新生成。
- 包内 `scripts/compare_baselines.py` 与 `reports/baseline_20261101/protocol.json`：按统计前写定的协议算 B0 描述符（RDKit，可旋转键 Strict）和 B0/B1/B2 对实验的 ρ、τ-b、成对次序、精确 p；Hou B2 自检复现 `ranking.json`。结果 RESULTS §21.2。

## 2026-11-01 · 后续执行计划落盘

- 新增 `docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md` 和根目录 `TODO.md`：将基线对照、BRD4 分组统计、可复现验收包/论文收口及条件性的独立系列验证写成执行顺序、输入清单、交付和完成条件。
- 同步项目规则中的计划入口、BRD4/Hou 交接及论文框架 §8。保留 Hou 既有预算和评分来源、BRD4 两网格的展示与未决边界；未修改方法、代码或历史产物，未执行新计算。

## 2026-10-31 · 在线 MM/PBSA 方法论文框架

- 新增 `docs/ONLINE_MMPBSA_PAPER_OUTLINE.md`：按用户同意的定位建立主张/证据表、摘要与正文骨架、五张主图和 SI 计划、收口顺序与未决事项。区分数值、在线成本和应用排序证据，保留 Hou 网格未见收敛、a6 错序及 39 在线 + 3 离线恢复的来源边界。
- 同步 BRD4/Hou 执行入口及历史多肽框架的写作指向。未修改代码、默认参数或历史结果，未运行新的分子计算。

## 2026-10-30 · Hou 固定采样预算收尾审计与离线恢复入口

- 后续：已存在的 `scores/window10_default/` 恢复产物通过严格输入/模型/精度/覆盖复核（42/42 副本、42,000 帧可用）；新增 `scripts/rank_hou_sampling.py` 按设计 §9 计算同系列、全七条与补充组的 Spearman/τ-b/21 对次序，穷举小样本精确 p，分开做副本＋时间块和候选组成重采样。报告采用冻结默认网格，h0.4 保留为数值敏感性。结果、图、来源和限制在 `reports/hou_ranking_20261030/`、RESULTS §20.10；排序与完整性回归共 20 passed。未新增 MD/PBSA，未修改原在线标记。
- 新增 stdlib-only `scripts/summarize_hou_sampling.py`：严格核对 42 个副本的在线提交覆盖与 strip 时间映射，输出逐副本/时间块/固定 cohort/成对差异和 97 个越界帧；有坏帧或缺副本时不生成完整窗口均值。历史 3 × 5 ns 基线使用原始在线值，不被恢复结果覆盖。
- 新增 `hou_recover_margin_20261030.sh`：默认 dry-run；对 3 个越界副本各重评完整 10 ns，输出到新目录。后续汇总只接纳完整、有效、模型精度及输入绑定一致的 whole-replica 离线恢复，单列与原有效/无效帧的配对差异。不改默认模型、原 MD 或在线 CSV。GPU 恢复本次尚未启动。
- 新增保护回归 13 项通过；实测收尾和待运行计划见 `RESULTS.md` §20.9 与 `docs/HOU_SAMPLING_RECOVERY_HANDOFF_20261030.md`。

## 2026-10-29 · 在线覆盖核对的时间容差

- `scripts/smallmol_records.online_rows`：核对在线行的时间与 step × dt 时，容差从 `abs_tol=1e-6` 改为 `0.25 * dt`。OpenMM 的时间是每步累加 dt，Hou 延长到 10 ns 时在 9.91 ns 偏差 1.0003e-6 ps，导致 41 个已跑完的副本收尾失败（`docs/HOU_BOUNDED_SAMPLING_PLAN_20261028.md` §7.3）。只要偏差小于 dt/2，仍能唯一对应一个 step。`scored_frames` / `sensitivity_frames` 比较的是 strip 写出的同一个四舍五入值，没有改。
- 10-28 的脚本改动：`online_overhead.py` 加 `--hou-run-id/--grid/--cache-dir`；`score_smallmol_replica.py` 加 `--pilot npt`；`grid_sensitivity_smallmol.py` 加 `h0.3` 配置（RESULTS §20.7、§20.8）。

## 2026-10-26 · Hou 同帧 Amber/网格对照（D3）

- `amber_reference_smallmol.py`、`grid_sensitivity_smallmol.py` 加 `--ours online`（Hou 只有在线分数）；`smallmol_records.online_rows_by_frame`；包内 `summarize_reference_q3.py` 可选 `hou`。新增 `hou_ref.sh/.pbs`。yayoi10 跑完 63 帧，结果见 RESULTS §20.4。
- 新增 `hou_h04.sh/.pbs`：Hou 21 副本 × 500 帧按 `--grid h0.4` 重评，yayoi10 约 40 min、0 FAIL；包内 `scripts/compare_hou_grids.py`。修正 −1.64…+3.48，次序 Spearman 0.893（RESULTS §20.5）。

## 2026-10-25 · BRD4 15 000 帧 h0.4 重评完成

- yayoi04，约 1 h，0 FAIL / 0 标记；q3 90 帧与 h0.4 参照逐位一致。样本次序与默认网格相同（Spearman 1.000），修正量 −0.35…+2.30（RESULTS §19.8）。新增包内 `scripts/compare_brd4_grids.py`。

## 2026-10-25 · 评分脚本加 `--grid h0.4`（BRD4 重评用）

- `score_smallmol_replica.py --grid {default,h0.4}`：h0.4 = C/R 0.4 / 配体 0.2，输出 `scores/h0.4/`；default 不变。汇总脚本可指定 grid。新增 `brd4_h04.sh/.pbs`。183 冒烟：帧 124 与已有 h0.4 参照的 ΔG_MMPBSA 差 1.2e-4。计划 `docs/BRD4_H04_RESCORE_PLAN_20261025.md`。

## 2026-10-25 · BRD4 默认网格偏差诊断 D0、D1 完成

- D1（183，11.5 min）：相位平均后偏差仍在；C−R 偏差最大的一项是配体电荷在 C/R 0.5 网格上的自项 LL_C（BRD4-3/6/8 为 +2.1～2.6），受体项随样本变号（RESULTS §19.7）。

- D0（只读）：BRD4-3/6 的偏差全在 C−R，9/9 帧同号；3 个随机转动的平均不能把它消掉，所以排除"相位冻结"，是真离散误差（RESULTS §19.6）。
- 新增 `scripts/grid_diag_smallmol.py`（D1）：直接调用 TripletSolver，把 C−R 拆成 RR_C−G_R、LL_C、RL_C 三项，另做 8 个亚格点平移和 tol 1e-6 对照。183 冒烟通过，和已有同帧值相差 ≤0.0015；正式运行交 PBS（`experiments/…/grid_diag.pbs`）。更正 RESULTS §19.5 和 `grid_sensitivity_smallmol.py` 头注释里"无法施加亚格点平移"的说法。

## 2026-10-25 · 评分续跑不再按 DCD 内容摘要绑定

- `score_smallmol_replica.py`、`grid_sensitivity_smallmol.py` 的 protocol 升到 schema 2：不再对 `solute_dry.dcd`/`frame_times.csv` 算 SHA256（strip 每次调用都重写 DCD，文件头里带创建时间，重新 qsub 或 MD 延长后评分续跑会被误拒）；改为按路径和原子数绑定，逐帧 step/time 照旧核对，网格敏感性额外记录参考帧的 step/time。详见工程修复交接 §7。

## 2026-10-24 · 工程完整性、输入绑定与参考时间修复

- 第二轮：setup 缓存与当前来源/R-L 选择核对；在线生产绑定 NPT pilot、实际分析默认参数与精度；完整 CSV schema/提交范围/边界阈值检查；拒绝非整步时间与不足两帧的 pilot；修复离线延长的生产 step/time 恢复。完整工程回归 131 passed / 2 deselected。详见交接 §6。

- 修复在线续跑在 MD 完成、尾部 worker 失败后错误标记完成；保存失败 session，按已提交 step 检查评分覆盖。
- 新评分目录在首帧前记录输入/协议绑定；检查重复、残缺和时间错配的 CSV。历史未绑定评分保留，新评分使用新 `--out`。
- 修复 Amber 对照 `used[j-1]` 时间错位；备份并修正 q3 30 份对照表的 90 帧时间标签，逐字段确认能量不变。
- reporter/预热增加非有限能量检查；warm trajectory 不再传递失败势；拒绝不支持的非键表示、跨 R/L 例外与错误索引。
- 网格敏感性逐帧续跑；Hou 汇总严格检查覆盖和有效性，修复时间块边界重复纳入。Hou 样本分数/排序不变。
- 验证、兼容性和待节点验收事项见 [工程修复交接](docs/ENGINEERING_REPAIR_HANDOFF_20261024.md)。

## 2026-10-23 · Hou α-thrombin 准备与 MD 中在线评分（不改方法与默认参数）

- `scripts/run_smallmol_md.py run --online`：生产段挂 `AsyncPBSAReporter`（同卡 worker，MPS 线程上限 `--mps-percentage`，默认 30），每个保存帧评一次；NPT 阶段存的 1 ns 溶质帧作为 `pilot_coords_A`；每个进程写自己的 `online/pbsa_from_chunk_NNNN.csv`，续跑后 step/time 接得上。不加 `--online` 时行为不变。
- 数据包 `scripts/prepare_hou_thrombin.py`：Hou a1–a7 → ff14SB/GAFF2/AM1-BCC/TIP3P 的 prmtop（CCD 键级与立体化学、pH 7.4 配体态、reduce 定 His、ACE/NME、SSBOND、0.15 M NaCl），清单 `manifests/hou_inputs.csv`；`make_runs_csv.py` 追加 Hou 21 行（BRD4 行不变）。
- 作业 `hou_online.pbs` / `hou_online.sh`。准备结果、冒烟与耗时见 `docs/BRD4_HOU_EXECUTION_HANDOFF.md` §7.12。

## 2026-10-19 – 10-21 · 小分子基准的入口与同帧对照脚本（不改方法与默认参数）

- `scripts/run_smallmol_md.py`、`scripts/score_smallmol_replica.py`：从 Amber prmtop/crds 跑 setup、MD、strip 和逐帧评分（时间轴 (k+1)×5000 步，窗口 (start, end]）。
- `scripts/amber_reference_smallmol.py`：同帧 Amber MM/PBSA + MM/GBSA（ante-MMPBSA.py 拆拓扑，RADII 改写为我们的 mbondi2，-eo 逐帧输出）。
- `scripts/grid_sensitivity_smallmol.py`：在固定帧上测 PB 网格加密（h0.4/0.2、h0.35/0.167）和刚体转动的影响。
- 数据包 `experiments/2026-10-17_small_molecule_benchmark/scripts/`：`make_runs_csv.py`、`summarize_brd4_round1.py`、`summarize_reference_q3.py`、`build_thrombin_loop_modeller.py`。
- 结果见 RESULTS.md §19，执行记录见 `docs/BRD4_HOU_EXECUTION_HANDOFF.md` §7。

## 2026-10-18 · OnlineMMPBSA 溶质净电荷整数检查放宽到 1e-2 e（BRD4 体系测试发现的 bug）

- `jaxpbsa/online.py`：`OnlineMMPBSA.__init__` 的「净电荷非整数」检查容差从 1e-3 改为 1e-2 e。公开 BRD4（MobleyLab benchmarksets）AM1-BCC 配体自带舍入残差，最大 0.0050 e，1e-3 把 BRD4-1/5/6/7/10 的正常输入全部拦下（首轮 15/30 副本评分失败）。1e-2 是实测最大残差的 2 倍；切片错位的典型偏差（多切进离子 ±1、水 O −0.834、漏掉末端配体原子 0.05–0.42 e）仍远大于它。只影响是否接受输入，不改变任何能量值；已评分的副本无需重算。依据与结果见 `docs/BRD4_HOU_EXECUTION_HANDOFF.md` §7.4、§7.8。2026-10-19 在 yayoi04 补评：首轮被拦的 15 个副本全部通过，500/500 帧有效。

## 2026-10-06 · 投稿审查（docs/JCIM_SUBMISSION_REVIEW.md §5）的两处修复

- `scripts/validate_mmpbsa.py`：逐帧检查 converged / margin_A / margin_lig_A（判据同在线
  reporter），任一帧不可用直接失败、不进均值；我们这边先算，坏帧在跑 MMPBSA.py 之前就报。
  测试 `tests/test_validate_mmpbsa.py`（无需 GPU / Amber）。
- `pb/energy.py`：`nonlinear=True` 且 `temperature_K != 298.15` 时报错。源项与能量单位固定在
  298.15 K，sinh(u) 在其他温度下会静默给错数；线性 PB 不受影响。

## 2026-10-03 · 单 GPU 在线后台调度

- 新增 `AsyncPBSAReporter.from_system`，分析器只在常驻子进程构造/预热；OpenMM
  回调复制并提交坐标后返回，worker 计算、检查和及时写 CSV。
- MPS 线程限额/低优先级与 CPU 线程设置仅作用于 worker，启动前检查 MPS 服务。
  显式队列上限、积压策略、失败传播、结果延迟统计和排空退出。
- `online_overhead.py --mode both` 对照同步/异步，报告 MD 与含排空的端到端吞吐。
  调度与 OpenMM reporter 共存测试无需 CUDA；真实分析器数值对照使用运行时 JAX
  后端。
- 2080 Ti 实测（RESULTS §18.22）：不限额异步 +1.5%；MPS 30% 限额下 5000 步间隔
  MD overhead 15% → ~10%，最大延迟 0.68 s。单次 5 万步。
- 4090 复测（RESULTS §18.23）：20 万步、MPS 20/30/40/50% 四档，5000 步间隔
  11.8% → 3.9%（30%）；20–40% 持平、50% 变差。示例与推荐值统一为 30%。
- 父子进程计时统一使用 POSIX `CLOCK_MONOTONIC`；遗漏关闭且存在 pending 帧时
  发出 `ResourceWarning` 并尝试正常排空。context manager 退出默认 60 秒超时，可通过
  `close_timeout` 配置，超时后可重试关闭，不强杀 CUDA/MPS client。
- 带 MPS 参数的基准在计时前核对 MD/worker PID 是否出现在 MPS client 列表；
  worker 保留父进程 `sys.path`，`scripts/` 中应避免模块同名遮蔽。

---

## 1. 正确性修复（全部是静默失败：不报错，只给看起来合理的错数字）

| # | 问题 | 位置 | 怎么暴露的 |
|---|---|---|---|
| 1 | PB 源项漏 `BJERRUM_VAC = 560.75` 因子 → G_PB 小 560 倍 | `pb/charges.py` | T1 Born 解析值 |
| 2 | `κ̄²` 与 `κ²` 混用 → 体相屏蔽长度差 √78.5 ≈ 8.9 倍 | `constants.py`, `pb/operator.py` | T2 Debye 拟合 |
| 3 | Dirichlet 边界写进算子 → CG 收到**仿射**算子，前提破产 | `pb/solver.py` | T1 给出 −512 而非 −82 |
| 4 | 雅可比对角元在 **8 个角点恒为 0** → `0/0=NaN` → `NaN*0=NaN` 污染内积 → `cond` 恒 False → **求解器一次迭代都不跑，静默返回 u0**。单帧路径靠 XLA 把 `x*0` 折成 `0` 侥幸没暴露 | `pb/solver.py` | `test_batch_invariance` |
| 5 | **停止判据用了预处理器范数** `sqrt(rᵀM⁻¹r)/‖b‖` 而非真残差 `‖r‖/‖b‖` → 换预处理器就换了判据，「相同 tol」不是相同精度 | `pb/solver.py` | code review |
| 6 | 真残差只**报告**不**执行**：超标也直接返回 | `pb/solver.py` | code review |
| 7 | 零源项 `0/0` → `NaN > tol` 为 False → 假收敛 | `pb/solver.py` | code review |
| 8 | `pcg_solve` 加 `precond` 形参后**位置参数错位**：`tol` 进了 `precond` | `pb/energy.py` | code review |
| 9 | `TER` 写在残基循环里 → 每个残基被当成独立链，肽键全断 | `scripts/prep_s4.py` | 代码复查 |
| 10 | NeRF 第一项符号错 → 键角做成 `180°−angle` | `scripts/prep_s4.py` | `_measure`/`_nerf` 往返自检 |
| 11 | 残基名与实际质子化态不符：`addHydrogens` 按自己的 pH 逻辑选态并**无视输入名字**，HIS90/96 名为 HIE 实为 HID | `scripts/prep_s4.py` | tleap FATAL |
| 12 | **制备管道不幂等**：三层随机性叠加，重跑一次 1835 原子全动（RMSD 0.78 Å） | `scripts/prep_s4.py` | 重跑后能量跳变 54 kcal/mol |
| 13 | **`warm=False` 并没有冷启动**：carry 被当成 `u_prev` 传了进去，等于"每帧从同一个给定初值出发"。作为 warm start 的对照组，它含了调用方的信息，**比较不公平** | `pb/energy.py` | 逐位断言的跨块测试 |

> 第 13 条的额外教训：**代码和文档不符时，先信代码。** 我在 docstring 里写了
> 「`warm=False` 时 initial_state 被忽略」，但实现并没有忽略它。
> 是测试（而不是复读文档）把这个差异暴露出来的 —— 所以断言要写成**逐位相等**，
> `rtol=1e-6` 这种松断言会让 2e-6 的差异溜过去。

### dtype 静默提升链（不报错，只是悄悄变回 fp64 且变慢一倍）

- `jaxpbsa/__init__.py` **强制开 x64**，覆盖调用方设置 → **两轮 fp32 测量全是 fp64 跑的**
- `grid.origin` 存的是 **np.float64 标量**（强类型，会提升 fp32；Python float 是弱类型不会）
- `jnp.zeros(shape)` / `jnp.where(m,1.,80.)` / `fori_loop` 初值 **不带 dtype 时在 x64 下默认 float64**
- `while_loop` carry 初值 dtype 不一致
- `.astype(fp64).sum()` **物化整个 float64 副本**（百万元素）→ **fp32 比 fp64 还慢**；应写 `sum(dtype=)`
- 静态参数**每帧重传设备**，占 MM 总时间一半

守卫：`test_dtype_is_not_silently_promoted`。

---

## 2. 性能改动

| 改动 | 收益 | 位置 |
|---|---|---|
| DST 直解参考方程（奇延拓 + `rfft`） | **2.14×** | `pb/dst.py`（新） |
| MG **作 CG 预处理器**（独立求解器在 ε 跳变 1:80 下发散） | **2.27×** @ h=0.5 | `pb/multigrid.py`（新） |
| fp32 数组 + **fp64 归约** | **2.0–2.96×** | 全局 |
| 矩形网格（按各轴包围盒） | **1.41×**，节点 −30.4% | `pb/grid.py` |
| 形态学 `dynamic_slice` → **静态 `lax.slice`**（traced 起点阻止 XLA 融合） | 形态学 **6.0×**，逐位相同 | `pb/surface.py` |
| 参考边界原子分块 | kernel **5.28×**，端到端 1.07% | `pb/operator.py` |
| Coulomb/LJ 共用一次距离计算 + 参数预置设备 | MM **2.1×** | `mm/` |
| tol 1e-7 → 1e-5（两档 G_PB 只差 0.002） | 1.8× | `pb/energy.py` |

**合计 1800 → 76.8 ms/帧（23.4×，单 species PB）。**

### 试过但无效 / 不可行（已写进代码注释，免得重试）

- pad 提到循环外 —— **零收益**，XLA 本来就 CSE 掉了
- 形态学分组展开（chunk 8–128）—— **零收益**，静态起点后 XLA 本就跨整条链融合
- 球分解 `ball(R) → m×ball(R/m)` —— **不可行**，格点上 `ball(a)⊕ball(b) ≠ ball(a+b)`（h=0.5/R=2.0 拆两次得 185 点 vs 目标 257）
- MG 最粗层轮数 —— **无差别**，4 与 50 都是 11 次迭代
- batching —— h=0.5 下 **0.63×（负收益）**；仅粗网格 + B≈2 有效

---

## 3. 新增能力

- **C/R/L 三体求解** `solve.triplet()`：ΔG_PB = G_C − G_R − G_L。参考解 **3 次降 2 次**
  （`u_ref_C = u_ref_R + u_ref_L`，**组合的是势不是能量** —— 参考能量含 R–L 交叉项）。
  R/L **补齐到 complex 的原子数**（屏蔽原子挪到盒外），三者**共用一份编译**。
- **轨迹接口** `solve.trajectory(..., initial_state=)` → `(每帧能量, final_state)`。
  势场只作 `lax.scan` 的 **carry**，不进输出（否则 T=10000 单 species 要 268 GiB）。
  **失败帧的势不传给下一帧**。
- **规范产物**：`prep_s4.py` 是转换器，输出 `S4_complex.cif` + `S4_system.xml` + 双 sha256；
  `load_canonical()` 加载并校验，下游**不再 import ForceField**。
- **跨设备 benchmark** `scripts/benchmark.py`：每配置独立子进程、关预分配、编译/稳态分离、
  **拒绝未收敛与非有限结果**；报告强制分「换卡不变」与「本机专属」两块。
- **分阶段 profile** `scripts/profile_stages.py`：与生产路径同边界、同 dtype、同能量公式。

---

## 3.5 跨设备验证（RTX 2080 Ti + RTX 5080）

**换卡不变量逐位一致** —— G_PB（complex/receptor/ligand/ΔG_PB）、迭代数、真残差
在两张卡上完全相同。两台机器从同一个 `S4_complex.cif`（sha256 `78d8db8e…`）出发。

由此定下两条：

- **`precond` 默认改为 `"auto"`**（阈值 `MG_NODE_THRESHOLD = 1.5 M` 节点）。
  MG 的交叉点在 **0.9–2.1 M 节点**之间，**两卡一致**；但幅度相反 ——
  粗网格上卡越快 MG 越吃亏（0.45× on 5080 vs 0.87× on 2080 Ti，粗层是启动延迟主导），
  细网格上则相反（2.81× vs 1.74×）。原先硬编码 `"mg"` 在 h=1.0 上**让程序慢一倍多**。
- **batching 的机制解释撤回**。先前说「batch 维在最前打散 stencil 合并访存」，
  它预测恒定惩罚（平的），符合 2080 Ti（一步跌到 0.62 后完全平）但**不符合 5080**
  （逐步下滑 0.87→0.73→0.70）。能确定的只有：两卡均为负收益、**显存都不是原因**
  （5080 的 B=8 只用 2.47 GiB / 16 GB）。

---

## 4. 测量方法上的教训

每条都曾让我得出**错误的物理结论**：

1. **同进程顺序跑多个配置** → 编译产物与显存单调累积 → batch 断崖数据作废（真因是 kernel 效率，不是显存）
2. **并发跑两个 GPU 进程** → JAX 预分配 75% → 伪 OOM，甚至 `bfc_allocator` 崩溃
3. **没分离编译时间** → 「scan 比单帧慢 7.6×」全是重复 tracing（实测 scan 与 Python 循环差 0.8%）
4. **测量运行期间改源码** → 前三行立方网格、最后一行矩形网格，报出假的 1.50×
5. **纸面流量反推优化空间** → 「算子融合还有 2–3×」无依据，已撤回
6. **判据不一致就比加速比** → 等于把「精度变松了」当成「计算变快了」
7. **h 选在 ≥1.5** → `ball_offsets(1.4,h)` 退化为单点，SES 塌回 vdW，G_PB 来回摆 600 kcal/mol
8. **拿绝对值的相对误差估差值的误差棒** → 会得到 28 kcal/mol 的假误差棒，实际 1.1（**差 25 倍**）
9. **跨数量级的单位换算没落到纸面** → `86400/1e8 = 8.64e-4 s` 被写成 `0.86 μs`
   （实为 864 μs），**1005 倍纯单位错误**，让 online PB 的可行性结论差两个数量级。
   ns/day → 步/天 → 秒/步 → μs/步 是四级跳，必须逐步写出单位。
   **这类错误不触发直觉警报** —— 0.86 μs/步听起来"像快机器"，864 μs/步听起来"太慢"，
   **错的那个更符合预期**。（归因记录：输入假设 200 ns/day 本身偏保守，实测 1397，
   所以错的不是估计值而是换算。）

---

## 5. 被实测推翻的预判

| 原先的说法 | 实测 |
|---|---|
| surface 是真瓶颈 | 优化前 29.4%、优化后 **8.1%**；solve 占 **82.4%** |
| Jacobi-PCG 需 500–1500 次迭代 | **193 次**，标度 `(L/h)^0.64` |
| MG 可达 10–50× | **1.7–2.8×** @ h=0.5（每个 V-cycle ≈ 8 个 Jacobi 迭代的成本），且**粗网格上是负收益**（0.45–0.87×）→ 改 `precond="auto"` |
| warm start 是核心 novelty | **1.08×** —— 与 MG 此消彼长，MG 把迭代压到 11 后只能再省 1 次 |
| 「Batch first」 | h=0.5 下 **0.63×（负收益）** |
| 算子融合还有 2–3× | 依据不成立，已撤回 |


---

## 6. 下一步（按价值）

**ΔG_SA**（补完 ΔG_MM/PBSA 的最后一项）> **外部对拍 T6**（用 Amber `pbsa`，**不能用 GB**）
> **M0 轨迹** > **溶剂求解器访存效率**（占 82–85%，但必须按 profile 判断，
不能再用纸面流量反推）。

展开见 [RESULTS.md §9](./RESULTS.md)；明确不做/已证伪的方向见 §10。

---

## SA stage 2：JAX Shrake–Rupley（`sa/jax_sr.py`）

SA 现在**只有这一个实现**。zsasa 那个外部 CLI **不是后端也不是依赖** —— 它的唯一职责
是给 JAX 版做性能/结果对拍，所以从 `jaxpbsa/sa/zsasa.py` 挪到 `tests/zsasa_ref.py`，
和 APBS 同一个待遇（「外部进程调用，不进包」）。

`jaxpbsa.sa.sasa()` / `delta_g_sa()` 的 `backend=` 形参一并删掉：**一个实现的分派
就是多余的抽象**，留着只会让人以为 zsasa 是个可选运行时路径。

**不显式构造采样点。** 字面照抄 SR 要建 `[N,K,3]` 的 `p_ik = x_i + R_i·u_k` 再逐个测距；
代进埋藏判据展开后 `|p_ik − x_j|² = d_ij² + R_i² + 2R_i·u_k·(x_i − x_j)`，
判据变成一次 `u[P,3] @ disp[K,3]ᵀ` 的矩阵乘。顺带白捡：对 u_k 取极值得
`d < R_i + R_j` 才可能遮挡，**远邻居 / padding(d²=∞) / 自身项(disp=0) 全部自动失效，
不用写掩码**。

| 项 | 值 |
|---|---|
| vs zsasa 总 SASA（S4, P=960 / 4000） | rel **4.6e-5 / 5.8e-5** |
| vs 暴力法（显式建 p_ik 测距） | rel < 1e-6 = 逐点判定完全相同 |
| 单 species（2080 Ti, B=16） | 41 → **7.2 ms/帧** |
| 三 species ΔG_SA（B=16） | 131 → **13.7 ms/帧**，占每帧 6%（PB 222） |

**加了一道硬闸**：K 近邻是有损的，`K ≥ maxᵢ|{j : d_ij < R_i + R_max}|` 才保证不丢遮挡。
不满足**直接报错并报出该填多少**（`需要 ≥ 134 / 重试: k_neighbors=134`），一次重试必中。

自动探测试过两条，**都不安全**：扫 CIF 给 131（真空最小化态，不是 MD 系综构象），
MD 帧 0 给 126，而 10 ns 轨迹实际跨 **123–136** —— 单帧探不出系综上界，且 k 是
static argnum，逐块重探会反复触发重编译。默认 192（40% 余量，堆积密度限的量，
换蛋白不变）。

没有闸就是第 1 节那种「不报错、只给看起来合理的错数字」：误差恒为正且单调
（丢遮挡 ⇒ 暴露变多），k=32 时 +1.4%，没有任何提示。

## 探针半径的 source of truth

`probe_radius=1.4` 原来有 **3 份独立字面量**（`pb/energy.py` 的 `PBParams`、
`sa/jax_sr.py`、`sa/zsasa.py`）。两边用不同的 r_p **不会报错**，只会悄悄给出偏掉的
ΔG_MM/PBSA —— PB 的分子表面和 SA 的可及表面对应不同的溶剂，相加没有物理意义。

现在唯一来源是 `constants.PROBE_RADIUS`。
`tests/test_constants.py::test_pb_and_sa_share_one_probe_radius` 既比默认值、
也扫源码里的裸字面量。

顺带把边界写进 DESIGN.md §3.10：**SA 与 PB 只共享物理定义**（r_i 来自
`openmm_io.assign_radii`，r_p 来自 `constants.PROBE_RADIUS`），
**PB 的 voxel mask / atom→grid 映射 / erosion 表示 / 网格间距 h 一律不外泄**。
两者的几何体确实同一个（`dilate(vdw, ball(R_probe))` 就是 SR 采样的那张球面所围的体），
但从布尔占据格点数边界面取面积是 Cauchy 阶梯 —— 对球精确算是 **6πR² vs 真值 4πR²，
系统性 +50%**，且会把 SA 的精度绑死在 PB 的 h 上。不合。


---

## PB 求解器：`_smooth` 展开（−11%）与一次几乎全错的诊断

**改动只有两处**，完整归因见 [RESULTS §14](./RESULTS.md)：

| | |
|---|---|
| `pb/multigrid.py` `_smooth` | `fori_loop` → Python 展开。`sweeps` 是 host 侧常量，循环次数编译期已知，XLA 的循环边界挡住了跨迭代融合 |
| `pb/energy.py` `PBParams.mg_min_n` | 新参数，接到 `build_levels(min_n=)`，控制 MG 粗化到哪一层为止。**默认仍是 7** |

```
默认 222.3  ->  展开 198.9  ->  + mg_min_n=41 截断 176.8 ms/triplet   (−20.5%)
GPU 工作量  219.4 -> 176.4 ms/triplet,  GPU 操作数 11,987 -> 3,610
ΔG_PB / 迭代数 / relres 全部不变
```

**同一个坑踩了两次**：`surface.py` 的 `dilate` 早就记过「静态起点后 XLA 跨链融合，
28.95 → 4.86 ms」。规律补进 DESIGN §3.5：**trip count 是 host 常量时不要用 `fori_loop`。**

### 测量陷阱：`nsys` 默认不展开 CUDA Graph

`nsys profile -t cuda` 默认 `--cuda-graph-trace=graph`，**graph 里的 kernel 只记一次
graph 启动**。kernel 时间被严重低估，低估的部分在时间轴上表现为「空隙」。据此得出过
一整串结论，**全部作废**：

| 作废的结论 | 真实情况 |
|---|---|
| ~~GPU 忙 38%、闲 62%~~ | **忙 96%**（改前改后都是）|
| ~~每个 GPU 操作 5.6 µs 调度成本~~ | 中位空隙 0.5 µs |
| ~~空闲集中在 161 个 ms 级的洞里~~ | 那些「洞」是没被采到的 graph 节点 |
| ~~MG 有 2.6× 空转，粗层融合是首选~~ | 有约 20% 可回收，但机制是融合减字节，不是填空隙 |

修法：`--cuda-graph-trace=node`，且**改前改后必须同一口径**。

> 更该记住的是**判断顺序错了**：`nvidia-smi` 报 SM 99%，我用一个错误的 trace 去推翻它。
> 两个信号打架时，先怀疑新工具的采集口径。
> （另：`nvidia-smi` 的 `utilization.gpu/memory` 是「有活动的时间占比」，
> 不是占用率也不是带宽占比 —— 可用于否证，不能用于确证。）

### 顺带修正一条旧结论

RESULTS §10 的「MG 最粗层轮数 4 与 50 无差别」**只在 min_n=7（终层 252 点）下成立**。
终层变大就完全不同：41×41×49 上 sweeps=4 让迭代数从 29 涨到 **69**。
`coarse_sweeps` 必须跟 `mg_min_n` 一起调。

---

## 误差定位：`δΔG_PB ≈ −δG_L`（可能影响 headline 数字）

**没有改代码，改的是对现有数字的判断。** 完整数据见 [RESULTS §15](./RESULTS.md)。

§0.17 早就测到 `δG_C ≈ δG_R`（h=0.75→0.5 抵消到 0.01）。把它写进差分就得到
`δΔG_PB ≈ −δG_L`：**C 与 R 共 1655/1835 个原子、共一张网格，误差是同一个东西；
孤立配体无可抵消。** 所以 ΔG_PB 的全部离散误差 = 配体那一次求解的误差。

顺着这条查下去：

```
G_C − G_R    h=0.75  265.478  ->  h=0.5  265.470     只动 0.008，早已收敛
G_L (180 原子)  h=0.5 −610.3 -> 0.375 −592.2 -> 0.25 −586.3   仍在走, 外推 ≈ −583
                                                    h=0.5 上差约 27 kcal/mol
```

组装：`(G_C−G_R)|0.75 − G_L|0.25 = 851.8`，当前全 h=0.5 共用网格给的是 **885.4**，
Amber pbsa 是 **842.9**。**与 Amber 那 5.0% 的差异，很可能主要是我们的配体欠解。**
ΔG_MM/PBSA 会从 −67.5 移到 −101 量级 —— 所以 README 的表已加 **under review** 标注。

未定论：只在 S4 一帧上测过；紧盒把 origin 对齐与 padding 混在一起没拆；
`G_L|0.25` 本身也未收敛。**没有改任何默认值。**

### 新指标：grid-origin 敏感性

同一构象、同一 h、同一 shape，**只挪 origin**（物理零变化，任何非零都是离散化误差）：
ΔG_PB 峰峰值 **8.39 kcal/mol** —— 比整个 h=0.75→0.5 全局加密的效果（5.49）还大。
轨迹上这是帧间噪声：网格固定、分子漂移，挪网格 ≡ 挪分子。

### 证伪一条设计理由

DESIGN §3.5 给 `swin=0.5` 的理由是「压帧间 grid 离散噪声」。实测
**swin=0.5 下 8.39，swin=0 下 7.22 —— 没压住，还略差**。它确实把答案整体挪了 16.2，
但那是**改变了表面定义**，不是降噪。`swin` 从此只应被当作一个必须在论文里写明的
表面定义参数。真正针对摆放噪声的是分数体积 ε。


---

## 非对称网格：又快又准（实测，尚未设为默认）

顺着 `δΔG_PB ≈ −δG_L` 往下做完了。**C/R 用 h=0.75 共用盒，配体用紧盒 h=0.25：**

| 方案 | 节点合计 | ms/帧 | ΔG_PB | 距 Amber pbsa 842.9 |
|---|---|---|---|---|
| 现状：三者共用 h=0.5 | 15.0 M | 197.1 | 885.405 | 42.5（5.0%）|
| **特化，配体 padding=8** | **8.3 M** | **162.0** | **851.695** | **8.8（1.0%）** |

**快 18%，同时把与 Amber 的差距从 5.0% 收到 1.0%。** 不是取舍 —— 现在的代码把算力
花在**会抵消的 C/R** 上，而配体欠解。三次编译在热进程里占一条 10k 帧轨迹的 **1.3%**。

### 顺带：padding 规则超配 2–4×

DESIGN §2 的 `padding = max(20 Å, 3κ⁻¹) = 23.6 Å`。实测 complex 网格 20 → 40 Å
ΔG_PB 只动 **0.003**；配体盒 padding 20 → 12 差 **0.029**、→ 8 差 **0.090**，
而节点数 18.6 M → 7.0 M → 4.0 M。真实需求约 1–1.5κ⁻¹。

### 又一条自己写完就被自己证伪的推断

我曾写「多中心边界 → padding 可收 → 配体网格减半」。**错了** —— 单中心 DH 边界在当前
padding 下早已收敛（0.003），没有可解锁的东西。多中心边界的优先级因此从第 1 降到第 4。

新排序：**① 非对称网格 → ② 分数体积 ε → ③ 曲面解析基准 → ④ 多中心边界 → ⑤ NLPB**。


---

## 在线 MM/PBSA（`jaxpbsa/online.py`，新增）

plan §21–§23 落地顺序的第 2 步。**同步原型**：MD 循环经 `PBSAReporter` 逐帧调用，
analysis 与 MD 同卡分时。async worker / 双 GPU 不做，等 `scripts/online_overhead.py`
的实测数说话。完整实现说明见 [ONLINE_PLAN.md](./ONLINE_PLAN.md)。

`OnlineMMPBSA` 把 MM cross / PB triplet / SA 合成一次调用（此前只在
`scripts/validate_mmpbsa.py` 里手工拼过），网格与全部参数在 `__init__` 冻结，
每帧只动坐标。

### 归位从包围盒中点改成质心 —— 又一条写完就被自己证伪的推断

ONLINE_PLAN 的原案是「归位用包围盒中点，与 `make_grid` 的约定一致，否则 padding
不对称」。**优先了 padding 对称，漏了 §15.4 的摆放噪声**：

包围盒中点由 6 个极端原子决定，远端侧链摆 1 Å → 中点移 0.5 Å ≈ 一个 h →
溶质相对格点的亚格点相位**逐帧扫动**。而同一构象只挪相位，ΔG_PB 峰峰值就是
**8.39 kcal/mol**（比 h=0.75→0.5 全局加密的 5.49 还大）。也就是说包围盒归位等于
往在线 ΔG(t) 注入一层 ~8 的白噪声，而 §23 的 block drift / ESS 要测的正是这个
量级 —— **监控信号会被自己的归位方式埋掉**。

改用质量加权质心（`recenter_com`）：单原子动 1 Å 只挪 mᵢ/M（~1/2000 Å），相位基本
冻结，摆放噪声从「帧间噪声」退化成「常数系统偏移」。代价是 padding 不再对称
（S4 上质心偏离包围盒中点最多 4.6 Å），由 margin 余量吸收。

**这条比离线路径更干净**：离线是「整轨迹共网格、坐标不动」，刚体漂移照样扫相位。
所以以后在线 vs 离线逐帧比较，两边必须用同一套归位，否则散点就是这 ±8 ——
离线侧（`solve.trajectory` 及喂它原始帧的脚本）**尚未改**，见 ONLINE_PLAN §7。

### 三道守卫（都是「不报错、只给看起来合理的错数字」那一类）

| 守卫 | 挡住什么 | 不加会怎样 |
|---|---|---|
| `__init__` 末尾的预热自检 | `sa_k` 不够 / padding 不够 / PB 不收敛 | 三者在 `__call__` 里都是「不抛、打标记」，构造时静默通过，要等 MD 跑起来第一帧写进 CSV 才发现 |
| `margin_A` + `margin_min` | 溶质触到网格边界 | 越界原子被 `clip + 权重置零` 悄悄丢掉（`pb/charges.py`），`G_PB` 偏小。`margin ≥ 0` 只保证**没丢原子**，Dirichlet 边界的物理阈值更大（S4 传 `margin_min=12 ≈ 1.5κ⁻¹`）|
| reporter 的单位哨兵 | nm/Å 忘 ×10、`solute_idx` 切错 | 包围盒尺度对参考帧的比例立刻暴露。**只在第 0 帧查** —— 单位错不会跑到一半才出现，而配体解离会让包围盒真的暴涨，那是 `margin` 负责 flag 的物理事件，不该在 flag 模式下杀掉几小时的 MD |

### `origin` 为什么**不**改成运行期参数

plan §21.5 的结论是「`origin` 运行期化 **或** `origin` 进 artifact 键，二选一」。
Python 阶段后者已自动满足：`enable_compilation_cache()` 的查找键是编译产物（HLO）
哈希，而 `origin` 烤在 HLO 里 —— **键天然完备，不存在跨体系错配**。运行期化的唯一
收益（跨体系复用同一份 executable）只在 PJRT artifact store 阶段存在，而它有 5 处
使用点，其中 `shell_indices` 是 host 侧 numpy。

**失效条件**：一旦做显式 artifact store / 跨 Context 复用 executable，必须回来做。

### 两个只在**生产配置**下才现形的 bug（跑起来才抓到，测试抓不到）

两条都是本项目的老题材：不报错，只安静地给一个错的（或空的）结果。

| # | 问题 | 位置 | 为什么测试漏了 |
|---|---|---|---|
| 14 | `assign_radii` 对**整个溶剂化 topology** 求值，撞到 Na⁺ 直接 raise —— 而切片在它**之后**。`OnlineMMPBSA` 因此在真实体系上永远建不起来 | `online.py` / `openmm_io/radii.py` | 测试用 conftest 的干双链，`solute_idx` = 全部原子，走不到这条路 |
| 15 | `describeNextReport` 恒返回 `interval`。OpenMM 每轮重新问全部 reporter、取 `nextSteps = min(...)`、**只触发 `steps == nextSteps` 的那些**，所以只要同时挂了间隔更短的 reporter（生产里必然有 DCD），本 reporter **一次都不触发**。实测 DCD@500 + PBSA@25000 跑 150k 步 = **0 帧**，且不报错 | `online.py` | 单元测试和 S6 都只挂一个 reporter —— 那时 `nextSteps` 就是它自己，恒返回常数反而正常 |

修法：`assign_radii(..., atom_indices=)` 只给选中的原子定半径（其余留 `NaN`），
**不是**往 Bondi 表里加离子半径 —— 加了就等于默许把抗衡离子当溶质做介电图；
`describeNextReport` 改成 `interval - currentStep % interval`（「距下次报告还差几步」，
这才是 OpenMM 的语义）。回归断言：`currentStep=1200` 时必须返回 300，
恒返回 500 的实现会挂。

> **共同的教训**：这两条都需要**同时存在另一个 reporter / 真实溶剂化体系**才暴露。
> 「单元测试全绿 + S6 benchmark 全绿」在这里等于零证据 —— 两个绿灯用的都是
> 退化配置。生产配置本身就是一个必须跑的测试档。

### `PBSAReporter` 的 flag 不再静默

`on_violation="flag"` 且没给 `out_csv` 时，`_violation()` 算出的违规字符串原来被直接
丢弃（S6 的第一轮实测正是这个配置，68 帧有没有越界当时无从得知）。现在 flag 会计数、
首次 `print`、`close()` 汇总。

### 在线部分的实测结论（全部数据见 RESULTS.md §16）

- **同卡争用没有额外惩罚**：实测 overhead 与 `T_pbsa/(N·t_step)` 差 ±1–3 个点且非系统性。
  MD 与 PBSA 干净串行 —— plan §21 的双 GPU async worker**至今没有证据支持**。
- **host 往返只占 0.24%**（0.95 ms `getState` + 0.081 ms 归位 / 431 ms 一帧）。
  plan §21.5 写的「拷贝占 1e-5」差 100 倍：真实代价是全体系 fp64 的 559 KB 加同步开销，
  不是溶质那 22 KB。**所以 PJRT 插件的立项理由只能是 capability，不能是性能。**
- **网格白多了两档 dime**：(193,225,225) → (193,193,193) 省 26.9% 时间，
  而 ΔG_PB / ΔG_MMPBSA 到小数点后两位不变。每个 overhead 档位整体上移一格
  （每 50 ps：6.4% → 4.6%，进 always-on）。
- **试跑长度**：单帧定尺欠 3.79 Å（1 ns 够）。与 SA 的 `k_neighbors` 同一个判决 ——
  不自动探测，显式参数 + 每帧硬验。
- **reporter 不扰动轨迹**，但 OpenMM CUDA 本身不可逐位复现（20 步 2e-5 Å）。
  后果：在线 vs 离线对比必须在**同一批已保存的帧**上做，不能重跑 MD 再比。

---

## 非对称网格设为默认 + 离线质心归位（2026-09-23，RESULTS §17）

**新入口 `jaxpbsa.pb.TripletSolver`**：离线（`crl.py` / `validate_mmpbsa.py`）与在线
（`OnlineMMPBSA`）共用。C/R 按复合物质心、L 按配体自己的质心归位；C/R 共用 h=0.75，
配体单独 h=0.25 紧盒（离线 padding 8，在线 14）。`h_lig=None` = 旧的三者共用网格。

| | 旧（共用 h=0.5）| 新默认 |
|---|---|---|
| 20 帧 ⟨ΔG_PB⟩ vs MMPBSA.py 780.52 | 816.1（+4.6%）| **778.46（−0.26%）** |
| ms/帧（轨迹网格）| 285 | **209–225** |

附带改动：`make_grid(center=)`、`GridSpec.center`、`recenter_com` 挪到 `pb/grid.py`
（`online` 里仍可 import）、`solve.pair`（C/R 共一份编译）、在线新增 `margin_lig_A`
（CSV 多一列）、`crl.py` / `validate_mmpbsa.py` 新增 `shared` / `--shared`、`--stride`、
`--pbsa-h`（pbsa 仍 0.5，与我们的 C/R 间距解耦）。

### 又一条被实测推翻的判断

§15.2 的「C−R 在 h=0.75 已收敛（0.75→0.5 只动 0.008）」是**单一摆放的巧合**。
7 种亚格点相位下 C−R 的 sd：h=0.75 **6.0**、h=0.5 **3.2**；按相位平均两者只差 0.8。
所以结论「C−R 无系统偏差、偏差全在配体」**成立**，但代价是 C/R 在 0.75 下逐帧噪声翻倍。
默认仍取 0.75：帧间构象 sd 31.4，摆放噪声只让方差多 2.6%，同样算力多 46% 的帧。

§12.2「与 pbsa 的 5% 是方法差异、不会收敛掉」**撤回**：配体加密后只剩 0.26%。

### 一个差点进去的守卫 bug

第一版把两张网格的 margin 取 min 成一个 `margin_A`。reporter 拿它比 `margin_min=12`
（C/R 的 Dirichlet 物理阈值），而配体紧盒按设计只剩 ~2–8 Å —— **每帧都会被 flag**。
拆成 `margin_A`（按 `margin_min`）与 `margin_lig_A`（只判 < 0）。

---

## 1YCR（MDM2–p53）接入，以及一条被第二个体系推翻的判断（2026-09-23，RESULTS §18）

新增：`scripts/prep_1ycr.py`（ACE/NME 双帽，产物同 S4 格式、两次运行 sha256 逐位相同）、
`scripts/grid_check.py --name`（摆放扫描 + 20 帧非对称 vs 共用）、`load_canonical(name=)`、
`run_s4_md.py` / `strip_unwrap.py` / `validate_mmpbsa.py` 的 `--name`。`strip_unwrap.py`
的溶质选择从 `"protein or resname PTR"` 改成前 `n_atoms` 个原子（ACE/NME 是否算
protein 取决于 mdtraj 的残基表）。

**推翻**：§17.2「C/R 取 0.75，摆放噪声随帧平均掉」。质心归位把相位冻住，摆放误差成了
逐帧一致的 ~+10 kcal/mol 偏差（1YCR +10.6，sd 1.70；S4 扫描里同一相位 +10.1）。
S4 与 Amber 的 0.26% 因此不是一般性结论（1YCR −5.9%）。**默认值未改，两个修法待定。**

**又一条错的预言**：「p53 只带 −2，配体网格偏差会小得多」—— 实测 24.6（S4 31）。
误差来自每个原子部分电荷的自能，不是净电荷。

**确认在 1YCR 上也成立**：padding 已收敛（0.004）、pbsa 已收敛（0.06%）、配体 h=0.25
已收敛（→0.2 动 1.2）、ΔE_MM 逐位级（5e-5 / 1.7e-6）。

---

## C/R 默认 h 0.75 → 0.5（2026-09-24，RESULTS §18.8）

`TripletSolver` / `OnlineMMPBSA` / `crl.py` / `validate_mmpbsa.py --h` 的 C/R 默认改为 0.5，
配体仍 0.25。新增 `TripletSolver.__call__(..., shift=)`（归位后给 C/R 叠加平移，诊断用）与
`scripts/phase_jitter_check.py`。

**又推翻一条**：§18.5 的「冻结相位」机制。每帧随机相位后 0.75 相对 0.5 仍低 6.8（S4）/ 11.1（1YCR），
与不打散几乎一样 —— 是 C−R 在 0.75 的真离散偏差，单构象 7 相位扫描（0.8 / 2.9）没测出来。
修法 ②（保持 0.75 + 随机相位）因此作废。代价：C/R 每帧 ~205 → ~327 ms（2080 Ti）。
对 MMPBSA.py：S4 +0.55%、1YCR −2.9%，剩余差异来源未测。

## 分数面 ε（实验，默认关）+ MG 横向并联粗化（2026-09-24，RESULTS §18.9）

- `PBParams.surface`：`"binary"`（默认，原行为）| `"fraction"`（`surface.ses_level` 连续 SES 水平集 +
  面上按溶质边长比例调和混合）。Born 误差 h=0.75 5.7% → 1.6%，相位 sd 1/4；1YCR 0.75−0.5 −11.1 → +5.1（仍未过 < 2）。
- `multigrid._coarsen_faces` 横向 注入 → [1,2,1]/4：迭代 −20~30%，ΔG 逐位不变，默认每帧 326 → 284 ms。
- `phase_jitter_check.py --surface`。

## `ses_level` 修正：凹面一阶误差、盒外假溶剂、截断下界（2026-09-24，RESULTS §18.10）

- `pb/surface.py` `ses_level`：新增交线圆探针中心候选（`crease=True` 默认），双球凹面误差 h=0.5
  −0.10 → −0.0005 Å；改在外扩 reach 的网格上计算再裁回（盒外不再当溶剂）；截断值 reach → reach − h，
  G 成为严格下界。只影响 `surface="fraction"`，默认 binary 路径逐位不变。
- `tests/test_pb.py`：`test_ses_level_two_spheres`（双球解析，含盒面切球）。
- 1YCR：修正后 fraction 的 C−R 在 h=0.5 已收敛（0.5→0.35 差 0.17）。收益对用户不明显，就此停手，默认不改。
- 路线 A（平滑介电）试过不成立：DelPhi 高斯连续能量有限但比锐界面大 15×，网格一阶极慢收敛（与求解器无关，面中点取 ε 也一样）；三次样条收敛但粗网格不胜 binary。
  代码未留，见 RESULTS §18.11。

## 高斯 ε + 亚网格自项修正（2026-09-24/25，RESULTS §18.12–18.13）

- `PBParams(surface="gaussian", gauss_sigma=0.93, gauss_selfcorr=True)`：`pb/surface.gaussian_density`（DelPhi
  高斯密度，JAX scatter，3σR 截断），ε = ρ ε_in + (1−ρ) ε_out，面上调和平均。
- `pb/gauss_selfcorr.py`：电荷中心 ~0.2 Å 内层使 h=0.5 自项欠解析 ~60%；按等效孤立原子 b_i = σR_i/√C_i 离线
  打表 T(b/h, 相位)（`pb/_tables/`，h=1 打一次，尺度不变），运行时加到 `g_pb`（`g_pb_raw` 保留）。开销 ~0。
- 1YCR：修正后 ΔG_PB 在 C/R h 0.4–0.3 平台 ~455；h=0.5 低 8%，0.75 低 30% → 生产 h≈0.4。盐不影响修正。
- 外部基准：漫射界面 Kirkwood 球 LPB（J. Comput. Phys. 545 (2026) 114452）复现到 +0.001%（h=0.2）。
- 注意：该文献的 Gaussian PB 是 ε_gap=8 + 光滑面 S(r) 的另一模型，我们的实现不是它。默认 surface 仍 binary。
- `tests/test_pb.py::test_gauss_selfcorr_pair`；`pyproject.toml` 加 package-data（表文件）。

## 在线网格定尺改成算出来的（2026-09-25，ONLINE_PLAN §8 待办 1–3，RESULTS §18.15）

- `OnlineMMPBSA`：去掉魔数 `padding=30`/`padding_lig=14`；新增 `pilot_coords_A`（试跑轨迹 [T,N,3]，推荐）/
  `fluctuation_allowance`（Å）/ `margin_min`（默认 1.5κ⁻¹）。两者都不给、或 pilot 只有 1 帧 → `ValueError`。
  显式 `padding`/`padding_lig` 仍可覆盖。`analyzer.sizing` 记录定尺结果。
- `online.boundary_margin_min(params)`；`PBSAReporter(margin_min=None)` 默认取 analyzer 的值（原默认 0）。
- **行为变化**：旧调用（只给 ref_coords_A）现在会报错，需补 `pilot_coords_A` 或 `fluctuation_allowance`。
- S4：1 ns pilot → 193³ + 配体 (225,161,193)，496.5 → 288.3 ms/帧（−42%），ΔG_PB 差 0.02。
- `scripts/online_overhead.py` 用 S4 干轨迹前 1 ns 作 pilot；`scripts/fit_grid.py` 默认 h 0.75 → 0.5。
- 测试：`test_sizing_requires_fluctuation_info` / `test_boundary_margin_min_from_ionic_strength` /
  `test_pilot_sizing_covers_every_pilot_frame`。

## fraction 在 RTX 5090 上 OOM 修复（2026-09-29，RESULTS §18.16）

- `pb/surface.ses_level`：max 链每 8 个偏移插 `jax.lax.optimization_barrier`（数学不变，逐位相同）。
  5090 上 1YCR 配体紧盒临时显存 16.06 → 0.50 GiB（2080 Ti 0.52，不受影响）。只影响 `surface="fraction"`。
- `OnlineMMPBSA`：曾试「给了 pilot 就默认再加 3.5 Å」，5090 全轨迹检验后撤回（1YCR C/R 不升档、S4 配体盒
  白涨 60%；3.19 Å 的依据算错，真实撑开 ≥ 6.7 Å）。`fluctuation_allowance` 默认 0，不替调用方猜涨落；
  正确性由每帧 margin 守卫保证。

## margin_min 默认 1.5κ⁻¹ → 1.0κ⁻¹（2026-09-29，RESULTS §18.17）

- `online.boundary_margin_min` 返回 1.0·κ⁻¹（0.15 M：7.85 Å）。依据：1YCR 全轨迹最差 3 帧、相位固定只改盒子，
  1.0κ⁻¹ 相对 3.0κ⁻¹ 的 ΔG_PB 差 ≤ 0.010 kcal/mol（0.75κ⁻¹ ≤ 0.018）。在线定尺与 `PBSAReporter` 守卫同用，
  网格会相应变小。`scripts/fit_grid.py`、README、测试同步。新增 `data/md/boundary_conv.py`（gitignored）。

## 定尺目标与逐帧 flag 阈值分开（2026-09-29，RESULTS §18.17）

- 新增 `online.boundary_margin_flag(params)` = 0.1·κ⁻¹（无盐时 0）；`OnlineMMPBSA(margin_flag=None)` 默认取它，
  记入 `sizing`。`PBSAReporter(margin_min=None)` 默认改为 `analyzer.margin_flag`（原为定尺目标 margin_min）。
- 依据：1YCR 最差 3 帧相位固定只改盒子，0.1κ⁻¹（~0.9 Å）处 ΔG_PB 边界误差 ≤ 0.073 kcal/mol，平滑无悬崖；
  定尺目标 1.0κ⁻¹ 保留（兼作 pilot 未见涨落的缓冲）。1.0κ⁻¹ 下 1YCR 有 47% 帧 < 7.85 Å，但误差都 ≤ ~0.07。

## `surface="gaussian_gap"`：异质高斯 PB（JCP 545 (2026) 114452）（2026-09-29，RESULTS §18.18）

- `pb/surface.py`：`gaussian_density(..., m=)`（super-Gaussian）；`gcs_kernel` / `gcs_surface`（Gaussian convolution
  surface，MBE 18 (2021) 1370 §3，离散核归一化）；`gap_config`；`build_maps(gap=...)` 另出真空态面 ε_v。
- `pb/energy.py`：`PBParams(surface="gaussian_gap", gauss_m, eps_gap=8, gcs_probe=1.5, gcs_sigma2=1, gcs_tau=0.0025,
  eps_vacuum=1)`；参考解改为真空态 ε_v 的 MG-PCG（该 surface 下不用 DST），共用网格 triplet 里 C 单独解参考。
- 测试：`test_gaussian_gap_maps_bounds`、`test_gaussian_gap_vacuum_identity`。
- S4 / 1YCR：h=0.75 已收敛（相对 0.25 差 0.24% / 0.01%），摆放噪声 sd ≤0.3，每帧 28 / 25 ms（5090）；
  ΔG_PB 是 SES 的 0.59× / 0.72×（不同模型）。默认 surface 仍 binary。

## SA 在 Ampere+ 卡上的 TF32 精度 + zsasa 在跑不了的 CPU 上跳过（2026-09-29）

- `sa/jax_sr.py`：埋藏判据那次 `u @ disp.T` 改 `jnp.matmul(..., precision=HIGHEST)`。Ampere 以后的卡（5090 实测）上
  fp32 matmul 默认 TF32（10 位尾数），贴边采样点判反：`test_burial_identity_against_brute_force` 在 5090 上
  612.758 vs 612.663（1.6e-4 ≈ 一个点）。内维 3，代价可忽略；2080 Ti 上逐位不变。PB/MM 无 matmul（DST 走 FFT），不受影响。
- `tests/test_sa.py::_zsasa_available`：zsasa 二进制在、但被信号杀掉（rc < 0；fulda 上 rc=−4 = SIGILL，二进制按别的 CPU 编）
  时视为不可用而跳过；普通非零退出仍是失败。

## 非线性 PB + JCP26 Table 4 对照（2026-09-29/30，RESULTS §18.20–18.21）

- `pb/energy.py`：`PBParams(nonlinear=True, newton_max_iter=60, newton_inner_tol=1e-3)`；`npb_newton`（阻尼 Newton，
  J = κ̄² cosh u 的 LPB 算子，内层 PCG+MG fp32，外层 u / F / Φ fp64；凸能量 Armijo，试探点非有限减半，线搜索失败即停，
  验收 ‖F‖ ≤ tol·‖b‖）；有界双曲 `_bsinh/_bcosh/_bcosh_m1`（|u| > 30 一阶延拓，能量与导数一致）；`npb_ion_energy`
  （式 13 离子项，输出 `g_pb_ion`）。`PBParams.gcs_normalize`（默认 True；False = 原文不归一化核，仅对照用）。
- 测试：`test_npb_kappa0_equals_lpb[binary, gaussian_gap]`、`test_npb_weak_potential_limit`、`test_npb_strong_charge_converges`。
- `scripts/prep_pdb25.py`：Table 4 的 25 个蛋白制备；`VARIANTS`（`1C75_cym`：C32/C35 取 CYM；`1TQG_altB`）。
- `data/md/pdb25_conv.py`（gitignored）：逐 h / 逐相位记录（失败只记 status 不丢行），`--sigma= --gcs_raw --radii= --tag=`。
- 结论：σ = 1.0 时 22/25 个蛋白对原文中位差 1.7%（r = 0.9989）。
- `PBParams.gap_sigma = 1.0`（新，gaussian_gap 专用；为复现 JCP26 反推，正文没写 σ）；`gauss_sigma = 0.93` 只管
  `surface="gaussian"`。**行为变化**：gaussian_gap 的默认 σ 0.93 → 1.0（§18.18 的 S4 / 1YCR 数字是 0.93 下的）。
