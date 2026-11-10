# BRD4-5 重新准备与重跑方案（阿普唑仑，2026-11-01）

状态：已完成（2026-11-01）。183 冒烟（§8）后，yayoi28 作业 22051 跑完准备、3 × 5 ns 与评分；P1/P2 v2 已接回（RESULTS §21.4）。用户 2026-11-01 回答"要我写 BRD4-5 重新准备和重跑的方案吗？"时说"可以"。

## 必读（开工前全部读完）

| 文件 | 为什么要读 |
|---|---|
| `CLAUDE.md`（项目根） | 冻结方法、生产环境、准备与计算在哪台机器上做 |
| `docs/BRD4_HOU_EXECUTION_HANDOFF.md` §2、§7.4–§7.8 | BRD4 沿用上游 Amber 输入的规则、原 setup/MD/评分协议与实测耗时 |
| `experiments/2026-10-17_small_molecule_benchmark/reports/brd4_statistics_20261101/report.md` §1 | BRD4-5 输入错误的证据 |
| `docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md` §4–§5 | B1/B2 与 P2 统计口径，重跑结果要接回这里 |

## 1. 已核实的问题

- 上游（MobleyLab/benchmarksets commit 5233e0a）BRD4-5 配体的 sdf、mol2、prmtop 一致，都是 C17H15ClN4，37 个原子。mol2 中 NAL（原子 15）是 N.3、带 H8（原子 31），CAO（原子 16）是 C.3、带 H9（原子 32），NAL–CAO 为单键：这是 5,6-二氢阿普唑仑。
- 实验化合物是阿普唑仑 C17H13ClN4（PubChem CID 2118，InChIKey VREFGVBLTWBCJP-UHFFFAOYSA-N）；晶体配体 CCD 08H 同为 C17H13ClN4；BRD4(1)–08H 共晶只有 PDB 3U5J（RCSB 检索）。
- 已跑的 `md/brd4/BRD4-5/mobley_amber/` 3 × 5 ns 与全部评分对应二氢形式；P2 已移出主统计。

## 2. 这批结果决定什么

BRD4-5 能否回到 P2 主组（ITC 15 °C 同方案，n 从 4 变 5），以及同篇对 5/7 能否解释。其他 9 条不动。

## 3. 准备：只换配体，其余逐位沿用上游

原则与执行交接 §2.1 一致：BRD4 沿用上游 Amber 输入的蛋白、水、离子、盒子与参数，不重新溶剂化。只修这一个配体。

1. **配体坐标**：取上游 `BRD4-5.crds` 中配体（第 122 残基，原子 2035–2071）的坐标，即上游的晶体姿态与体系坐标系。删去 H8、H9，把 NAL=CAO 设为双键，其余原子、原子名与顺序不变（35 个原子）。
2. **身份核对**：RDKit 生成的 InChIKey 必须等于 VREFGVBLTWBCJP-UHFFFAOYSA-N，重原子连接与键级和 CCD 08H 同构。不等就停，不进入下一步。
3. **参数**：antechamber `-c bcc -at gaff`（与上游 README 的 AM1-BCC / GAFF 一致，不改 GAFF2），parmchk2；记录 AmberTools 版本（上游版本未知，这一点与其余 9 条不同，作为限制写明）。坐标只用于电荷计算，输出 mol2 坐标须与输入一致（同 Hou 准备的核对）。
4. **组装**：用 ParmEd 把上游结构中的蛋白（残基 1–121）、新配体、上游水和离子按原顺序拼接，盒子沿用；写 `complex_solv.prmtop/.crds`。
5. **静态核对**（记录在 `prep_record.json`，不设自动阈值）：
   - 原子数 35 137（原 35 139 − 2）；配体为第 122 残基、原子 2035–2069；
   - 非配体原子的电荷、LJ 类型、键/角/二面角参数与上游 prmtop 逐项相同；
   - 配体部分电荷和与溶质净电荷（用于 `OnlineMMPBSA` 的整数检查，容差 1e-2 e）；
   - parmchk2 有无 ATTN 项、tleap "Errors = 0"。
6. **残基名** 改为 08H，`manifests/brd4_inputs.csv` 格式的新清单只含这一行。

## 4. 目录与入口

- 新建独立小包 `experiments/2026-10-17_small_molecule_benchmark/brd4_5_alprazolam_20261101/`，内含 `manifests/{runs.csv,brd4_inputs.csv}`、`prepared/`、`md/`、`scores/`，先例是 `acceptance_online_20261028/`。**不改主包的 `manifests/runs.csv`**：Hou 10-30 的 `audit.json` 绑定了它的 sha256，改了会让 `scripts/rank_hou_sampling.py` 拒绝复现。
- model_id `alprazolam_gaff_fix`；3 个副本沿用原 BRD4-5 的种子（20261030/31/32 及对应 integrator/barostat 种子），使与旧结果相比只换了配体。
- 新脚本：包内 `scripts/prepare_brd4_5_alprazolam.py`（第 3 节）。
- 现有入口：`run_smallmol_md.py` 与 `score_smallmol_replica.py` 已支持 `--package/--inputs`，不改。`score_smallmol_start.py` 需加 `--model-id` 与 `--inputs` 选项（默认值不变）。
- 旧的 `md/brd4/BRD4-5/mobley_amber/` 与 `scores/*/brd4/BRD4-5/` 原样保留，标"身份错误"。

## 5. 运行与验证

| 步 | 内容 | 机器 | 依据 / 耗时 |
|---|---|---|---|
| 冒烟（一次） | 准备脚本 → `run_smallmol_md.py setup`（无约束最小化），到 setup 结束为止 | 183（2080 Ti） | 新代码只有准备这一段；Hou AM1-BCC 每条 2–4 min（70 原子级配体），setup 约 20 s |
| 正式 | 准备 → setup → 3 副本 × (NVT 0.1 ns + NPT 1 ns + 生产 5 ns) → strip → 评分默认与 h0.4 网格 (0,5] ns → B1 起点评分（默认） | 用户给的 PBS 节点 | A4000 实测每副本 26 + 328 + 5 × 352 s ≈ 35 min；评分每副本每套网格 500 帧 × 0.1–0.24 s + 预热 ≤2.3 min。3 块卡并行约 45 min，1 块卡串行约 2–2.5 h（未在目标节点实测） |

- 可续跑：沿用现有入口的 chunk/帧级跳过；准备与 setup 已完成就跳过。
- 不另设验收门。报告：分数、副本极差、配体 RMSD 与旧二氢形式的对比（只作描述）。

## 6. 接回统计

结果出来后，P2 协议新增一版（新输出目录，旧的保留）：BRD4-5 用新模型，主组变为 BRD4-5–9（n = 5），5/7 对可解释；P1 的 BRD4 组同样换源重算。改动写进新协议的 `changes` 字段。

## 7. 不做的事

- 不重新准备其余 9 条，不改 GAFF→GAFF2，不重新溶剂化。
- 不为对上实验而调整 BRD4-5 的质子化/互变异构以外的任何参数。

## 8. 冒烟记录（2026-11-01，183，RTX 2080 Ti）

- 包 `brd4_5_alprazolam_20261101_smoke/`，脚本 `brd4_5_smoke.sh`，日志 `brd4_5_smoke.log`。第一次在 ParmEd 拼接处失败（复制切片后的 AmberParm 会重查 LJ 表长度；日志 `brd4_5_smoke_try1_parmed_slice.log`），改为先转普通 Structure 后重跑通过。
- 准备 41 s：InChIKey VREFGVBLTWBCJP-UHFFFAOYSA-N（C17H13ClN4）；AM1-BCC 电荷和 −0.002；parmchk2 无 ATTN；mol2 与输入坐标最大差 0.0005 Å；AmberTools 24.8（conda 包）。
- 拼接：35 137 原子，配体第 122 残基、原子 2035–2069，残基名 08H；配体以外 35 102 个原子的名称、电荷、LJ、44 766 组成键项和坐标与上游 prmtop 完全相同。溶质净电荷 2.998（整数检查容差 1e-2 e 内）。
- setup 38 s：无约束最小化，配体重原子最大位移 1.27 Å。
- B1 起点评分有效：−24.11 kcal/mol（旧二氢形式的 B1 为 −20.78，只作描述）。冷启动初始化 125 s，评分 163 ms。
- 正式作业已写：`brd4_5_full.sh`（MODE=setup：准备 + setup + B1；MODE=worker：5 ns MD → strip → 默认与 h0.4 评分）、`brd4_5_full.pbs`（host 占位，最多 3 块空闲 GPU）。run/strip/score 入口已对冒烟包 dry-run。

## 未确认（Claude 推断）

- 上游多出的 2 个 H 可能是 openbabel 加氢时丢了 C6=N5 双键所致；没有核实上游的生成过程。
- 用上游坐标（二氢形式最小化前的晶体姿态）删 H 后，C6 处的几何仍是四面体残留，靠 setup 的无约束最小化放松；是否需要先单独优化配体，没有评估。
