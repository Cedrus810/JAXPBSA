# 运行命令（2026-10-11）

**状态（2026-10-17）：现行流程是 v2.1，实际使用的命令见文末“v2.1 实际使用的作业”一节。** 下面 §0 起的内容是 2026-10-11/12 的 v1 记录，保留备查。

原状态（2026-10-12）：PMI 分子冒烟（准备 → setup → 1 ns → strip）在用户机器 fulda（RTX 5090）上跑通，立体化学检查全部通过（见交接 §9.1）。

原状态记录： 下面每条命令都已用 `--dry-run` 静态核对过：参数解析、清单读取、路径映射、24 行身份与种子不冲突。准备、溶剂化、最小化、MD、去溶剂这些步骤一条都没有执行过，没有任何分子程序跑通的证据。节点、GPU 和生产长度都没定，不能直接提交。

## 必读（开工前全部读完）

| 文件 | 为什么要读 |
|---|---|
| `/home/ruigengji/JAXPBSA/docs/PUBLIC_DATA_EXECUTION_HANDOFF.md` | 执行边界与当前停点（§9）。 |
| `protocol.md`（本目录） | 末端、GAI、水、altloc、质子化的决定和依据（§1），物理设置（§2）。 |
| `proposed_runs.csv`（本目录） | `run_id`、种子、输出目录；`run` 子命令只认这张表。 |

## 0. 环境

**用户决定（2026-10-11）：** 分子运行放在用户自己的机器上，由用户执行；我不选节点、不提交作业。P2 诊断时长定为 10 ns。第一步是分子冒烟脚本 `smoke_pmi_rep01.sh`：PMI 的 prep → setup → 1 ns → strip，`--dcd-ps` 默认 10（我的建议，用户尚未确认），跑完后原样把 `--prod-ns` 改成 10 续跑。运行的机器必须能在 `/home/ruigengji/JAXPBSA` 看到这棵 NFS 目录树，因为 `proposed_runs.csv` 里存的是绝对路径。


- 解释器：`openmm_dev`（生产环境，只读使用，不安装、不升级）。181 上是 `/home/ruigengji/miniforge3/envs/openmm_dev/bin/python`，180 上是 `/home/canna/miniforge3/envs/openmm_dev/bin/python`。下文写作 `$PY`。指定节点后先确认该节点上这个路径存在。
- 工作目录：`/home/ruigengji/JAXPBSA`。下文写作 `$J`。
- 远程 shell 是 zsh，命令一律包在 `bash -c '…'` 里，后台作业按全局规则 §3 的 `setsid nohup … & echo $!` 写法启动，并记录 PID。
- CPU 线程：准备步骤用 Reference/CPU 平台加氢、建参数，没有最小化。运行前显式设置 `OMP_NUM_THREADS`；具体数值等节点定下后，按该机物理核数和占用情况决定。

## 1. 准备：每个样本一次（CPU）

```bash
$PY $J/scripts/prep_peptide_sample.py --sample PMI PMI-L10A P12Q_L10A --model-id model01 --seed 20261011   # P2 三样本
$PY $J/scripts/prep_peptide_sample.py --all --model-id model01 --seed 20261011                            # 八样本
```

- 输出：`prepared/<peptide_name>/model01/{complex_dry.cif,complex_dry.pdb,system_vacuum.xml,parameters.npz,prep_record.json}`。`prep_record.json` 最后写入，有这个文件才算该样本完成。
- 续跑：重跑同一条命令，已有 `prep_record.json` 的样本会跳过；加 `--force` 强制重做。换种子或换处理方式，就换一个新的 `--model-id`。
- 执行时出现以下情况会直接失败，不会继续：altloc A 与 PDBFixer 读到的坐标不一致；补建后序列不符；任一末端不是游离的 NH3+/COO−；ff14SB 找不到匹配模板。
- 要人工审阅的记录：His 状态、受体/肽/复合物净电荷、补建和突变残基附近最近的非键重原子对。这些只报告，不设阈值。审阅后在 `sample_structure_audit.csv` 中更新 `chemistry_ready`、`structure_ready`、`parameter_ready`，并写明依据。

## 2. 溶剂化与最小化：每个样本×模型一次（GPU）

```bash
$PY $J/scripts/run_peptide_md.py setup --sample PMI --model-id model01 --platform CUDA --device-index 0
```

- 输出：`md/<peptide_name>/model01/setup/{solvated.cif,solvated.pdb,system.xml,minimized_state.xml,setup_record.json}`。三个副本共用这份 setup。
- 同一个样本，必须先跑完 setup，再起副本。setup 已存在时会跳过；`--force` 会重建 setup，但已基于旧 setup 跑出的副本就和它不一致了，只能在没有副本时使用。
- 溶剂和离子放置的种子是 `--seed`，默认 20261011，写入 `setup_record.json`。
- 最小化分两段（2026-10-12 修复）：第一段对全部 ω 和 CA 手性二面角加约束（k = 1e4 kJ/mol/rad²，目标值取准备后结构的实测值）；第二段撤掉约束再最小化。两段结束后都与准备后的结构比较顺反、L/D 类别，有任何变化就报错，不写 setup_record。保存的 `system.xml` 不含这个约束，MD 全程也不加。原因：第一次冒烟时，无约束最小化把 Glu25–Thr26 翻成了顺式，见 `archive/smoke1_20261011_cis_glu25/README.md`。

## 3. 副本 MD（GPU）

```bash
$PY $J/scripts/run_peptide_md.py run --run-id PMI_model01_rep01 --platform CUDA --device-index 0 --prod-ns <待定>
```

- 阶段：NVT 100 ps → NPT 1000 ps → 生产 `--prod-ns`。生产按 `--chunk-ps`（默认 1000 ps）切块，DCD 每 `--dcd-ps`（默认 1 ps）存一帧。NVT 和 NPT 的长度沿用 `run_s4_md.py`，平衡是否充分要看诊断结果。
- `--prod-ns` 是必填项，没有默认值。生产长度在诊断后才定。用于诊断的小批次长度，等节点和预算定下后，由用户在作业计划里批准。
- **续跑：原样重跑同一条命令。** 已完成的阶段会跳过。生产从最后一个完成的块继续；没写完的块从上一块的状态重写。之后加大 `--prod-ns` 就是延长。不能改 `--chunk-ps`、`--dcd-ps`、`--nvt-ps`、`--npt-ps`、`--eq-restraint-k`，改了会报错并提示另开目录。
- 种子取自表中该行，实际用到的每个种子都写在 `status.json` 的 `contexts` 字段里。续跑后的副本与一次跑完的副本在统计上等价，但逐位不相同（XML 状态不保存 Langevin 随机数流）。
- 平台没有自动回退到 CPU。

（启动后 shell 立即显示 `Done`，`$!` 也只是 setsid 父进程的 PID，这些都正常；脚本真实的 PID 记在 log 的 `== start … pid N` 行。）

P2 诊断的整批脚本是 `p2_diagnostic.sh`：顺序执行，可续跑，日志固定写到包内的 `p2_diagnostic.log`；每个样本先 prep、再 setup，然后跑 3 个 10 ns 副本，每个副本跑完就 strip，`--dcd-ps 10`。已用 dry-run 静态核对：PMI 的 prep 和 setup 已完成会跳过，rep01 从第 1 块续跑，种子无冲突。

P2 诊断 9 行：`PMI_model01_rep0{1,2,3}`、`PMI-L10A_model01_rep0{1,2,3}`、`P12Q_L10A_model01_rep0{1,2,3}`。P3 另外 15 行，要等诊断后固定了协议再跑。

**2026-10-12 状态：** PMI 三个副本已跑完。PMI-L10A 的准备被立体化学检查拦下（A:GLU25 为 D 型），原因已经查明，修复已加入 `prep_peptide_sample.py`（`repair_built_chirality`），详见交接 §9.2。原样重跑 `p2_diagnostic.sh` 即可接着跑。当时查原因用的是下面这条命令（CPU 运行，不碰 `prepared/`）：

```bash
cd /home/ruigengji/JAXPBSA
PY="$(command -v python)"
"$PY" experiments/2026-10-11_public_data_execution_packet/scripts/diag_prep_stereo.py --sample PMI PMI-L10A P12Q_L10A \
    2>&1 | tee experiments/2026-10-11_public_data_execution_packet/diag_prep_stereo.log
```

**起点敏感性 model02**（用户 2026-10-12 批准；只改准备种子；准备后先和 model01 比较补建残基，起点相同就停）：

```bash
cd /home/ruigengji/JAXPBSA/experiments/2026-10-11_public_data_execution_packet
PY="$(command -v python)" DEV=0 MODEL=model02 PREP_SEED=20261012 setsid nohup bash p2_diagnostic.sh > /dev/null 2>&1 < /dev/null &
tail -f p2_diagnostic_model02.log
```

**2026-10-13：** model02 的 P12Q_L10A 在 setup 第二段被检查拦下，Thr26 翻成了 D 型（交接 §9.2）。下面这条命令用来查原因，几分钟内跑完，不碰 `md/`：

```bash
cd /home/ruigengji/JAXPBSA
PY="$(command -v python)"
"$PY" experiments/2026-10-11_public_data_execution_packet/scripts/diag_setup_stereo.py --sample P12Q_L10A --model-id model02 --watch A:THR26 \
    2>&1 | tee experiments/2026-10-11_public_data_execution_packet/diag_setup_stereo.log
```

**准备协议 v2 验证（2026-10-13；设计稿 `docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md`）。** MODELLER 起点已在 181 上建好。以下 V2（prep + setup）和 V3（每个模型跑 1 ns）在 fulda 上执行：

```bash
cd /home/ruigengji/JAXPBSA/experiments/2026-10-11_public_data_execution_packet
PY="$(command -v python)" DEV=0 setsid nohup bash p2_v2_validation.sh > /dev/null 2>&1 < /dev/null &
tail -f p2_v2_validation.log
```

## 4. 去溶剂与诊断（CPU）

```bash
$PY $J/scripts/strip_peptide_replica.py --run-id PMI_model01_rep01 --diag-stride 10 --dry-stride 1
```

- 只读取 `status.json` 里标记为已完成的块。输出：`<replica>/analysis/{solute_dry.dcd,solute_dry.pdb,diagnostics.csv,diagnostics_summary.json}`。
- 诊断量：受体 Cα RMSD；受体坐标系下的肽 Cα RMSD；第 12 位 Cα 位移；受体与肽之间的最小重原子距离；4 个晶体水位点（protocol §1.2）各自最近水 O 的距离；chunk 日志中的温度和密度。全部只报告，不设通过/拒绝阈值。
- 水位点用最小镜像距离计算，前提是盒子为正交盒。`addSolvent` 默认生成立方盒，满足这个前提。
- 可以在生产进行中运行；新的块完成后重跑即可覆盖 analysis 目录。

## 5. 评分入口：尚未补

JAXPBSA 逐帧评分、Amber/GB 同帧对照、统计汇总这几项，等 P2 定下取帧规则之后再补，本次没有写。读取的输入是 `analysis/solute_dry.dcd`、`solute_dry.pdb` 和 `prepared/.../parameters.npz`。

## 6. 静态检查记录（本机，2026-10-11）

- `py_compile` 三个入口：通过。
- 系统 python3（没有 openmm/numpy）运行 `--dry-run`：`prep --all` 八条的突变和补建残基与 `sample_structure_audit.csv` 一致，P12Q 第 12 位补 GLN；`run` 和 `strip` 24 行解析通过；种子冲突检查在 `--prod-ns 10`（10 块）和 `300`（300 块）下都是 0；未知 `run_id`、未知样本、长度不整除时都报错。
- 读 openmm 8.5.2、pdbfixer 1.12.0、mdtraj 1.11.1 的源码文本，核对所用 API 签名（`PDBFixer(platform=)`、`addMissingAtoms(seed=)`、`addHydrogens(platform=)`、`addSolvent(boxShape='cube')`、`DCDFile(firstStep, interval)`、`image_molecules(anchor_molecules=)`）。核对过程中没有 import openmm。
- 用随机点做 numpy 纯数学自检：Kabsch 拟合误差 1e-15，水位点逆变换误差 7e-16。
- 2026-10-12：`peptide_geometry.py` 的类别检查用第一次冒烟的产物验证过：准备后的结构 0 条违例；最小化后的结构报出 Glu25–Thr26 由 −176.3° 变为 −8.8°；旧轨迹 100/100 帧都报出这一项。修复后的两段最小化**还没有在节点上运行过**。
- **第一次冒烟之前没有做过**：任何 OpenMM Context、PDBFixer 补建、加氢、溶剂化、最小化、MD 或 mdtraj 读轨迹。第一次真实运行应该在指定节点上，先跑 PMI 的 prep → setup → run（短 `--prod-ns`）→ strip，这一次就是分子冒烟。


## v2.1 实际使用的作业（2026-10-14 至 10-17）

所有路径都相对于本包目录。PBS 作业由用户 `qsub`，日志写在本包里。各作业重新 `qsub` 即可续跑。

| 步骤 | 在哪跑 | 命令或脚本 | 日志与产物 |
|---|---|---|---|
| 突变 + 补建（重原子） | 181，系统 python 3.14 | `python3 experiments/2026-10-11_public_data_execution_packet/scripts/build_mutant_modeller.py --all --start v2m1 v2m2`（在项目根目录下） | `inputs/modeller_v21/<样本>/heavy_v2m{1,2}.pdb`、`build_record.json` |
| 加氢 | 183 | `bash experiments/2026-10-11_public_data_execution_packet/scripts/add_h_reduce.sh`（全部样本；reduce `-NOFLIP -HIS -NUClear -Nterm25`） | `reduced_v2m{1,2}.pdb` |
| 命名核对与静态检查 | 181，openmm_dev python（脚本说明：只做 numpy 和 XML 解析，不建 OpenMM 体系） | `$PY experiments/2026-10-11_public_data_execution_packet/scripts/finalize_allh_models.py --all` | `start_v21m{1,2}.pdb`、`finalize_record.json` |
| prep（C2）+ setup（C3）+ MD + strip | PBS | `p2_v21_validation.pbs`（V2/V3，1 ns）、`p2_v21_V5.pbs`（10 ns）、`p2_v21_ext20.pbs`（延长到 20 ns，并接着给 10–20 ns 打分）；三者都调用 `p2_v21_validation.sh` | `p2_v21_validation.log`、`prepared/`、`md/` |
| 评分 | PBS | `score_smoke.pbs`（1 个副本）、`score_all.pbs`（1–10 ns）、`score_1_20.pbs`（**主评分**，1–20 ns，统一定尺）；都调用 `score_all.sh`，可用环境变量 `START_NS`、`END_NS`、`OUT_ROOT`、`LOG` | `score_*.log`、`scores*/` |
| 汇总 | 183，openmm_dev python | `$PY scripts/aggregate_scores_ext20.py`（在本包目录下；默认读 `scores_1_20/`，`--source split` 重现分段版本） | `scores_1_20_summary/`、`scores_ext20/` |

单个副本的命令（与作业中相同）：

```bash
$PY scripts/prep_peptide_sample.py --sample PMI --model-id v21m1 --protocol v2.1      # 在项目根目录下
$PY scripts/run_peptide_md.py setup --sample PMI --model-id v21m1 --platform CUDA --device-index 0
$PY scripts/run_peptide_md.py run --run-id PMI_v21m1_rep01 --platform CUDA --device-index 0 --prod-ns 20 --dcd-ps 10
$PY scripts/strip_peptide_replica.py --run-id PMI_v21m1_rep01 --diag-stride 1 --dry-stride 1
$PY scripts/score_peptide_replica.py --run-id PMI_v21m1_rep01 --start-ns 1 --end-ns 20 --stride 1 \
    --out experiments/2026-10-11_public_data_execution_packet/scores_1_20/PMI/v21m1/rep01
```
