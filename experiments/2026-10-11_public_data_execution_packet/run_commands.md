# 运行命令（现行流程 v2.1）

日期：2026-10-11 建立，2026-10-17 整理。v1 时期（2026-10-11/12，PDBFixer、fulda 上手动执行）的命令说明已移到 `archive/run_commands_v1_20261017.md`，现在不再使用。

## 必读（开工前全部读完）

| 文件 | 为什么要读 |
|---|---|
| `/home/ruigengji/JAXPBSA/docs/PUBLIC_DATA_EXECUTION_HANDOFF.md` | 当前状态、机器分工、操作要点（§1、§6）。 |
| `/home/ruigengji/JAXPBSA/docs/PUBLIC_DATA_PREP_V2_MODELLER_DESIGN.md` | 每一步做什么、检查什么（§2）。 |
| `protocol.md`（本目录） | 输入化学、MD 设置、取帧规则。 |
| `proposed_runs.csv`（本目录） | `run_id`、种子、输出目录；`run` 和评分子命令只认这张表。 |

## 0. 环境

- **解释器：** `openmm_dev`（生产环境，只读使用，不安装、不升级），路径 `/home/ruigengji/miniforge3/envs/openmm_dev/bin/python`，下文写作 `$PY`。
- **MODELLER：** 181 的系统 python 3.14，不在 `openmm_dev` 里。
- **路径：** 运行的机器必须能在 `/home/ruigengji/JAXPBSA` 看到这棵 NFS 目录树，因为 `proposed_runs.csv` 里存的是绝对路径。
- **PBS 模板：** zsh；`#PBS -q default`；`#PBS -l select=1:ncpus=N:ngpus=1:mem=60gb:host=<节点>`；source mamba 后 `mamba activate openmm_dev`。现有的 `*.pbs` 都是这个格式。
- **续跑：** 每个作业重新 `qsub` 即可，已完成的步骤、块和帧会跳过。

## 1. v2.1 实际使用的作业（2026-10-14 至 10-17）

所有路径都相对于本包目录。PBS 作业由用户 `qsub`，日志写在本包里。各作业重新 `qsub` 即可续跑。

| 步骤 | 在哪跑 | 命令或脚本 | 日志与产物 |
|---|---|---|---|
| 第一段：补建 Glu25、第 12 位（PRO、GLN 各 20 个模型） | 181，系统 python 3.14 | `python3 experiments/2026-10-11_public_data_execution_packet/scripts/build_terminal_modeller.py --res12 PRO --models 20`（GLN 同理） | `inputs/modeller_v2/{PRO,GLN}/model_*.pdb`、`build_record.json` |
| C1 和选起点 | 181，openmm_dev python（只做文件解析和 numpy） | `$PY experiments/2026-10-11_public_data_execution_packet/scripts/select_terminal_models.py --res12 PRO`（GLN 同理） | `selection.json`、`start_v2m{1,2}.pdb` |
| 第二段：突变（重原子） | 181，系统 python 3.14 | `python3 experiments/2026-10-11_public_data_execution_packet/scripts/build_mutant_modeller.py --all --start v2m1 v2m2`（在项目根目录下） | `inputs/modeller_v21/<样本>/heavy_v2m{1,2}.pdb`、`build_record.json` |
| 加氢 | 183 | `bash experiments/2026-10-11_public_data_execution_packet/scripts/add_h_reduce.sh`（全部样本；reduce `-NOFLIP -HIS -NUClear -Nterm25`） | `reduced_v2m{1,2}.pdb` |
| 命名核对与静态检查 | 181，openmm_dev python（脚本说明：只做 numpy 和 XML 解析，不建 OpenMM 体系） | `$PY experiments/2026-10-11_public_data_execution_packet/scripts/finalize_allh_models.py --all` | `start_v21m{1,2}.pdb`、`finalize_record.json` |
| prep（C2） | 183（v2.1 的 16 个模型在这里跑过；PBS 作业里会因为已完成而跳过） | `$PY scripts/prep_peptide_sample.py --sample <样本> --model-id v21m1 --protocol v2.1`（在项目根目录下） | `prepared/<样本>/v21m{1,2}/` |
| setup（C3）+ MD + strip | PBS | `p2_v21_validation.pbs`（V2/V3，1 ns）、`p2_v21_V5.pbs`（10 ns）、`p2_v21_ext20.pbs`（延长到 20 ns，并接着给 10–20 ns 打分）；三者都调用 `p2_v21_validation.sh` | `p2_v21_validation.log`、`prepared/`、`md/` |
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
