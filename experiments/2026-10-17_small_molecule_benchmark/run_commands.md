# 运行命令（小分子基准）

状态以 `docs/BRD4_HOU_EXECUTION_HANDOFF.md` §7 为准。本文件只列**已实现**的命令；标"未在节点验证"的还没有真实跑过。

**2026-10-24 工程入口更新：** 见 `docs/ENGINEERING_REPAIR_HANDOFF_20261024.md`。
新评分目录先写 protocol.json，同输入/同参数可续跑；历史未绑定的 frames.csv / sensitivity.csv
保留但不追加，重新评分需给新 `--out`。在线 MD 的完成还要求已提交 step 的评分完整覆盖；
缺分时明确报错，保留 MD，并可在新离线评分目录回放。已有缺分运行不能因此改称完整在线运行。
第二轮起，setup 跳过或使用前会核对来源与 R/L 选择；新在线运行的 NPT pilot 与分析参数绑定，
续跑时不要替换 pilot。时间参数必须对应整数个积分步，在线 NPT 至少产生两帧 pilot。
历史来源记录的限制及新增失败提示见工程交接 §6。

## BRD4（上游 Amber 输入，模型 id `mobley_amber`）

在项目根目录 `/home/ruigengji/JAXPBSA` 下，用 openmm_dev 的 python：

```bash
python scripts/run_smallmol_md.py setup --sample BRD4-3 --platform CUDA --device-index 0
python scripts/run_smallmol_md.py run   --run-id BRD4-3_mobley_amber_rep01 --platform CUDA --device-index 0 --prod-ns 5
python scripts/run_smallmol_md.py strip --run-id BRD4-3_mobley_amber_rep01
python scripts/score_smallmol_replica.py --run-id BRD4-3_mobley_amber_rep01 --start-ns 0 --end-ns 5
```

- 每条命令加 `--dry-run` 只打印计划，不导入 OpenMM/JAX。
- MD 续跑：保持身份、输入、种子与协议，同一命令重跑；`--prod-ns` 调大就是延长。新绑定的评分目录也可同命令续跑；历史评分兼容性见文首。
- 冒烟作业：`qsub brd4_smoke.pbs`（yayoi07，16 核 / 2 GPU / 50 GB；BRD4-3 与 BRD4-9 各占一块 GPU；日志 `brd4_smoke_<sample>.log`）。**2026-10-18 在 yayoi07 验证通过**（交接 §7.5）。
- 首轮全量：`qsub brd4_full.pbs`（yayoi07，32 核 / 4 GPU；30 副本 × 5 ns + strip + 评分；日志 `brd4_full_*.log`；交接 §7.6），2026-10-18 跑完（§7.7）。
- BRD4-1/5/6/7/10 首轮评分被整数电荷检查拦下（交接 §7.4）；检查已改为 1e-2 e，补评作业 `qsub brd4_rescore.pbs`（yayoi04，交接 §7.8），2026-10-19 验证通过。

## 同帧数值对照（交接 §7.9）

```bash
python scripts/amber_reference_smallmol.py --run-id BRD4-3_mobley_amber_rep01 --frames 124,249,374   # CPU
python scripts/grid_sensitivity_smallmol.py --run-id BRD4-3_mobley_amber_rep01 --frames 124,249,374  # GPU
```

- 冒烟：BRD4-3 rep01 帧 124，2026-10-19 在 183 上验证通过（交接 §7.9）。183 上直接调用：`/home/ruigengji/miniforge3/envs/openmm_dev/bin/python scripts/...`（脚本会自己设置 AMBERHOME 和 PATH）。

- 全量：`qsub ref_full.pbs`（yayoi30；30 副本 × q3 三帧；Amber 在 CPU、网格在 GPU，两部分同时跑；交接 §7.9）。2026-10-21 已完成；2026-10-24 只修正对照 CSV 时间标签，能量未变。

## Hou α-thrombin a1–a7（模型 id `ph74_gaff2`，在线 MM/PBSA；交接 §7.12）

```bash
# 准备（CPU；AmberTools 在 openmm_dev 里，脚本自己设置 AMBERHOME/PATH）
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B experiments/2026-10-17_small_molecule_benchmark/scripts/prepare_hou_thrombin.py --sample all --jobs 7
# MD + 在线评分（每条命令都要带 --inputs）
IN=experiments/2026-10-17_small_molecule_benchmark/manifests/hou_inputs.csv
python scripts/run_smallmol_md.py setup --inputs $IN --sample a1 --target hou --model-id ph74_gaff2 --platform CUDA --device-index 0
python scripts/run_smallmol_md.py run   --inputs $IN --run-id a1_ph74_gaff2_rep01 --platform CUDA --device-index 0 --prod-ns 5 --online --mps-percentage 30
python scripts/run_smallmol_md.py strip --inputs $IN --run-id a1_ph74_gaff2_rep01
```

- 准备：2026-10-23 在 183 上 7/7 完成。
- 在线路径冒烟：`smoke_hou_online/smoke.sh`，2026-10-23 在 180 上通过。要先启动 MPS daemon；`--mps-percentage 0` 表示不用 MPS。在 180 上必须用 NFS 上的 env，180 本地的 openmm_dev 只有 CPU jaxlib。
- 全量：`qsub hou_online.pbs`（yayoi26，1 × 3090，串行 21 个副本，walltime 24 h）。2026-10-24 在 yayoi26 上跑完（作业 18350，10.2 h）。
- 汇总：`python -B scripts/summarize_hou_online.py`（在 183 上运行）→ `reports/hou_online_*`。

## 未实现

描述符基线；Hou 的同帧网格 / Amber 对照。BRD4 首轮汇总：`scripts/summarize_brd4_round1.py`（在 183 上运行，181 的 NFS 有滞后）。
