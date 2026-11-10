# P2 BRD4 分组统计 v2：BRD4-5 改用阿普唑仑（2026-11-01）

取代 `reports/brd4_statistics_20261101/`（保留不动）的 BRD4-5 处理。协议 `protocol.json`（`changes` 字段列出与 v1 的差别），标签 `label_audit.csv`（只改 BRD4-5 行）。命令：

```
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python experiments/2026-10-17_small_molecule_benchmark/scripts/brd4_statistics.py \
    --out experiments/2026-10-17_small_molecule_benchmark/reports/brd4_statistics_20261101_v2
```

BRD4-5 来源：独立包 `brd4_5_alprazolam_20261101/`（`docs/BRD4_5_REPREP_PLAN_20261101.md`；作业 22051，yayoi28 RTX 3090，3 × 5 ns，两套网格各 1500 帧全部有效）。样本分 默认 −20.64 ± 0.52，h0.4 −20.43 ± 0.72（旧二氢形式 −18.09 / −16.89）。脚本改动后对 v1 输入逐位复现 v1 产物。

## 统计（ρ；成对 一致/不一致；精确 p；MD 重采样 ρ 2.5–97.5%，1 ns 块）

| 组 | n | 默认 | h0.4 |
|---|---:|---|---|
| ITC 15 °C 同方案 5–9（主） | 5 | 0.40；7/3；0.52；0.30…0.40 | 0.30；6/4；0.68；0.30…0.40 |
| ITC 全部（+3） | 6 | 0.26；9/6；0.66 | 0.20；8/7；0.71 |
| 定量跨测定 3–10 | 8 | 0.57；20/8；0.15 | 0.55；19/9；0.17 |
| 主组去掉 5（= v1 主组） | 4 | 0.40；4/2；0.75 | 同左 |

## 涉及 BRD4-5 的对（默认网格；分差 / 合并副本 SE）

- 5/7（同篇同方案 ITC）：与实验一致，两套网格都是（2.93 / 0.60）。
- 5/9、5/10、4/5：一致；5/8：相反（BRD4-8 仍偏弱，见 v1 §3）；3/5：相反。
- 5/6：默认网格一致（1.08 / 0.75，比值 1.4），h0.4 相反（−0.78 / 0.81，比值 1.0）。两套网格的主组差别全来自这一对，两种网格下它都没有分开。

## 限制

同 v1 §4。另：BRD4-5 的配体参数由 AmberTools 24.8 生成，其余 9 条用上游参数（版本未知）；主组 5 条，精确 p ≥ 0.52。

## 原文核对（2026-11-02，用户提供的论文与 SI）

用户把 13 个文件放在 `/home/ruigengji/JAXPBSA/`（未移动）。逐条结果 `label_verification_20261102.csv`；统计用的 `label_audit.csv` 与结果未改。

- **BRD4-5–9 全部核到原表**：BMC 2012 Table 1、PNAS SI Table S2/S3、Nature SI Table 3，ΔG 与维护者表一致；均为 15 °C、50 mM HEPES（pH 7.4 或 7.5）、150 mM NaCl、反向滴定。主组"ITC 15 °C 同方案"的条件由原文确认。
- 维护者表的"ΔG SEM"在这 5 条上都是原表 ΔH 的误差，不是 ΔG 的误差；本项目没有用它做统计。
- **BRD4-3**：Angew SI Table S4 的 XD1（colchiceine），KD 46 μM、ΔG −5.9（维护者 −5.95）；**25 °C**、20 mM HEPES pH 7.5，与主组不同实验室和温度，留在主组外是对的。
- **BRD4-10**：ACS MCL Table 1 化合物 3，BRD4 BD1 IC50 0.026 ± 0.01 μM（AlphaLISA）；−10.41 = RT ln IC50 在 300 K 的值，0.23 是 IC50 的 SD 传递。是 IC50 换算，不是 Kd。
- **BRD4-4：所引论文里找不到 −6.36**。Fish 2012 正文只说片段 7、8 对 BRD4 "30–90 μM"（298 K 下 −6.2…−5.5），−6.36 对应约 22 μM；正文该句引用 Philpott 2011（未读）。BRD4-4 只在探索组"定量跨测定"里，该组结果需带此限制。
- **BRD4-1/2**：Vidler 2013 Table 3，化合物 8a inactive @ 250 μM、化合物 4 31% @ 250 μM（维护者写 32%）。删失次序（BRD4-2 强于 BRD4-1）成立。
