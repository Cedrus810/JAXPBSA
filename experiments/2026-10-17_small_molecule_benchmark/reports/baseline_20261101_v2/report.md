# P1 基线对照 v2：BRD4-5 改用阿普唑仑（2026-11-01）

取代 `reports/baseline_20261101/` 中 BRD4 部分（v1 保留）。协议 `protocol.json`（分组与 v1 相同，`changes` 列出差别）。Hou 部分不变（Hou B2 自检仍复现 `ranking.json`）。命令：

```
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python experiments/2026-10-17_small_molecule_benchmark/scripts/compare_baselines.py \
    --out experiments/2026-10-17_small_molecule_benchmark/reports/baseline_20261101_v2
```

BRD4-5 的 B0 取新 `ligand_in.sdf`（重原子 22、可旋转键 1，与旧版相同），B1 −21.08（yayoi28 setup 的最小化结构），B2 默认 −20.64、h0.4 −20.43。

| 组 | n | B0 尺寸 | B1 起点 | B2 MD | B2 h0.4 |
|---|---:|---:|---:|---:|---:|
| BRD4 ITC 六条（跨来源） | 6 | 0.26 | 0.43（10/5） | 0.26（9/6） | 0.20（8/7） |
| BRD4 定量八条（跨测定） | 8 | 0.34 | 0.62（20/8） | 0.57（20/8） | 0.55（19/9） |

- B2 相对 B1 反转的 BRD4 对只有两对：4/5 改对（B2 分差 / SE 6.9），5/8 改错（4.3）。BRD4 上仍看不到 MD 平均带来的次序增益；v1 的结论不变。
- B1 对最小化本身敏感：同一份 BRD4-5 prmtop，183（2080 Ti）冒烟的最小化结构 B1 = −24.11，yayoi28（3090）为 −21.08，相差 3.0（最小化后能量 −580 423 vs −576 689 kJ/mol）。这一量级的单结构不确定性对所有 B1 都适用，v1 没有量化，原因未查。

## B1 对最小化的敏感性（2026-11-01 补）

把 17 条起点的 setup 最小化在另一块卡上重做一遍（`b1_repeat.sh`，PBS 22221，yayoi37 RTX 5080，24 min，17/17 有效），用同一冻结默认网格评 B1。比较脚本包内 `scripts/compare_b1_repeat.py`，产物 `b1_repeat_{samples,groups,pairs}.csv`。

- 重做减正式：|差| 中位 0.81，最大 5.00。BRD4 十条都在 ±1.5 内；Hou a4 +3.26、a5 +3.90、a7 +5.00，其余 ≤0.94。
- BRD4：两次 B1 的样本次序相同（Spearman 1.000），ITC 六条与定量八条的 ρ、成对计数完全一样。上面的 BRD4 结论不受影响。
- Hou：a1–a4 的 B1 ρ 从 0.00 变为 0.40（a2/a4 的 B1 次序翻转；正式 B1 中这一对只差 0.27），全七条 0.36 → 0.32。B2 相对重做 B1 反转的对是 a1/a3、a2/a3、a5/a6、a3/a7，仍全部朝实验方向、没有反向改坏的；与正式 B1 相比，a2/a4 退出、a3/a7 进入。
- 所以 B1 单结构有最大约 5 kcal/mol 的最小化不确定性（Hou 上更大），"Hou 上 B2 次序优于 B1、BRD4 上两者相近"的结论在两次最小化下都成立；具体哪几对被 B2 改动依赖于 B1 用的是哪次最小化。183 冒烟里 BRD4-5 差 3.0，在这次（yayoi28 对 yayoi37）只差 0.03，那次不代表 BRD4 的一般情况。
