# 第一次分子冒烟（2026-10-11，fulda / RTX 5090）——已作废，留作缺陷证据

四步全部跑通，但 setup 的无约束最小化把受体 Glu25–Thr26 肽键由反式（−176.3°）翻成了顺式（−8.8°），1 ns 生产全程保持顺式。原因：PDBFixer 用软核力场补建的末端残基留有重叠（Glu25 CG–O 0.93 Å、Pro12 OXT–Ser11 O 1.13 Å），PDBFixer 自带的冲突检查不检查同一残基内的原子对；跨残基的重叠缓解循环失败时也不报错。
修复：setup 最小化分两段（第一段对 ω 和 CA 手性加约束，第二段无约束），并在准备与 setup 之后检查顺反/L–D 类别。见 `scripts/run_peptide_md.py`、`scripts/peptide_geometry.py` 和交接 §9。
这里的轨迹和诊断不进入任何分析。
