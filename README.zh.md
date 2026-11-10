# JAXPBSA

用 JAX 实现的 Poisson–Boltzmann 静电解算器，配 OpenMM 侧的对接/IO（MM 交叉项、参数转换等）。
目前是研究代码，已完成在线方法论文的本地审阅交付：[正文](docs/ONLINE_MMPBSA_PAPER_DRAFT.md)、[SI](docs/ONLINE_MMPBSA_PAPER_SI_DRAFT.md)、[五张主图](docs/paper_assets/README.md)、[收尾记录](docs/PAPER_CLOSEOUT_20261106.md)。完整原始数据的公开存档尚未执行。

最小离线复现（numpy/scipy/matplotlib；不做分子计算）：

```bash
python scripts/reproduce_paper.py --out experiments/2026-11-06_paper_closeout/reproduction
```

源码/报告快照、环境和完整计算所需材料的边界见 [复现说明](docs/PAPER_REPRODUCIBILITY.md)。

## 现状

- `jaxpbsa/` — 核心包：PB 解算（`pb/`）、MM 交叉项（`mm/`）、OpenMM 系统/参数导出（`openmm_io/`）、benchmark
- `scripts/` — 体系准备（S4 复合物）、MD 运行与分析、基准/性能分析脚本
- `tests/` — 常数、radii、MM cross、PB 的单元测试
- `5080.csv` — RTX 5080 上的基准结果（jacobi / multigrid 预条件，fp32/fp64）
- `DESIGN.md` / `RESULTS.md` / `JAXPBSA_OpenMM_Plan.md` / `CHANGELOG.md` — 设计、验证结果、计划与变更记录

## 未包含

原始 prepared 体系、MD 轨迹与全部逐帧记录不在精简审阅包；保留在完整项目的数据目录。重新从 PDB 制备不替代历史输入的精确复现。

## 环境

开发在 CPU 上，实际计算跑在用户的 GPU 机器（openmm_dev conda 环境）。依赖见 `pyproject.toml`。

## 单 GPU 在线调度

`AsyncPBSAReporter.from_system(...)` 在常驻子进程内构造、预热分析器。OpenMM
reporter 只复制并提交溶质坐标，后台进程计算并及时写 CSV，结束时排空已接受帧；默认退出超时 60 s，可重试 close，不强杀活动 worker。
同卡 MPS 可给 PBSA worker 单独设置低优先级和执行线程限额，不改 OpenMM 的配置。
队列满默认报错，不静默丢采样帧。使用方法和对照基准见
[在线调度说明](docs/ONLINE_SCHEDULING.md)；5000 步间隔 MD overhead，MPS 30% 限额：2080 Ti 15% → ~10%（RESULTS §18.22，5 万步），4090 11.8% → 3.9%（§18.23，20 万步）。

## 运行测试

```bash
pytest
```

## 许可证

Copyright © 2026 Cedrus810。本项目以 [MIT License](./LICENSE) 发布。
