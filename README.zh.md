# JAXPBSA

用 JAX 实现的 Poisson–Boltzmann 静电解算器，配 OpenMM 侧的对接/IO（MM 交叉项、参数转换等）。
目前是一个**暂存（staging）仓库**：代码和文档先推上来存个档，打包、CI、示例数据等后期再补。

## 现状

- `jaxpbsa/` — 核心包：PB 解算（`pb/`）、MM 交叉项（`mm/`）、OpenMM 系统/参数导出（`openmm_io/`）、benchmark
- `scripts/` — 体系准备（S4 复合物）、MD 运行与分析、基准/性能分析脚本
- `tests/` — 常数、radii、MM cross、PB 的单元测试
- `5080.csv` — RTX 5080 上的基准结果（jacobi / multigrid 预条件，fp32/fp64）
- `DESIGN.md` / `RESULTS.md` / `JAXPBSA_OpenMM_Plan.md` / `CHANGELOG.md` — 设计、验证结果、计划与变更记录

## 未包含

`data/`（结构、prepared 体系、MD 轨迹等，约 3 GB）不入库；可由 `scripts/prep_s4.py` 等重新生成或从 RCSB 拉原始 PDB。

## 环境

开发在 CPU 上，实际计算跑在用户的 GPU 机器（openmm_dev conda 环境）。依赖见 `pyproject.toml`。

## 单 GPU 在线调度

`AsyncPBSAReporter.from_system(...)` 在常驻子进程内构造、预热分析器。OpenMM
reporter 只复制并提交溶质坐标，后台进程计算并及时写 CSV，结束时排空已提交帧。
同卡 MPS 可给 PBSA worker 单独设置低优先级和执行线程限额，不改 OpenMM 的配置。
队列满默认报错，不静默丢采样帧。使用方法和对照基准见
[在线调度说明](docs/ONLINE_SCHEDULING.md)；5000 步间隔 MD overhead，MPS 30% 限额：2080 Ti 15% → ~10%（RESULTS §18.22，5 万步），4090 11.8% → 3.9%（§18.23，20 万步）。

## 运行测试

```bash
pytest
```

## 许可证

Copyright © 2026 Cedrus810。本项目以 [MIT License](./LICENSE) 发布。
