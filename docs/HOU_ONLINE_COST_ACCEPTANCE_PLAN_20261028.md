# Hou 同卡在线成本与恢复验收：计划（2026-10-28，Claude 起草，未执行）

## 必读（开工前全部读完）

| 路径 | 为什么要读 |
|---|---|
| `/home/ruigengji/JAXPBSA/CLAUDE.md` | 冻结基线、生产环境、PBS 选卡写法、4090/5090 的使用限制。 |
| `/home/ruigengji/JAXPBSA/docs/ONLINE_SCHEDULING.md` | 同步、异步、MPS 三种在线方式的语义、积压与排空规则、MPS 核对（`MPS_VERIFIED`）。 |
| `/home/ruigengji/JAXPBSA/RESULTS.md` §18.22、§18.23、§20.1、§20.7 | 已有的同卡实测（S4 肽，2080 Ti / 4090）；Hou 生产在线的实测（3090，256–298 s/ns，没有 baseline）；Hou 网格结论。 |
| `/home/ruigengji/JAXPBSA/docs/ENGINEERING_REPAIR_HANDOFF_20261024.md` §6、§7.1 | 在线续跑的已修内容，以及 183 上中断/续跑冒烟的证据与剩余项。 |
| `/home/ruigengji/JAXPBSA/scripts/online_overhead.py`、`scripts/run_smallmol_md.py`（`start_online`，第 534–559 行）、`scripts/score_smallmol_replica.py` | 本批要改或复用的三个入口。 |

## 0. 来由与用户决定

- 内容来自用户转贴的他人 comment（2026-10-28）第 3 步：用选定精度，在同卡、同体系、同评分间隔下比较 baseline、同步、异步、异步 + MPS；报告 MD 吞吐、评分覆盖、延迟、积压、结束排空和冷启动成本；工程验收再补同 pilot 回放、真实 warm-start 失败恢复和生产条件下的恢复。
- 用户原话（2026-10-28）："是的"（回答"要我接着写第 3 步（同卡在线成本验收：baseline、同步、异步、异步 + MPS）的计划表吗？"）。
- 报告用默认还是 h0.4，仍由用户定。

## 1. 要回答的问题，以及比的是什么

| 问题 | 量 | 对照 |
|---|---|---|
| 在线评分让 Hou 的 MD 慢多少 | 同卡、同体系、每 5000 步（10 ps）一帧时的 MD 吞吐 ns/day；overhead = 1 − 吞吐 / baseline 吞吐 | 同一个作业、同一张卡、同一个 MPS daemon 下的 baseline；baseline 在开头、中间、结尾各测一次，它们的离散就是噪声底 |
| 能不能跟上 | 评分覆盖（完成帧数 / 提交帧数）、最大延迟（对比每帧对应的 MD 墙钟间隔）、积压（`max_pending=2` 时有没有报错） | 同上 |
| 结束和启动要花多少 | 排空 `drain_s`；worker 冷启动（编译缓存为空）和命中缓存时的预热时间 | 同上 |
| 生产入口的实际成本 | `run_smallmol_md.py run` 加和不加 `--online`，生产阶段的 s/ns | RESULTS §20.1 的 256–298 s/ns（3090，在线 + MPS 30%）缺 baseline，这一项补上 |
| 在线与离线是否一致 | 同一帧、同一 pilot、同一网格下，在线减离线的 ΔG_MMPBSA 及各分项 | 期望只差 GPU 浮点不可复现的量级（§16.6）；§7.1 冒烟是在 pilot 不同的条件下比的（ΔG_PB +0.004），这次用同一个 pilot 重比 |
| 生产条件下中断能不能续上 | 在节点上用生产入口跑在线 MD，第 2 个 chunk 后发 SIGTERM，用原命令重跑，再跑一次 | `complete: true`；已提交的 step 无缺无重；`online_valid: true` |

## 2. 本批不做，以及原因

- **真实 warm-start 失败恢复**：在线路径不使用 warm start（`jaxpbsa/online.py:69` 的不做清单："triplet warm start … 1.08× 不值得"）。warm start 只在离线的 `solve_trajectory` 里（`jaxpbsa/pb/energy.py:220`、`:449`）。它不属于在线交付的验收，需要的话另立一项。
- 不测 MPS 的其它档位：§18.22/§18.23 已经定了 30%。不测 10000/25000 步的间隔：Hou 生产用的是 10 ps = 5000 步。
- 不改 `jaxpbsa/`，不改默认参数，不改生产入口的数值配置。
- 不用 4090 / 5090。

## 3. 代码改动（只在脚本层）

1. `scripts/online_overhead.py`：
   - 加 `--hou-run-id <run_id>`：从包里读 `setup/system.xml`、该副本的 `npt_end.xml`（坐标、速度、盒子）、`npt_solute_A.npy`（pilot，ref 取 pilot[-1]，与生产一致），以及 setup_record 里的溶质和配体范围。
   - 与生产对齐：加 MonteCarloBarostat（1 atm，298.15 K，OpenMM 默认频率）和 CUDA mixed 精度（生产入口 `run_smallmol_md.py:104`）。不给 `--hou-run-id` 时，S4 路径不变。
   - 加 `--grid {default,h0.4}`：只往 `analyzer_kwargs` 里传 `h/h_lig`（0.4/0.2），和 `score_smallmol_replica.py` 的 `GRIDS` 相同。
   - RESULT 行加三个字段：提交帧数、完成帧数、`max_pending`。
2. `scripts/score_smallmol_replica.py`：加 `--pilot {window,npt}`，默认 `window` 不变。`npt` 时 pilot = 该副本的 `npt_solute_A.npy`，ref = pilot[-1]，与在线构造相同。输出写到新目录 `scores/replay_npt/...`，不碰现有 scores。
3. 生产入口不改。B 段用一个独立的小包 `experiments/2026-10-17_small_molecule_benchmark/acceptance_online_20261028/`：自己的 `manifests/runs.csv` 放两个新 run id；`md/hou/a1/ph74_gaff2/setup` 软链接到正式 setup（run 只读 setup，`run_smallmol_md.py:201–216` 的写入只在 setup 子命令里）。这样正式的 `runs.csv` 和 21 个生产副本都不动。

## 4. 运行内容（一个作业，一张卡，三段串行，避免互相干扰计时）

体系：Hou a1，40 794 原子，从 rep01 的 NPT 终态出发。

| 段 | 内容 | 规模 |
|---|---|---|
| A 成本 | `online_overhead.py --hou-run-id a1_ph74_gaff2_rep01 --steps 200000`，每帧间隔 5000 步。配置：baseline；同步；异步不限额；异步 + MPS 30%（优先级 1）。后三种各用 default 和 h0.4 两套网格。每个在线配置都跑 `--order openmm` 和 `jax` 两次；baseline 在开头、中间、结尾各测一次（脚本里 baseline 只有一种顺序，它不建 JAX） | 6 个在线配置 × 2 + baseline 3 = 15 个进程，每个独立子进程。全程在同一个 MPS daemon 下，含 MPS 的配置输出 `MPS_VERIFIED` |
| B 生产入口 | 小包里两个副本：`acc_base` 只跑 MD；`acc_online` 加 `--online --mps-percentage 30`。都用生产默认：NVT 100 ps、NPT 1 ns、生产 1 ns、10 ps 存一帧、100 ps 一个 chunk。`acc_online` 在 chunks_done ≥ 2 时由 wrapper 发 SIGTERM（按记录的 PID），然后原命令重跑到结束，再跑一次确认跳过 | 2 个副本，各 2.1 ns |
| C 同 pilot 回放 | `score_smallmol_replica.py --pilot npt --stride 20`，取每个样本的 rep01（7 个副本 × 25 帧），与在线 CSV 同帧对比 | 175 帧 |

编译缓存：作业用自己的新缓存目录。每套网格第一次启动 worker 时是冷启动，其后命中缓存，这两种预热时间分开报告。

## 5. 冒烟（代码改完后跑一次，183 的 2080 Ti，不占 PBS）

一个脚本，三段各缩到最小：
- A：`--steps 10000`，default 和 h0.4 各跑一次同步与异步 + MPS 30%，外加 baseline。
- B：小包 `acc_smoke`，`--npt-ps 100 --prod-ns 0.2 --online`，不做中断（中断续跑已在 §7.1 冒烟过）。
- C：`--pilot npt --end-ns 0.05` 测 5 帧。

记录每段的初始化时间和单步/单帧耗时，用来填正式计划表。

## 6. 计划表（正式运行，节点由用户指定，qsub 由用户执行）

| 项 | 内容 |
|---|---|
| 命令 | 新 wrapper `hou_online_acc.sh` 按 A → B → C 顺序串行；PBS 脚本照模板 `/home/kasuga/.claude/templates/pbs.zsh`，MPS daemon 的启停照 `hou_online.sh` 的写法（作业专用 pipe 目录，EXIT 时 quit） |
| 节点 | 最好是 3090 节点：与 Hou 生产同型号，B 段可以直接对照 §20.1。卡内对比在任何单卡上都成立。双卡节点只用一张卡，另一张空着，不让它干扰计时 |
| 预计耗时 | **大部分不知道，冒烟后按实测填。** 已知依据：Hou 在线生产 256–298 s/ns（3090，含评分）；worker 冷启动 69 s，命中缓存约 9–10 s（§19.1、§20.1）。粗估：A 段 18 × (200 000 步约 0.4 ns × 约 250 s/ns ≈ 100 s，加上初始化) ≈ 45–70 min；B 段 2 × 2.1 ns × 约 270 s/ns ≈ 20 min，加重跑约 25 min；C 段 7 × (初始化约 1–3 min + 25 × 0.36 s) ≈ 15 min。合计约 1.5–2 h，walltime 4 h |
| 能否续跑 | A：不能按配置跳过（每个配置是一次计时，重跑就是重测），整段重跑约 1 h；B：能（生产入口按 chunk 续跑）；C：能（评分入口按帧续跑） |
| 输出 | A：`reports/hou_online_cost_<日期>.log`（RESULT 行）；B：`acceptance_online_20261028/md/...`；C：`scores/replay_npt/hou/...`；汇总脚本运行后写 `reports/hou_online_acc_<日期>.csv` |
| 这批结果决定什么 | ① 论文里 Hou 在线评分成本的数字（overhead、延迟、排空、冷启动），以及交付时推荐的调度方式；② 网格从 default 换成 h0.4 的在线成本差；③ 在线结果和离线回放在同 pilot 下是否一致，能否直接当离线结果用；④ 在节点上中断能否无缺续跑 |

## 7. 结果怎么读（预先写定）

- overhead 和 baseline 的离散（三次 baseline 的极差）、`order` 差（§18.23 是 0.8%）并列报告。比这两者小的差别，不说"有差别"。
- 最大延迟和每帧对应的 MD 墙钟间隔并列；有积压报错就如实报告，不改 `max_pending`、不切到 `skip`。
- 回放差异和 §16.6 的 GPU 浮点不可复现量级并列；如果明显更大，先查网格尺寸和 ref 是否一致，不放宽比较。
- 不设新的验收阈值。

## 8. 执行记录

### 8.1 代码与冒烟（2026-10-28，183 2080 Ti，16:46–17:04）

- 改动：
  - `scripts/online_overhead.py`：加 `--hou-run-id`、`--grid`、`--cache-dir`；RESULT 行加 system、grid、expected、submitted、skipped、max_pending。不给 `--hou-run-id` 时 S4 路径不变。
  - `scripts/score_smallmol_replica.py`：加 `--pilot {window,npt}`。npt 时 pilot 只按路径绑定，不用 sha256；window 路径的 protocol 不变。
  - 小包 `acceptance_online_20261028/`（runs.csv 只放三个新 run id；setup 是软链接）。
  - `hou_online_acc.sh`、`hou_online_acc.pbs`。
  - 相关回归 `tests/test_online_overhead.py`、`tests/test_smallmol_preflight.py` 共 45 项通过，只作回归检查。
- 冒烟（`acceptance_online_20261028/smoke_smoke1028.log`）：183 CPU 负载约 35/80，耗时只作参考。
  - A（各 10 000 步）：
    - baseline 255.8 ns/day。
    - 同步、默认网格：每帧 299–308 ms；冷启动预热 269 s，命中缓存 15.8 s。
    - 异步 + MPS 30% + h0.4：两次运行都输出了 `MPS_VERIFIED`，2/2 帧完成，没有跳帧；每帧 1.25 s，最大延迟 1.55 s，排空 2.2–4.9 s；冷启动 424 s，命中缓存 17 s。
  - B（`accsmoke`，NPT 100 ps，生产 0.2 ns，在线 MPS 30%）：`complete`、`online_valid` 都是 true，20/20 帧，0 标记；worker 预热 66 s，最大延迟 0.62 s。生产每 100 ps 用 41–43 s（含在线评分），NVT 100 ps 用 26 s，NPT 100 ps 用 35 s。
  - C（a1 rep01，`--pilot npt`，5 帧）：网格与在线相同（193³ / 129×129×161）。离线减在线：ΔG_MMPBSA 在 −1.4e-3…+3.1e-4 之间，E_coul ≤3.5e-4，ΔG_SA 为 0。
- 预计（按 2080 Ti 冒烟的数字算上限；3090 更快）：
  - A：15 个进程 × 每个约 150 s（0.4 ns 按 256 ns/day 约 135 s，加初始化），再加两次冷启动约 12 min，合计约 50 min。
  - B：accbase 约 12 min；acconline 约 15 min（含中断后重启的预热）。
  - C：7 × 约 3 min ≈ 20 min。
  - 合计约 1.5 h，walltime 4 h。

### 8.2 正式运行（完成，结果见 RESULTS §20.8）

- 用户 2026-10-28 给出节点："yayoi10   : [----------------]               ( 0 / 32) free GroupA  AMD EPYC 7452/2.35GHz + RTX 3090 x 2, 128GB"（已删去终端颜色转义码）。作业 19854，17:42–18:57，0 FAIL，用的是 GPU 0。
- 汇总：包内 `scripts/summarize_hou_online_acc.py 19854`。
- 结论：
  - 默认网格的在线 overhead 是 3–5%，三种方式之间的差别在噪声内。h0.4 约贵一倍（7.5–11.6%）。
  - 全部配置 40/40 帧完成，没有积压；最大延迟 ≤1.14 s，小于 2.61 s 的帧间隔。
  - 生产入口加在线评分慢 5.2%；中断续跑无缺无重。
  - 同 pilot 回放 174/175 帧差 ≤3.8e-3，1 帧差 0.10，原因没查。

## 未确认（Claude 推断）

- 默认网格和 h0.4 的在线 overhead 差别可能很小（3090 离线每帧 360 vs 386 ms，D3 实测），但同卡争用下会怎样不知道。（2026-10-28 实测否定：h0.4 每帧约是默认的 2 倍，overhead 也约 2 倍；360 ms 那个默认值取自在线 CSV，是 MPS 限额下的耗时，不可比。见 RESULTS §20.8。）
- MPS 30% 在 3090 上是否仍是好档位，§18.22/§18.23 只测了 2080 Ti 和 4090；这批不扫档位，只测 30%。
