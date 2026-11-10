# P3 证据清单（W1）与完整成本表（W3）（2026-11-02）

计划：`docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md` §6；论文框架 `docs/ONLINE_MMPBSA_PAPER_OUTLINE.md` §2、§5、§8。本目录只整理已有产物，没有新的分子计算。

## 1. 主张/图表 → 证据（`claims.csv`）

9 条主张（框架 §2 的 7 条 + P1 基线 C8、P2 BRD4 统计 C9）、5 张图、2 张表，每行给出 RESULTS 小节、源文件、生成命令、配置与必须同时写出的限制。主张的数字以 RESULTS 为准。

## 2. 复现（`reproduction.csv`、`reproduction_runs.json`）

命令：

```
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python experiments/2026-10-17_small_molecule_benchmark/scripts/reproduce_reports.py \
    --mirror <仓库外的空目录> --out experiments/2026-10-17_small_molecule_benchmark/reports/evidence_20261102
```

在仓库外建镜像包（`reports/`、`scripts/` 为拷贝，其余为只读软链接），依次重跑 24 条已记录的汇总/统计命令（BRD4 首轮与 h0.4、同帧参照、D0/D1、Hou 网格、副本诊断、数值验收、在线验收、采样汇总、严格审计 ×2、排序、P1/P2 v1/v2、B1 汇总与重做、BRD4-8 分项），逐个比较重新生成的文件。2026-11-02 结果：24/24 命令 rc = 0；

| 结果 | 文件数 |
|---|---:|
| 与原件逐字节相同 | 73 |
| 只有路径字符串不同（数值差 0）：`sources.json` ×2、`statistics.json`（v2）、`ranking.json` | 4 |
| PDF 只差内嵌时间戳（同名 PNG 逐字节相同） | 3 |
| P2 v1 `statistics.json`：只差记录的 `brd4_statistics.py` sha256（v1 由加 `--out` 之前的脚本生成），统计值全同 | 1 |
| 本轮新建、镜像中不存在（`evidence_20261102/cost_*.csv`） | 5 |
| 没有可复跑的命令（下表） | 31 |

复现中发现并修正的两个脚本问题（修正后各自输出与原件逐字节相同）：

- `scripts/diag_hou_replicas.py`（§20.6）：读取副本全部帧，而 rep01–03 在 10-28 之后已延长到 10 ns、另加 rep04–06，h0.4 分数只覆盖 0–5 ns，运行报 KeyError。改为固定 rep01–03 与 (0, 5] ns。
- `scripts/summarize_hou_online_acc.py`（§20.8）：验收副本 `acconline` 在 10-28 19:27 被 `smoke_extend.sh` 当作延长冒烟续到 chunk 12，现状汇总会把这段算进去（chunks 12、max 444 s/ns）。改为只取验收时的前 10 个 chunk。§20.8 的数字不变。

没有可复跑命令的 31 个文件：

- 手写的协议/报告/标签表（11）：`baseline_20261101{,_v2}/{protocol.json,report.md}`、`brd4_statistics_20261101{,_v2}/{protocol.json,report.md,label_audit.csv}`、`label_verification_20261102.csv`。
- v2 目录里从 v1 拷来的参考副本（2）：`baseline_20261101_v2/{inventory.csv,start_scores.csv}`。
- 本目录自身（3）：`claims.csv`、`reproduction.csv`、`reproduction_runs.json`。
- 历史快照，按设计不再重生成（5）：`*.before_contractfix_20261024.*` ×3、`reference_q3_frames.before_timefix_20261024.csv`、`reference_timefix_q3_20261024.json`。
- 已停用的 `summarize_hou_online.py` 产物（3）：`hou_online_{replicas,samples}.csv`、`hou_online_ranking.json`（首轮 3 × 5 ns，后续以 §20.9–§20.10 为准）。
- 冒烟产物（2）：`grid_diag_d1_{,frames_}smoke_20261025.csv`。
- 其他记录（5）：`engineering_repair_validation_20261024.json`、`hou_thrombin_pdb_inventory.json`、`static_input_check.json`、`jaxpbsa_diff_vs_3737870.patch`、`hou_replica_diag_20261028.log`。生成方式在各自交接中，本轮未复跑。

## 3. 完整成本表（`cost_*.csv`，包内 `scripts/cost_table.py`）

只从运行记录抽取：setup_record.json、status.json（每个生产 chunk 取运行它的 context 的设备）、评分 summary.json（只记 host，按用户给的节点信息映射到 GPU）、prep_record.json。按设备分组，不跨设备或精度拼数。

**MD（mixed 精度，`cost_md.csv`）**

| 体系 | GPU | 生产 s/ns 中位（范围） | 生产墙钟合计 | NVT 0.1 ns | NPT 1 ns |
|---|---|---|---:|---:|---:|
| BRD4 离线 | RTX A4000 | 354（347–368） | 14.8 h | 28 s | 333 s |
| BRD4-5 重跑 离线 | RTX 3090 | 223（221–230） | 0.9 h | 17 s | 210 s |
| Hou 在线（MPS 30%） | RTX 3090 | 269（254–346） | 8.1 h | 19 s | 244 s |
| Hou 在线（MPS 30%） | RTX 5090 | 155（140–1020） | 14.6 h | 10 s | 130 s |
| Hou 在线（MPS 30%） | RTX 2080 Ti | 418（405–424） | 0.5 h | — | — |

Hou 的 NPT 1 ns 同时是在线网格定尺的 pilot。5090 上最大 1020 s/ns 的 chunk 对应 a5 rep05 的积压中断（§20.9）。

**在线评分（`cost_online.csv`）**：每帧 PBSA（MPS 30% 下的 worker 忙时间 / 报告数）3090 372 ms、5090 112 ms、2080 Ti 608 ms；最大延迟 3090 0.47 s、5090 8.5 s（a5 rep05 积压）、2080 Ti 0.72 s；worker 预热中位 36 / 5 / 119 s。5090 段的 97 个标记帧即 §20.9a 的越界帧。

**离线评分（`cost_scoring.csv`，ms/帧中位；初始化含编译）**：BRD4 默认 A4000 231、3090 92；BRD4 h0.4 3090 239；Hou h0.4 3090 402；Hou 回放 3090 167；Hou 恢复（窗口 10 ns）2080 Ti 300；B1 单帧 4080 SUPER 113（BRD4）/ 206（Hou）、5080 100 / 147。初始化中位 14–121 s，冷启动与命中缓存混在同一列，单列的冷/热数字见 RESULTS §20.8。

**准备（`cost_prep.csv`）、setup（`cost_setup.csv`）**：逐条记录。

**缺测（不补造）**：PBS 排队等待、文件 I/O 与 NFS 同步、strip 耗时、Amber 参照的逐帧成本（只在日志与 RESULTS §19.3/§20.4 文本中）、MODELLER 补建（§7.10 记 ~3 s/模型）。在线开销与 MD-only 基线的同卡对比只有 §20.8（Hou、3090），其他设备上没有同卡基线，不计算 overhead。

## 4. 下一步

W4：按框架 §5 作论文图（F2–F5）与表（T1、T2），并写 Methods/Results 初稿；W5：复现入口（本复现驱动 + 环境说明 + 最小示例）。软件能力对照（框架 W3 第二项）尚未开始，需要先查文献与工具文档。
