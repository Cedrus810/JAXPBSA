# 存档：JCP26 Table 4 的 25 个蛋白 —— gaussian_gap LPB / NPB 对照（2026-09-29 … 30）

结论和解读见 [RESULTS.md](../../RESULTS.md) §18.19（LPB）、§18.20（NPB + Kirkwood）、§18.21（全 25 个 NPB、σ 扫描）。
这里只存原始输出，保证数字可追溯。

- 算力：全部在 fulda（RTX 5090）上跑，命令见下表。原来的文件在 `data/md/`（gitignored，NFS 共享），这里是副本。
- 代码：基线提交是 7256814，在它上面还有未提交改动，快照见 `code_state_vs_7256814.diff`（jaxpbsa/tests/scripts 相对 HEAD 的 diff）。
- 脚本：`scripts/` 里是**最终版**。早期的几轮是用当时的旧版脚本和旧版 Newton 跑的，下面各行分别注明。
- 输入：`inputs_pdb25_prep.tar.gz` 是 `data/pdb25/prep/`，由 `scripts/prep_pdb25.py` 从 RCSB 原始 PDB 生成（PDBFixer + ff14SB + mbondi2，
  固定 seed），其中含 `1C75_cym`、`1TQG_altB` 两个变体。原始 PDB 没有存，可以从 RCSB 重新下载。
- 校验：`SHA256SUMS`。

## 各轮（按时间）

| # | 文件（runs/） | 内容 | 代码状态 | 能不能用 |
|---|---|---|---|---|
| 1 | `pdb25_conv_fulda.log`、`pdb25_conv.npy` | 25 个，**LPB**，σ=0.93，h=1/0.5/0.25 ×3 相位 | 旧脚本（整行失败即丢） | 可用：LPB 基准（§18.19） |
| 2 | `pdb25_npb_fulda.log`、`pdb25_conv_npb_14of25.npy` | 25 个，NPB，σ=0.93 | Newton 是 fp32 外层，有停滞出口，10·tol 验收 | **作废**：只有 14/25，其余 11 个整行丢失 |
| 3 | `pdb25_npb_retry_fulda.log`、`pdb25_conv_npb_retry.npy` | 第 2 轮失败的 11 个，NPB，带失败诊断 | 同 #2 | 仅诊断：暴露两类失败（爆掉 / fp32 残差底线） |
| 4 | `pdb25_var_fulda.log`、`pdb25_conv_var.npy` | `1C75_cym`、`1TQG_altB`，LPB | 同 #2 | 可用：1C75 LPB ratio 0.560 → 0.927 |
| 5 | `pdb25_npb_var_fulda.log`、`pdb25_conv_npb_var.npy` | 同上，NPB | 同 #2 | 作废：1C75_cym 爆掉 |
| 6 | `kirkwood_npb_fulda.log` | Kirkwood NPB，a=20，h=1…0.2 | **最终版 Newton** | 可用（从终端转录） |
| 7 | `pdb25_npb_mp_fulda.log`、`pdb25_conv_npb_mp.pkl` | 25 个 + 2 个变体，NPB，**σ=0.93** | 最终版 Newton，max_newton=30 | 可用：2FDN h=0.25 有 2/3 相位撞 30 步上限 |
| 8 | `pdb25_scan_{base,s070,s080,s100,s110,gcsraw,bondi}.log`、`pdb25_conv_npb_scan_*.pkl`、`pdb25_scan_report_h0.5.txt` | 8 个蛋白，NPB，h=0.5 ×2 相位，单因素扫 σ / GCS 核 / 半径 | 最终版，max_newton=60 | 可用：σ 扫描（§18.21） |
| 9 | `pdb25_npb_s100_fulda.log`、`pdb25_conv_npb_s100.pkl` | 25 个（1C75 用 CYM 版），NPB，**σ=1.0** | 最终版，max_newton=60 | **主结果**：22/25 中位差 1.7% |
| 10 | `pytest_fulda.log` | 全量测试 53 passed 1 skipped（zsasa SIGILL） | 最终版（含 gap_sigma=1.0） | — |

命令（在仓库根目录下，`P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python`）：

- #1：`$P data/md/pdb25_conv.py`
- #2：`$P data/md/pdb25_conv.py --npb`
- #7：`$P data/md/pdb25_conv.py <25 个 + 1C75_cym,1TQG_altB> --npb --tag=mp`
- #8：`$P data/md/pdb25_conv.py 1CBN,1IUA,1VB0,2FMA,1ZZK,1M1Q,1TG0,2FDN --npb --hs=0.5 --phases=2 --tag=scan_<t> [--sigma=… | --gcs_raw | --radii=bondi]`，
  汇总用 `$P data/md/pdb25_scan_report.py h=0.5`
- #9：`$P data/md/pdb25_conv.py <25 个，1C75→1C75_cym> --npb --sigma=1.00 --tag=s100`
- #6：`$P data/md/kirkwood_diffuse.py --npb`

## 文件格式

- `.pkl`（#7 起）：`{"hs", "npb", "phases", "result"}`，其中 `result[pdb][h][phase] = {G, converged, newton_iter, cg_iter, residual,
  residual_ref, ms}`，出异常时是 `{error}`。G 的单位是 kcal/mol，ms 是单次求解时间（第 0 相位含编译）。
- `.npy`（#1–5）：object 数组，每行是 `(pid, [G(h) …], Table4 GEC 三列)`，只含整行成功的蛋白。
- Table 4 的原文值在 `scripts/prep_pdb25.py` 的 `TABLE4` 里。
