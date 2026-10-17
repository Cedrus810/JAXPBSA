# 性能审查（2026-09-28）

全包通读（~2900 行核心代码 + RESULTS/DESIGN 既有实测）后的性能审查记录。基线口径：5090 binary 0.5 = 59.1 ms/帧（S4）/ 53.8（1YCR）；在线端到端 128.7 ms/帧（5080）；solver 占 82%、surface 8.1%、DST 5.2%（RESULTS §0.2/分阶段实测）。

**TLDR：性能纪律在同类项目里是顶级的，大头（fp32、MG 预条件、DST、静态 slice 形态学、非对称网格）都已吃掉且有实测背书。剩余空间集中在两处：`apply_operator` 的 scatter 写法（solver 占 82%，下一个真正的台阶），和编译时间（barrier-chunk 有现成实测背书可以推广）。另有 4 处确定的小免费赢，其中 `harmonic_smooth` 是自家「不要用 fori_loop」规律的第三处残留。在线 Python 侧和 MM/SA 没有值得动的东西。**

按「值不值得上 GPU 机器验证」排序。

---

## P1 · apply_operator 的 zeros + 3×scatter_add —— solver 82% 里的最大单一嫌疑

`operator.py:33-37`：`au = zeros_like(u)` 然后 3 次 `.at[1:-1].add(...)` scatter-add，最后再 `au + kbar2*u`。scatter-add 无法与输出侧融合：zeros 写一遍、`au` 被读改写三遍、最后再读改一遍。改成 pad + 三个差直接相加 + `kbar2*u` 的纯 elementwise 链，理论上一趟写完。

MG 路径每 V-cycle 约 30 次 apply × 11 次 CG 迭代 × 3 species，apply 占 solver 流量大头——如果成立，量级是 solver 1.3–1.4×、端到端 ~1.2×。

**按自家规矩（§0.16 教训）：先 microbench 再下结论。** `roofline.timeit` + `device_arrays`，同一张真实 ε map 上跑两个变体对拍（结果要逐位相同），半小时的事。如果赢了，`jacobi_diagonal`（同款 scatter）跟着改。

## P2 · 把 barrier-chunk 推广到形态学链，砍编译时间（与运行时正交的战场）

两个实测数据点背书：

- ses_level 的 461 偏移链每 32 偏移插屏障：**15.7 → 15.2 ms，不伤运行时**（§18.16）；
- dilate 当年 chunk=1 和 chunk=128 都是 0.65 ms（运行时对分块不敏感，§0.16）。

而 binary 首帧编译 34–52 s（5090 实测）、测试套件 206 s，都归因于形态学静态展开撑大图。

实验：dilate/erode/ion 链每 32/64 偏移插 `jax.lax.optimization_barrier`，量编译时间 + 稳态 + 逐位对拍。风险极低，收益是文档里反复抱怨的痛点。

## P3 · V-cycle 参数按 ms/帧重扫一遍

`mg_nu=2/2`、`coarse_sweeps=50`、`min_n=7` 都是按**收敛行为/迭代数**调的（energy.py 注释「V(3,3) 略优于 V(2,2)」也是迭代口径）。目标函数应是 ms/帧：V(1,1)/V(1,0) 每周期便宜 30–50%，CG 迭代数会涨，净效应只有测了才知道。一次 10 帧中位数的扫描在 GPU 机器上成本很低。

## P4 · h_lig 上探（0.25 → 0.3/0.35），先测再决定

配体紧盒 h=0.25 有 ~7.0M 节点，和 C/R 网格同量级 → L 求解 ≈ 每帧 1/3 时间；h 0.25→0.3 节点数 ×0.58。但核心结论 **δΔG_PB ≈ −δG_L** 意味着 L 的离散误差 **1:1 进 ΔG**（§18.14 里 G_L 0.25 vs 0.2 就差 1.1 kcal/mol），上探会更差。是拿精度换 10–15% 端到端还是不换，跑一组 3 相位误差对照再定。

---

## 确定的小免费赢（逐位不变或近零风险，可顺手做）

1. **`harmonic_smooth` 还是 fori_loop + dynamic_slice**（`pb/surface.py:119-125`）—— `dilate`/`_smooth` 同一个坑的第三处残留，DESIGN「trip count 是 host 常量时不要用 fori_loop」直接覆盖它。RESULTS §0.16 实测它 1.02 ms/次且当年就没被修；7 偏移静态展开预期 ~0.2 ms，C/R 两张满网格 + 配体一张 ≈ 每帧省 ~2 ms（5090 binary 59 ms 的 ~3%）。加法顺序不变，逐位相同。
2. **pcg_solve 的 dot 和更新里冗余 `*mask`**（`pb/solver.py:61`、`:82`）—— r/z/p/ap 在构造处都已 mask，dot 里再乘一次是每 CG 迭代 4 次多余的 7.2 MB bool 读。shell 上已是零，去掉逐位同。
3. **coulomb_boundary_values 用 fori_loop**（`pb/operator.py:139`）—— `nb` 是 host 常量，又一条自家规律；1.8 ms/帧（2.3%）里有一部分是逐轮谓词回传，静态展开 15 块预期省 0.3–0.5 ms。
4. **dh_boundary_values 每帧重算静态量**（`pb/operator.py:98`）—— `norm(shell_xyz - center)` 与帧坐标无关，可在 `make_frame_solver` 里 host 侧算好闭包进来。
5. 文档小抄：`operator.py:122` 写「atom_block 默认 64」，`PBParams` 实际是 128。

## 复核过、明确不用动的

- **在线路径**：§18.15 的争用测量已证明串行干净、开销与公式吻合；逐帧 ~6 次 sync 和 3 次 SA 派发相对 130 ms/帧设备时间是噪声，PB 占 91.5% 的结论成立。不做清单里的每条决策（SA 不融 jit、triplet 不 warm start、不做 async）复核过，同意。SA 的 golden_spiral 表每次调用重建 + H2D（每帧 3×11 KB）纯属仪式性开销，几十 µs，不值得动。
- **MM 项**：距离只算一次、参数预置设备、`sum(dtype=)` 归约——都已最优。
- **warm start / species 补齐 / 球分解 / 纸面流量论证**——文档已有否决结论，别重试。

## 两个未决项（已在追踪，确认方向对）

- fraction 5090 上配体盒 8.70 GiB 临时显存之谜：`data/md/fraction_mem_probe.py`（无屏障/32/8/无 crease 对照）是正确的下一步。
- 若 fraction 重回路线：crease 循环是 461 偏移 × 每次 4 张满网格读（src+yx+yy+yz），而 ok 的交线圆点其实是稀疏的（O(表面积/h²)，几千个）——scatter 式点云→网格距离变换理论上低 2–3 个数量级，但属于大改，真要做先 profile。binary 默认路径不受影响。

## 建议动手顺序

1. P1 microbench 与 P2 barrier 实验（两件都便宜、互不依赖、一个动运行时一个动编译）；
2. 小免费赢 1–3 顺手清掉；
3. P3/P4 各一次 10 帧扫描。
