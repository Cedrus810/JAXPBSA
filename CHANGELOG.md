# 本轮改动记录

对应两份 code review 的回应 + 后续实测。所有性能数字见 [RESULTS.md](./RESULTS.md)。

---

## 1. 正确性修复（全部是静默失败：不报错，只给看起来合理的错数字）

| # | 问题 | 位置 | 怎么暴露的 |
|---|---|---|---|
| 1 | PB 源项漏 `BJERRUM_VAC = 560.75` 因子 → G_PB 小 560 倍 | `pb/charges.py` | T1 Born 解析值 |
| 2 | `κ̄²` 与 `κ²` 混用 → 体相屏蔽长度差 √78.5 ≈ 8.9 倍 | `constants.py`, `pb/operator.py` | T2 Debye 拟合 |
| 3 | Dirichlet 边界写进算子 → CG 收到**仿射**算子，前提破产 | `pb/solver.py` | T1 给出 −512 而非 −82 |
| 4 | 雅可比对角元在 **8 个角点恒为 0** → `0/0=NaN` → `NaN*0=NaN` 污染内积 → `cond` 恒 False → **求解器一次迭代都不跑，静默返回 u0**。单帧路径靠 XLA 把 `x*0` 折成 `0` 侥幸没暴露 | `pb/solver.py` | `test_batch_invariance` |
| 5 | **停止判据用了预处理器范数** `sqrt(rᵀM⁻¹r)/‖b‖` 而非真残差 `‖r‖/‖b‖` → 换预处理器就换了判据，「相同 tol」不是相同精度 | `pb/solver.py` | code review |
| 6 | 真残差只**报告**不**执行**：超标也直接返回 | `pb/solver.py` | code review |
| 7 | 零源项 `0/0` → `NaN > tol` 为 False → 假收敛 | `pb/solver.py` | code review |
| 8 | `pcg_solve` 加 `precond` 形参后**位置参数错位**：`tol` 进了 `precond` | `pb/energy.py` | code review |
| 9 | `TER` 写在残基循环里 → 每个残基被当成独立链，肽键全断 | `scripts/prep_s4.py` | 代码复查 |
| 10 | NeRF 第一项符号错 → 键角做成 `180°−angle` | `scripts/prep_s4.py` | `_measure`/`_nerf` 往返自检 |
| 11 | 残基名与实际质子化态不符：`addHydrogens` 按自己的 pH 逻辑选态并**无视输入名字**，HIS90/96 名为 HIE 实为 HID | `scripts/prep_s4.py` | tleap FATAL |
| 12 | **制备管道不幂等**：三层随机性叠加，重跑一次 1835 原子全动（RMSD 0.78 Å） | `scripts/prep_s4.py` | 重跑后能量跳变 54 kcal/mol |
| 13 | **`warm=False` 并没有冷启动**：carry 被当成 `u_prev` 传了进去，等于"每帧从同一个给定初值出发"。作为 warm start 的对照组，它含了调用方的信息，**比较不公平** | `pb/energy.py` | 逐位断言的跨块测试 |

> 第 13 条的额外教训：**代码和文档不符时，先信代码。** 我在 docstring 里写了
> 「`warm=False` 时 initial_state 被忽略」，但实现并没有忽略它。
> 是测试（而不是复读文档）把这个差异暴露出来的 —— 所以断言要写成**逐位相等**，
> `rtol=1e-6` 这种松断言会让 2e-6 的差异溜过去。

### dtype 静默提升链（不报错，只是悄悄变回 fp64 且变慢一倍）

- `jaxpbsa/__init__.py` **强制开 x64**，覆盖调用方设置 → **两轮 fp32 测量全是 fp64 跑的**
- `grid.origin` 存的是 **np.float64 标量**（强类型，会提升 fp32；Python float 是弱类型不会）
- `jnp.zeros(shape)` / `jnp.where(m,1.,80.)` / `fori_loop` 初值 **不带 dtype 时在 x64 下默认 float64**
- `while_loop` carry 初值 dtype 不一致
- `.astype(fp64).sum()` **物化整个 float64 副本**（百万元素）→ **fp32 比 fp64 还慢**；应写 `sum(dtype=)`
- 静态参数**每帧重传设备**，占 MM 总时间一半

守卫：`test_dtype_is_not_silently_promoted`。

---

## 2. 性能改动

| 改动 | 收益 | 位置 |
|---|---|---|
| DST 直解参考方程（奇延拓 + `rfft`） | **2.14×** | `pb/dst.py`（新） |
| MG **作 CG 预处理器**（独立求解器在 ε 跳变 1:80 下发散） | **2.27×** @ h=0.5 | `pb/multigrid.py`（新） |
| fp32 数组 + **fp64 归约** | **2.0–2.96×** | 全局 |
| 矩形网格（按各轴包围盒） | **1.41×**，节点 −30.4% | `pb/grid.py` |
| 形态学 `dynamic_slice` → **静态 `lax.slice`**（traced 起点阻止 XLA 融合） | 形态学 **6.0×**，逐位相同 | `pb/surface.py` |
| 参考边界原子分块 | kernel **5.28×**，端到端 1.07% | `pb/operator.py` |
| Coulomb/LJ 共用一次距离计算 + 参数预置设备 | MM **2.1×** | `mm/` |
| tol 1e-7 → 1e-5（两档 G_PB 只差 0.002） | 1.8× | `pb/energy.py` |

**合计 1800 → 76.8 ms/帧（23.4×，单 species PB）。**

### 试过但无效 / 不可行（已写进代码注释，免得重试）

- pad 提到循环外 —— **零收益**，XLA 本来就 CSE 掉了
- 形态学分组展开（chunk 8–128）—— **零收益**，静态起点后 XLA 本就跨整条链融合
- 球分解 `ball(R) → m×ball(R/m)` —— **不可行**，格点上 `ball(a)⊕ball(b) ≠ ball(a+b)`（h=0.5/R=2.0 拆两次得 185 点 vs 目标 257）
- MG 最粗层轮数 —— **无差别**，4 与 50 都是 11 次迭代
- batching —— h=0.5 下 **0.63×（负收益）**；仅粗网格 + B≈2 有效

---

## 3. 新增能力

- **C/R/L 三体求解** `solve.triplet()`：ΔG_PB = G_C − G_R − G_L。参考解 **3 次降 2 次**
  （`u_ref_C = u_ref_R + u_ref_L`，**组合的是势不是能量** —— 参考能量含 R–L 交叉项）。
  R/L **补齐到 complex 的原子数**（屏蔽原子挪到盒外），三者**共用一份编译**。
- **轨迹接口** `solve.trajectory(..., initial_state=)` → `(每帧能量, final_state)`。
  势场只作 `lax.scan` 的 **carry**，不进输出（否则 T=10000 单 species 要 268 GiB）。
  **失败帧的势不传给下一帧**。
- **规范产物**：`prep_s4.py` 是转换器，输出 `S4_complex.cif` + `S4_system.xml` + 双 sha256；
  `load_canonical()` 加载并校验，下游**不再 import ForceField**。
- **跨设备 benchmark** `scripts/benchmark.py`：每配置独立子进程、关预分配、编译/稳态分离、
  **拒绝未收敛与非有限结果**；报告强制分「换卡不变」与「本机专属」两块。
- **分阶段 profile** `scripts/profile_stages.py`：与生产路径同边界、同 dtype、同能量公式。

---

## 3.5 跨设备验证（RTX 2080 Ti + RTX 5080）

**换卡不变量逐位一致** —— G_PB（complex/receptor/ligand/ΔG_PB）、迭代数、真残差
在两张卡上完全相同。两台机器从同一个 `S4_complex.cif`（sha256 `78d8db8e…`）出发。

由此定下两条：

- **`precond` 默认改为 `"auto"`**（阈值 `MG_NODE_THRESHOLD = 1.5 M` 节点）。
  MG 的交叉点在 **0.9–2.1 M 节点**之间，**两卡一致**；但幅度相反 ——
  粗网格上卡越快 MG 越吃亏（0.45× on 5080 vs 0.87× on 2080 Ti，粗层是启动延迟主导），
  细网格上则相反（2.81× vs 1.74×）。原先硬编码 `"mg"` 在 h=1.0 上**让程序慢一倍多**。
- **batching 的机制解释撤回**。先前说「batch 维在最前打散 stencil 合并访存」，
  它预测恒定惩罚（平的），符合 2080 Ti（一步跌到 0.62 后完全平）但**不符合 5080**
  （逐步下滑 0.87→0.73→0.70）。能确定的只有：两卡均为负收益、**显存都不是原因**
  （5080 的 B=8 只用 2.47 GiB / 16 GB）。

---

## 4. 测量方法上的教训

每条都曾让我得出**错误的物理结论**：

1. **同进程顺序跑多个配置** → 编译产物与显存单调累积 → batch 断崖数据作废（真因是 kernel 效率，不是显存）
2. **并发跑两个 GPU 进程** → JAX 预分配 75% → 伪 OOM，甚至 `bfc_allocator` 崩溃
3. **没分离编译时间** → 「scan 比单帧慢 7.6×」全是重复 tracing（实测 scan 与 Python 循环差 0.8%）
4. **测量运行期间改源码** → 前三行立方网格、最后一行矩形网格，报出假的 1.50×
5. **纸面流量反推优化空间** → 「算子融合还有 2–3×」无依据，已撤回
6. **判据不一致就比加速比** → 等于把「精度变松了」当成「计算变快了」
7. **h 选在 ≥1.5** → `ball_offsets(1.4,h)` 退化为单点，SES 塌回 vdW，G_PB 来回摆 600 kcal/mol
8. **拿绝对值的相对误差估差值的误差棒** → 会得到 28 kcal/mol 的假误差棒，实际 1.1（**差 25 倍**）

---

## 5. 被实测推翻的预判

| 原先的说法 | 实测 |
|---|---|
| surface 是真瓶颈 | 优化前 29.4%、优化后 **8.1%**；solve 占 **82.4%** |
| Jacobi-PCG 需 500–1500 次迭代 | **193 次**，标度 `(L/h)^0.64` |
| MG 可达 10–50× | **1.7–2.8×** @ h=0.5（每个 V-cycle ≈ 8 个 Jacobi 迭代的成本），且**粗网格上是负收益**（0.45–0.87×）→ 改 `precond="auto"` |
| warm start 是核心 novelty | **1.08×** —— 与 MG 此消彼长，MG 把迭代压到 11 后只能再省 1 次 |
| 「Batch first」 | h=0.5 下 **0.63×（负收益）** |
| 算子融合还有 2–3× | 依据不成立，已撤回 |


---

## 6. 下一步（按价值）

**ΔG_SA**（补完 ΔG_MM/PBSA 的最后一项）> **外部对拍 T6**（用 Amber `pbsa`，**不能用 GB**）
> **M0 轨迹** > **溶剂求解器访存效率**（占 82–85%，但必须按 profile 判断，
不能再用纸面流量反推）。

展开见 [RESULTS.md §9](./RESULTS.md)；明确不做/已证伪的方向见 §10。

---

## SA stage 2：JAX Shrake–Rupley（`sa/jax_sr.py`）

SA 现在**只有这一个实现**。zsasa 那个外部 CLI **不是后端也不是依赖** —— 它的唯一职责
是给 JAX 版做性能/结果对拍，所以从 `jaxpbsa/sa/zsasa.py` 挪到 `tests/zsasa_ref.py`，
和 APBS 同一个待遇（「外部进程调用，不进包」）。

`jaxpbsa.sa.sasa()` / `delta_g_sa()` 的 `backend=` 形参一并删掉：**一个实现的分派
就是多余的抽象**，留着只会让人以为 zsasa 是个可选运行时路径。

**不显式构造采样点。** 字面照抄 SR 要建 `[N,K,3]` 的 `p_ik = x_i + R_i·u_k` 再逐个测距；
代进埋藏判据展开后 `|p_ik − x_j|² = d_ij² + R_i² + 2R_i·u_k·(x_i − x_j)`，
判据变成一次 `u[P,3] @ disp[K,3]ᵀ` 的矩阵乘。顺带白捡：对 u_k 取极值得
`d < R_i + R_j` 才可能遮挡，**远邻居 / padding(d²=∞) / 自身项(disp=0) 全部自动失效，
不用写掩码**。

| 项 | 值 |
|---|---|
| vs zsasa 总 SASA（S4, P=960 / 4000） | rel **4.6e-5 / 5.8e-5** |
| vs 暴力法（显式建 p_ik 测距） | rel < 1e-6 = 逐点判定完全相同 |
| 单 species（2080 Ti, B=16） | 41 → **7.2 ms/帧** |
| 三 species ΔG_SA（B=16） | 131 → **13.7 ms/帧**，占每帧 6%（PB 222） |

**加了一道硬闸**：K 近邻是有损的，`K ≥ maxᵢ|{j : d_ij < R_i + R_max}|` 才保证不丢遮挡。
不满足**直接报错并报出该填多少**（`需要 ≥ 134 / 重试: k_neighbors=134`），一次重试必中。

自动探测试过两条，**都不安全**：扫 CIF 给 131（真空最小化态，不是 MD 系综构象），
MD 帧 0 给 126，而 10 ns 轨迹实际跨 **123–136** —— 单帧探不出系综上界，且 k 是
static argnum，逐块重探会反复触发重编译。默认 192（40% 余量，堆积密度限的量，
换蛋白不变）。

没有闸就是第 1 节那种「不报错、只给看起来合理的错数字」：误差恒为正且单调
（丢遮挡 ⇒ 暴露变多），k=32 时 +1.4%，没有任何提示。

## 探针半径的 source of truth

`probe_radius=1.4` 原来有 **3 份独立字面量**（`pb/energy.py` 的 `PBParams`、
`sa/jax_sr.py`、`sa/zsasa.py`）。两边用不同的 r_p **不会报错**，只会悄悄给出偏掉的
ΔG_MM/PBSA —— PB 的分子表面和 SA 的可及表面对应不同的溶剂，相加没有物理意义。

现在唯一来源是 `constants.PROBE_RADIUS`。
`tests/test_constants.py::test_pb_and_sa_share_one_probe_radius` 既比默认值、
也扫源码里的裸字面量。

顺带把边界写进 DESIGN.md §3.10：**SA 与 PB 只共享物理定义**（r_i 来自
`openmm_io.assign_radii`，r_p 来自 `constants.PROBE_RADIUS`），
**PB 的 voxel mask / atom→grid 映射 / erosion 表示 / 网格间距 h 一律不外泄**。
两者的几何体确实同一个（`dilate(vdw, ball(R_probe))` 就是 SR 采样的那张球面所围的体），
但从布尔占据格点数边界面取面积是 Cauchy 阶梯 —— 对球精确算是 **6πR² vs 真值 4πR²，
系统性 +50%**，且会把 SA 的精度绑死在 PB 的 h 上。不合。
