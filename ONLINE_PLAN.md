# 在线 MM/PBSA 实现计划

对应 `JAXPBSA_OpenMM_Plan.md` §21–§23 的落地顺序第 2 步（固定网格的在线接口）。
第 1 步（stage-2 JAX SA）已完成：`jaxpbsa/sa/jax_sr.py`，13.7 ms/帧，进图可 jit。

**分工**：本文件是实现说明，代码由人写，写完我审。

---

## 0. 三条契约（先写进 docstring，再写代码）

1. **网格建一次，全程不动**（`origin` / `shape` / `h`）。`make_grid` 只在 `__init__` 里
   调用一次，不暴露给调用方，避免它在循环里被重新调用。
2. **每帧只动坐标**：切溶质 → 平移到网格中心 → 喂求解器。坐标平移是运行期操作，
   实测不触发重编译、`G_PB` 不变（plan §21.5 约束 3）。
3. **越界不静默**。超出 padding 的原子当前被 `clip + 权重置零` 悄悄丢掉
   （`pb/charges.py:23`、`pb/surface.py:52`）—— 不报错，只给一个偏小的
   `G_PB`。必须每帧输出一个 margin 诊断位。

---

## 1. 前置决定：**`origin` 不改成运行期参数**（与 plan §21.5 的措辞不同）

plan §21.5 写的是「导出接口冻结前，`origin` 改成运行期 buffer 参数，**或者** `origin`
进 artifact 键，二选一，不能都不做」。

在 Python 阶段**第二个条件已经自动满足**：`enable_compilation_cache()` 用的是 JAX 自带
persistent cache，查找键是编译产物（HLO）哈希，而 `origin` 是烤进 HLO 的常量 ——
**键天然完备，不存在错配**。所以改成运行期参数的唯一收益是「跨体系复用同一份
executable」，那只在 PJRT artifact store 阶段才有意义。

代价不小：`origin` 有 5 处使用点

| 位置 | 用途 |
|---|---|
| `pb/charges.py:18` | `_weights` 的格点坐标换算（assign + interpolate 共用）|
| `pb/surface.py:46` | `rasterize_spheres` 的最近节点 |
| `pb/operator.py:58` | `node_coords`（jnp）|
| `pb/operator.py:68` | `shell_indices`（**host 侧 numpy**）|
| `pb/energy.py:95` | `center`（DH 边界）|
| `pb/energy.py:282` | `far`（species 屏蔽的「挪到盒外」）|

要动 PB 核心四个文件，其中 `shell_indices` 是 host 侧 numpy、`far` 是屏蔽语义
——而在线场景靠「每帧归位」完全等价。

**结论：这一项从「在线必做」降级为 PJRT 阶段的前置项。**
在 `online.py` 的 docstring 里写明这个判断**和它的失效条件**：
一旦要做显式 artifact store / 跨 Context 复用 executable，就必须回来做这一步。

这同时把原先的「每帧归位」和「origin 运行期化」合并成同一件事：**归位**。

---

## 2. 文件与接口

一个文件 `jaxpbsa/online.py`（~150 行），两个类。不新建子包，reporter 不单独开文件。

```python
class OnlineMMPBSA:
    def __init__(self, system, topology, solute_idx, ligand_local_idx, ref_coords_A,
                 *, h=0.5, padding=30.0, pb_params=PBParams(precond="auto"),
                 gamma=GAMMA_INP1, beta=BETA_INP1, sa_k=192): ...
    def __call__(self, coords_A) -> dict: ...     # [N_solute,3] Å -> 全部字段

class PBSAReporter:                               # OpenMM Reporter 协议
    def __init__(self, analyzer, interval_steps, solute_idx,
                 out_csv=None, on_violation="flag"): ...
    def describeNextReport(self, simulation): ...
    def report(self, simulation, state): ...
```

### 索引约定（最容易出错的地方，写进签名文档）

- `solute_idx`：**溶剂化体系**里的全局索引（切水/离子用）。
- `ligand_local_idx`：**切完之后** `[0, N_solute)` 里的局部索引。
- `receptor_local = 补集`，在 `__init__` 里算，不让调用方给。

两套索引混用不会报错，只会给一个错的 ΔG。`__init__` 里断言
`ligand_local_idx.max() < len(solute_idx)` 且两者互补覆盖。

### 返回字段

字段名照 plan §26 / `DESIGN.md` §4 对齐，加三个在线专有的：

```text
e_coul_rl, e_lj_rl
g_pb_complex / g_pb_receptor / g_pb_ligand, delta_g_pb
sasa_complex / sasa_receptor / sasa_ligand, delta_g_sa
delta_g_mmpbsa
solver_iters, converged
margin_A          <- 新增: 溶质膨胀面到网格边界的余量(Å), 负数 = 已经在丢原子
step, time_ps     <- 新增: 由 reporter 填
```

---

## 3. 分步实现

### S0 — 先把 padding 量出来（不写代码，跑一次现有数据）

padding 是**预定**的，不是拍脑袋的。已有 `data/md/S4_dry.dcd`（10 ns）：
对整条轨迹算每轴包围盒半长的最大值，与 canonical 帧比，得出构象涨落带来的增量。
`RESULTS.md` §15.8 的结论是现状 padding 超配 2–4×，但那是**离线按整条轨迹共同包围盒**
建网格；在线只有第 0 帧，必须把这个涨落量显式加回去。

产出：一个数（比如「S4 上 30 Å 覆盖 10 ns 全部构象，余量 8 Å」），写死进脚本默认值
并在 docstring 里注明来源。**这一步的输出是 S1 的输入。**

### S1 — `__init__`：参数装配（一次性，全部 host 侧）

```python
mmp    = extract_nonbonded(system)                 # 全溶剂化体系
q      = mmp.charge[solute_idx]                    # 切溶质
radii  = assign_radii(topology)[solute_idx]
prep   = prepare_cross(MMParams(切片后), lig_local, rec_local)
grid   = make_grid(ref_coords_A[None], h=h, padding=padding)
sv     = make_frame_solver(grid, radii, pb_params)
```

三条一次性检查（不是每帧）：

1. `round(q.sum())` 与 canonical 干体系的净电荷一致 —— 抓切片索引错位。
2. 切片得到的 `radii` 与 `load_canonical()` 的 `radii` 逐位相等 —— 抓
   `assign_radii` 在溶剂化 topology 上的行为差异。
3. `ligand_local ∪ receptor_local == range(N_solute)` 且不相交。

**预热**：`__init__` 末尾用 `ref_coords_A` 跑一次完整 `__call__`，把编译
（~8.5 s ×3 species + SA 三种形状）挡在 MD 开跑之前。否则第一次 `report` 会阻塞
MD 十几秒，还会污染 overhead 测量。配合 `enable_compilation_cache()`（36.5 → 5.9 s）。

### S2 — `_recenter`：归位

```python
grid_center = origin + 0.5 * (shape - 1) * h       # Python float, 见下面 dtype 坑
bbox_center = 0.5 * (coords.min(0) + coords.max(0))  # 与 make_grid 同约定
return coords - bbox_center + grid_center
```

- **用包围盒中点，不用质心** —— 必须与 `make_grid` 的约定一致，否则 padding 不对称。
- **PBC 不用自己 unwrap**：取坐标时用 `enforcePeriodicBox=False`，OpenMM 不做 wrap，
  分子天然完整。溶质是两个分子（受体 / 配体），只要初始同 image 且不解离就不会跳 image。
  真跳了 → 包围盒暴涨 → S3 的 margin 直接抓到负数。**所以不写 unwrap 代码。**
- **dtype 坑**：`grid_center` 里的分量必须是 Python `float` 或显式 dtype 的 jnp 数组。
  `np.float64` 标量在 JAX 里是**强**类型，会把整条 fp32 路径静默提升回 fp64
  （`pb/grid.py` 里 `origin` 用 `float()` 就是为了这个，同一个坑）。

### S3 — margin guard

```python
half   = 0.5 * (np.array(shape) - 1) * h
reach  = r_max + probe_radius + ion_radius + swin        # 介电图/离子排除的膨胀
margin = (half - jnp.abs(coords - grid_center).max(0) - reach).min()
```

- 在 `online.py` 里算，**不动 PB 核心**（一个极小的 kernel，相对 222 ms 可忽略）。
- 出在结果 dict 里，host 侧判。
- **策略在 reporter 里，不在 analyzer 里**：`on_violation="flag"`（默认，记录并继续）
  / `"raise"`。在线跑几小时，一帧异常不能杀掉 MD。
- `converged` 同理：不收敛的帧必须标记出来，不能混进平均值里。

### S4 — `__call__`：组合

```python
c = self._recenter(coords_A)
mm  = mm_cross_prepared(c[None], self._prep)                  # [B,N,3] 入口
pb  = self._sv.triplet(c, self._q, rec_local, lig_local)
sa, areas = delta_g_sa(c, self._radii, rec_local, lig_local,
                       gamma=..., beta=..., k=self._sa_k)
total = mm["e_coul_rl"] + mm["e_lj_rl"] + pb["delta_g_pb"] + sa
```

- SA 仍走现有的 host 侧 numpy 路径（三种形状 = 三次编译，一次性摊销，20 ms/帧，
  占 8.3%）。**不为在线把它融进同一个 jit** —— 等 S6 测出争用真的卡在这里再说。
- `sasa()` 的 `k` 是 static argnum，构象超出 K 近邻上界时它会 **raise**（设计如此，
  不给偏大的面积）。在线必须 catch 成 flag，同 S3 策略。
- **最后一次性同步**：所有 `float(...)` 转换集中在返回前做一次，不要逐项同步。

### S5 — `PBSAReporter`

- `describeNextReport` 返回 **6 元组** `(steps, True, False, False, False, False)`，
  最后一位是 `enforcePeriodicBox=False`（OpenMM ≥ 7.7；老版本是 5 元组，
  docstring 里注明最低版本）。
- `report`：`state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)`
  → `[solute_idx]` → `analyzer(...)`。**单位是 nm，×10 —— 一个断言写在这里。**
- CSV 一行一帧，**每行 flush**。跑几小时崩了不能丢数据。
- 自己记每次 `report` 的墙钟并累计，结束时打印实测 overhead —— 这是 S6 的数据源。

### S6 — 共卡争用实测（**目前唯一缺的数**）

plan §21.5 约束 4 的那张 overhead 表，`t_step = 247 μs` 是 MD **独占** 2080 Ti 的值；
在线时 PBSA 和 MD 抢同一张卡，表里 18% / 9% 全建立在「互不干扰」之上。

`scripts/online_overhead.py`：同一张卡，同一体系，跑固定步数，

| 配置 | 期望 |
|---|---|
| 无 reporter | 基线 ns/day |
| reporter @ 5000 步 | 表里 18% |
| reporter @ 10000 步 | 表里 9% |
| reporter @ 25000 步 | 表里 3.6% |

**必须先设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`** —— JAX 默认预占 75% 显存，
OpenMM 的 CUDA Context 会起不来或 OOM。同时试 `XLA_PYTHON_CLIENT_MEM_FRACTION`
限额，并记录「先建 OpenMM Context 还是先预热 JAX」两种顺序的差异。

读法：如果两者是干净串行的，实测应当**对上**公式 `T_pbsa/(N·t_step)`；
明显更差就说明瓶颈是显存/上下文切换，那时才轮到 plan §21 的双 GPU async worker。
**在这个数出来之前不要动 async。**

---

## 4. 测试：`tests/test_online.py`，三条断言，不需要跑 MD

1. **平移不变**：canonical 帧 + 随机平移 100 Å → `delta_g_mmpbsa` 与原帧
   `rtol=1e-5`（fp32 口径）相同。这条同时验归位和「平移不重编译」。
2. **padding guard**：`padding=2.0` 建 analyzer → `margin_A < 0`，且
   `on_violation="raise"` 时确实抛错。**抓的是静默丢原子。**
3. **组合一致**：online 路径 vs `scripts/validate_mmpbsa.py` 里的手工拼法，
   同一帧，逐项相等。**抓的是单位（nm/Å）、索引（全局/局部）、符号（β 不抵消）。**

第 3 条最值钱 —— 前两条只验新代码，它验的是「新代码和已对过拍的老路径是同一个量」。

---

## 5. 明确不做（写进 docstring，免得后面重捡）

| 不做 | 理由 |
|---|---|
| async worker / 双 GPU | 等 S6 的数。同步版够不够用是实测问题 |
| `jax.export` / PJRT plugin | plan 已判决跟随上游 #5320，不领先 |
| 零拷贝 | 实测只占 1e-5 |
| §23 的 JSD / ESS / 自相关 | CSV 存下来，离线算。在线不需要 |
| SA 融进同一个 jit | 占 8.3%，PB 占 91.5%，先优化错的项 |
| triplet 的 warm start | `_triplet` 不走 `u_prev`；且相隔 20–40 ps 的帧构象无关，1.08× 不值得 |
| `origin` 运行期参数 | §1，Python 阶段的缓存键已完备 |

---

## 6. 顺序与规模

```text
S0  padding 量化        跑一次现有轨迹, 出一个数        无代码
S1  __init__ 装配       ~50 行   验收: 三条一次性检查过
S2  _recenter           ~15 行   验收: 测试 1
S3  margin guard        ~10 行   验收: 测试 2
S4  __call__            ~30 行   验收: 测试 3
S5  PBSAReporter        ~40 行   验收: 短跑 1000 步出 CSV
S6  争用实测            ~80 行脚本  验收: 对上 / 对不上 §21.5 的表
```

S1–S4 之间不要并行写：S4 的测试 3 是唯一能抓住前三步组合错误的断言，
前面每步的「验收」只是让错误早暴露，最终判据是它。

---

## 7. 离线路径的归位 —— **已做**（2026-09-23，RESULTS §17）

`jaxpbsa.pb.TripletSolver` 把归位与网格收进同一个类，离线脚本（`crl.py` /
`validate_mmpbsa.py`）与 `OnlineMMPBSA` 都走它，两边自动同一套归位。
`solve.trajectory`（单 species）与 `benchmark.py` 等纯性能脚本**没改**。
下面是原来的待办说明，留作记录。

`recenter_com` 只在在线路径生效。离线侧 —— `pb/energy.py` 的 `solve.trajectory`，
以及喂它原始 MD 帧的 `scripts/crl.py` / `validate_mmpbsa.py` / `benchmark.py` ——
仍是「整轨迹共网格、坐标不动」，刚体漂移照样扫亚格点相位。

约束目前**只写在 `online.py` 的 docstring 里**，没有代码强制、没有测试覆盖。

**做在线 vs 离线逐帧对比之前**，先给离线侧加同一套质心归位（质量从 System 取），
否则散点就是 RESULTS §15.4 的那 ±8 kcal/mol 摆放噪声 —— 与物理无关，且很容易
被误读成在线路径有 bug。

---

## 8. 实测之后的待办（2026-09-20，数据见 RESULTS.md §16）

跑起来之后新增的，按价值排：

| # | 待办 | 依据 | 规模 |
|---|---|---|---|
| 1 | ~~**网格定尺改成算出来的**~~ **已做**（2026-09-25，`OnlineMMPBSA(pilot_coords_A=/fluctuation_allowance=)`，`analyzer.sizing`）（`scripts/fit_grid.py` 已经在做这件事，把它搬进 `__init__`）：`fluctuation_allowance` / `pilot_coords` 替代魔数 `padding=30`，内部 `need_half = max\|x−COM\| + allowance + reach + margin_min` → 选最小 dime | §16.8：白多两档 dime，省 26.9% 时间且 ΔG 不变。每个 overhead 档位上移一格 | ~15 行 |
| 2 | ~~**`margin_min` 从 κ 算**~~ **已做**（`online.boundary_margin_min`，reporter 默认跟 analyzer），不再钉死 12（= S4 在 0.15 M 下的 1.5κ⁻¹）。0.05 M 下该是 ~20 —— 换体系时钉死的 12 是错的 | §16.9 | ~3 行 |
| 3 | ~~两者都不给时**报错而不是拿单帧猜**~~ **已做**（单帧 pilot 同样报错）（单帧欠 3.79 Å，实测）。抄 `sasa()` 的 `k_neighbors` 判决 | §16.9 / §11.2 | 合在 1 里 |
| 4 | ~~离线路径补质心归位（§7）~~ **已做**（RESULTS §17）。对比仍**必须用同一批已保存的帧**，不能重跑 MD | §16.6：单条轨迹不可逐位复现 | — |
| 5 | 生产脚本改用 `mdtraj.reporters.DCDReporter(..., atomSubset=solute_idx)` | §16.7：DCD@1 ps 比 PBSA@50 ps 还贵，只写溶质少 13× | 1 行 |

**不做**（实测后更确定）：双 GPU async worker（§16.1 争用干净，无损失可回收）、
PJRT 插件按性能立项（§16.5 上限 0.24%）、运行中自适应网格（重编译 + 往 ΔG(t) 插台阶）。
