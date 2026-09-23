# JAXPBSA 实施设计文档

对应 [JAXPBSA_OpenMM_Plan.md](./JAXPBSA_OpenMM_Plan.md)。本文把计划落到可写代码的粒度：
单位约定、每个 kernel 的算法、数据形状、API、测试矩阵、验收标准。

---

## 0. 环境事实（已调研）

| 项 | 值 |
|---|---|
| 开发环境 | `openmm_dev` (Python 3.12.13, JAX 0.11.1 + jax-cuda12-plugin, OpenMM 8.5.2, mdtraj, pdbfixer, **openmmforcefields 0.16.0**（phosaa14SB 等力场）, scipy, pytest 9.1.1) |
| 开发机 | 24 核 CPU, 128 GB RAM, **RTX 2080 Ti (11 GB, sm_75)**；JAX 默认后端已是 gpu。fp64 = 1/32 fp32，显存限 h=0.5 下 B≈5 |
| 目标 GPU | 用户侧 RTX 5090 (32 GB, sm_120)。jax 0.11.1 的 cu12 plugin 支持 CUDA 12.8+，兼容 5090 |
| 外部参照物（**都不进包**）| **MMPBSA.py 14.0 + pbsa**（AmberTools，在 `openmm_dev` 里；需 `AMBERHOME`）—— 端到端 ΔG_MM/PBSA，`scripts/validate_mmpbsa.py`<br>**APBS 3.4.1** (`~/software/APBS-3.4.1.Linux/bin/apbs`) —— ΔG_PB 逐参数归因<br>**zsasa** (`tests/zsasa_ref.py`) —— SASA 对拍 |
| 离线测试体系 | pdbfixer/OpenMM Modeller 从氨基酸序列离线构建 S1–S3；真实体系 S4 = 1SPS/1SPR 已下载至 `data/raw/`（shell 默认沙箱无网络，需关沙箱联网，如 `curl files.rcsb.org`） |

---

## 1. 单位与数值约定（全项目统一）

| 量 | 内部单位 | 说明 |
|---|---|---|
| 长度 | **Å** | OpenMM 输入是 nm，在 `openmm_io` 入口一次性 ×10 转换 |
| 电荷 | e | 与 OpenMM/Amber 一致 |
| 势 u | **kT/e** | solver 内部量 |
| 能量输出 | **kcal/mol** | 1 kT = 0.5921868 kcal/mol (298.15 K)；换算集中在 `jaxpbsa/constants.py` |
| Coulomb 常数 | 332.06371 kcal·Å·mol⁻¹·e⁻² | |
| **真空 Bjerrum 长度** | **`BJERRUM_VAC = 560.75 Å`** | `= 332.06371 / 0.5921868 = e²/(4πε₀k_BT)`。**PB 源项和 DH 边界都必须乘这个因子**，见 §3.4 / §3.6 |
| 精度 | **数组 fp32 + 归约 fp64**（实测定案，见下）。`jax_enable_x64` **保持开启**——它是让 float64 存在的*能力*；数组 dtype 才是*策略*，由 `jaxpbsa.set_precision(32\|64)` 控制 | |
| 默认 tol | **1e-5**（相对残差）。fp64 下 tol 1e-7 与 1e-5 的 G_PB 只差 0.002 kcal/mol，1e-7 是纯粹的过度求解 | |

kappa —— **是两个量，不要混用**（本项目最容易出错的一处）：

```
物理量（用于 padding 取值、T2 远场拟合）：
  κ²  = 8π · l_B(ε_out) · I_n
  l_B = e²/(4πε₀·ε_out·k_B·T)      (Bjerrum 长度，298.15 K, ε=78.5 → 7.135 Å)
  I_n = I[M] × 6.0221e-4            (mol/L → 离子数密度 Å⁻³，1:1 盐 I_n = ½Σnᵢzᵢ²)

算子里的量（APBS 约定，§3.6 的 stencil 用的是它）：
  κ̄²(r) = ε(r) · κ²(r)
```

`l_B` 里已经含了 ε_out。只有在体相（ε = ε_out）下 `∇·(ε∇u) − κ̄²u = 0` 才给出衰减长度 κ⁻¹。
**若在算子里直接写 κ² 而不是 ε·κ²，体相屏蔽长度会错 √78.5 ≈ 8.9 倍。**

校验点：I = 0.15 M 纯水 → κ⁻¹ = 7.86 Å（写入单元测试 T2）。

默认 PB 参数：

```
eps_in=1.0, eps_out=78.5, I=0.15 M, T=298.15 K
probe_radius=1.4 Å, ion_radius=2.0 Å (APBS 默认)
radii 模型: mbondi2 (H 半径随成键原子区分: 1.2/1.3/1.4 Å; 重原子 Bondi: C 1.7, N 1.55, O 1.5, S 1.8, P 1.8 Å)
h (grid spacing) 默认 0.5 Å; benchmark 档 {1.0, 0.75, 0.5}
SA: γ=0.005 kcal/mol/Å², β=0.0 (可配置, Amber MMPBSA.py 默认)
```

---

## 2. 数据结构与形状

**静态（topology 相关，构建时固定，不参与 JIT tracing）：**

```
charges[N], sigma[N], epsilon_LJ[N], radii[N]        fp32 (归约时升 fp64)
receptor_idx[NR], ligand_idx[NL]                     int32
elements[N] (元素符号→半径查表用)
```

**C/R/L 共用同一个 grid（重要——不是三套）：**

```
grid_origin[3], nxyz (int 静态), h      # 由 complex 决定，receptor / ligand 原样复用
```

> **padding 规则实测超配 2–4×（RESULTS §15.8）**：下面写的是 `max(20 Å, 3κ⁻¹)`，
> 而 κ⁻¹ = 7.86 Å ⇒ 23.6 Å。实测 complex 网格 padding **20 → 40 Å，ΔG_PB 只动 0.003**；
> 配体紧盒 20 → 12 差 **0.029**、→ 8 差 **0.090**，而节点数 18.6 M → 7.0 M → 4.0 M。
> **真实需求约 1–1.5 κ⁻¹。** 单中心 DH 边界不是瓶颈 —— 这条也否掉了「多中心边界能让
> padding 变小」的推断（没有可解锁的东西）。代码里一直传 `padding=20.0`（低于文中的
> 23.6），既然规则本身超配，这个不一致无后果，但规则该按实测改。

box 规则：对 **complex** 全部帧的坐标取 center = 几何中心（**逐帧** 取中心可减小 box，但会引入逐帧平移；第一版用 **整个 batch 固定 center**，保证 batch 维 static shape），半长 = max 原子距离 + padding；padding = max(20 Å, 3κ⁻¹)。h 三方向一致。**nxyz 从 APBS 允许的 `dime` 集合里取**（形如 `2^a·c+1`：65 / 97 / 129 / 161 / 193 / 225 / 257），这样对拍时不需要插值，见 §3.12。

> **为什么必须公共 grid**：G_PB 的绝对值里含有几十 kcal/mol 量级的 grid 自能离散误差。只有在 **同一 origin、同一 nxyz、同一 h** 下做 C − R − L 差分，这部分误差才会抵消。三个 species 各用各的 grid 会让 ΔG_PB 的离散噪声**大于信号**——这是 MM/PBSA 的经典坑。
>
> **实测证实（RESULTS.md §0.17）**：h=0.75→0.5 加密时 **G_C 变 +31.35、G_R 变 +31.36 —— 抵消到 0.01 kcal/mol 以内**。ΔG_PB 只变 −5.49（0.62%），而绝对值各自变了 3.2% / 2.5%。
>
> 更有用的推论：**剩余误差几乎全部来自配体**（`31.35 − 31.36 − 5.48 = −5.49`）。complex 的误差与 receptor 的误差几乎完全对消（受体 1655 原子主导 1835 原子的复合物），净剩下配体那 5.48。**所以要提高 ΔG_PB 精度，该加密的依据是配体而不是复合物。**
>
> **限定（RESULTS §15，2026-09-14 实测）**：真正需要共用网格的只有 **C 与 R 这一对**
> —— 它们共 1655/1835 个原子，误差是同一个东西。**孤立配体的 G_L 无可抵消**，
> 所以 `δΔG_PB ≈ −δG_L`，ΔG_PB 的离散误差就等于配体那一次求解的误差。
> **（2026-09-23 修正，RESULTS §17.1）**：下面那个 0.008 是单一摆放下的巧合 —— 按 7 种亚格点
> 相位平均，C−R 在 0.75/0.5 下差 0.8、各自摆放噪声 sd 6.0/3.2；G_L 的偏差（h=0.5 → 0.25 差 31）
> 按相位平均后仍成立。**非对称网格现已是默认**（`TripletSolver`：C/R h=0.75，配体 h=0.25 紧盒）。
>
> 实测 `G_C−G_R` 在 h=0.75 已收敛（0.75→0.5 只动 **0.008**），而 **G_L 在 h=0.5
> 差约 27 kcal/mol**。推论是非对称网格（C/R 粗、L 细），代价是三次编译而非一次。
> **尚未改默认** —— 只在 S4 一帧上验过，且紧盒把 origin 对齐与 padding 混在了一起。
>
> 附带好处（而且更懒）：三个 species 形状一致 → **一次编译覆盖三者**。实现上把 R/L 补齐到 complex 的原子数（屏蔽原子挪到盒外、电荷与半径清零），见 `solve.triplet`。不补齐就是三种形状、三次编译（各 ~8.5 s）。

**批内动态量（被 vmap 的轴在最前）：**

```
coords[B, N, 3]                       Å
rho        [B, Nx, Ny, Nz]            e/Å³ (trilinear 分配)
eps_x/y/z  [B, Nx-1, Ny, Nz] 等       面上介电（调和平均）
kappa2     [B, Nx, Ny, Nz]            Å⁻²，SES 内为 0
phi        [B, Nx, Ny, Nz]            kT/e
```

**内存预算（单个 solve 的活跃数组 ≈ φ,r,p,Ap,ρ,κ²,3×ε面 ≈ 10 张 grid；下表按 fp64 算，fp32 减半）：**

| h | complex box ~120 Å → n | 每帧-species 活跃内存 |
|---|---|---|
| 1.0 Å | 161³ | 10 × 33 MB ≈ 0.33 GB |
| 0.75 Å | 193³ | ≈ 0.58 GB |
| 0.5 Å | 225³–257³（S4 实际 box ~95 Å → 193³） | 0.58–1.2 GB |

→ 注意公共 grid 后**每帧占 3 个 species slot**（C/R/L）。5090 (32 GB) 上 h=0.5 / fp64 的帧数上限约 **B ≈ 5–8**（= 15–24 个 slot），fp32 翻倍。显式 **microbatch 流水**：`lax.scan` 外层按 chunk 喂帧，不要求整条轨迹常驻。

---

## 3. 模块设计

### 3.1 `openmm_io/`（M1，纯 numpy/python，无 JAX）

`openmm_io` 是 **input adapter，不是 runtime bridge**：不依赖任何 OpenMM–JAX 插件。
`openmm-jax`（JaxForce）方向相反（JAX 势 → OpenMM Force → MD loop），本项目不用；
v1 原则：**JAX does not enter the MD loop**（plan §1/§21）。

- `extract(system, topology) -> MMParams`：遍历 `NonbondedForce` 取 charge/sigma/epsilon（σ, ε 单位转换 → Å, kcal/mol）；记录 exceptions（本版本 cross-energy 用不到，但保存备用）。
- `assign_radii(topology, model="mbondi2")`：element + 成键邻居 → 半径（覆盖 H 的 1.2/1.3/1.4 规则；重原子 Bondi 查表；未覆盖元素报错而不是静默猜）。
- `receptor/ligand index`：接口收 `ligand_atoms`（如链 ID 或显式索引数组），receptor = 补集。
- `from_mdtraj(trajectory)` 辅助：mdtraj 加载 DCD/XTC → `coords[B,N,3]` Å，并校验 atom 顺序与 topology 一致。
- **不做 PBC 处理**：v1 假设轨迹已 unwrapped（complex 内无跨边界断裂）。带盒子时 MM cross 项提供 `periodic=True` 的最小镜像选项；PB 是有限 box，不受影响。

### 3.2 `mm/`（M2）

只算 cross interaction（single-trajectory 假设，bonded 项抵消）：

```
E_coul = Σ_{i∈R, j∈L} 332.06371 · qᵢqⱼ / rᵢⱼ
E_LJ   = Σ 4√(εᵢεⱼ)·[(σᵢⱼ)¹²/rᵢⱼ¹² − (σᵢⱼ)⁶/rᵢⱼ⁶],  σᵢⱼ=(σᵢ+σⱼ)/2   (Lorentz–Berthelot，与 OpenMM NonbondedForce 默认一致)
```

- 向量化：`vmap(frames)` × `(NR,1)-(1,NL)` 广播；NR×NL ≈ 数百万对，直接算。
- R–L 之间无 exceptions（分子间），无需排除表。
- **验收测试**：同一帧上与 OpenMM `CustomNonbondedForce`（只开 R–L 交叉、相同组合规则）对拍，目标 < 1e-6 相对偏差。

### 3.3 `pb/grid.py`

`make_grid(coords_batch, species_idx, h, padding) -> GridSpec`：按 §2 规则求 box。**每条 run 一次**，形状 static → 一次编译。

### 3.4 `pb/charges.py`

trilinear 点电荷分配（APBS `chgm spl0`，对齐最容易）：

```
ρᵍ = Σᵢ qᵢ · W₃(rᵢ − g)/h³            (W₃ = 三线性权重，每原子只写 8 个格点；单位 e/Å³)
b  = 4π · BJERRUM_VAC · ρᵍ            (BJERRUM_VAC = 560.75 Å，见 §1)
```

> **不要写成 `b = 4πρ`。** 在「长度 Å + 电荷 e + 势 kT/e」这套无量纲化下，
> `∇·(ε∇u) = −4π·(e²/4πε₀k_BT)·ρ = −4π·560.75·ρ[e/Å³]`。
> 漏掉这个因子会让 G_PB 小 560 倍。T1（Born 离子）会立刻抓到。

后续可选 B-spline（`spl2`）作为精度选项。

### 3.5 `pb/surface.py` —— 项目最难的 kernel，单独分层

**输入**：coords[B,N,3], radii[N], GridSpec。
**输出**：`inside[B,...]`（溶剂不可及的 SES 内部 0/1 map）+ `ion_out[B,...]`。

算法（全 batched，无 python 循环，形状 static）：

1. **vdW 占据**：把原子按 `floor((x−origin)/h)` 装进 lattice 自身的 voxel（cell list = grid 本身）。每个 voxel 的原子数上限 `MAX_PER_CELL`（构建期统计 batch 实际最大值 + 余量，static）。对每个格点，gather 27 邻 cell 中的原子（padding 到 MAX_PER_CELL），`inside_vdw = min_i(dᵢ − rᵢ) < 0` 判占据（即「到某原子的距离小于它的半径」）。复杂度 O(27·MAX_PER_CELL·Ngrid) 次距离计算，vmap over B。
2. **SES = 数学形态学闭运算**：`dilate(O, R_probe)` 再 `erode(R_probe)`。球形结构元素在 lattice 上展开为整数偏移集合 `{|o|·h ≤ R}`（h=0.5, R=1.4 时 123 个偏移，static）。dilate/erode 用 `lax.fori_loop` 累积 max/min（内存 O(grid)，不展开偏移维）。粗网格 h=1.0 时 R=1.4 只覆盖 2 层 → 偏移集很小，代价自动下降。
3. **离子可及**：`ion_region = dilate(SES_occ, R_ion)`，κ² 只在该区域非零。
4. ε 面 = 相邻节点 ε 的调和平均：`ε_f = 2εᵢεⱼ/(εᵢ+εⱼ)`（APBS 同款）。
5. **对 ε 和 κ̄² map 做半径 `swin` 的球邻域调和平均平滑（默认 `swin=0.5 Å`，对应 APBS `srfm smol` 的窗机制；`swin < h` 时自动退化为不平滑）。锐利 SES（`srfm mol`，即 swin=0）只作为对拍选项。**
   ~~理由：MM/PBSA 看的是帧间差分，锐利介电边界的 grid 离散噪声在帧间是随机的，会直接污染
   ΔG_PB 的方差和 block-SEM。~~
   **⚠ 这条理由已被实测证伪（RESULTS §15.4）**：只挪 grid origin（物理零变化）时
   ΔG_PB 峰峰值 **swin=0.5 下 8.39，swin=0 下 7.22 —— 平滑没压住摆放噪声，还略差**。
   它确实把答案整体挪了 16.2 kcal/mol，但那是**改变了表面定义**，不是降噪。
   真正针对摆放噪声的手段是**分数体积 ε**（face 被溶质占多少，而非二值），见 §15.5。
   在那之前 `swin` 只应被当作一个**必须在论文里写明的表面定义参数**，不是降噪开关。
   **注意**：APBS 侧的 `swin` 默认 0.3 Å 在 h=0.5 下是无操作；T6 对拍时两边必须显式传同一个
   swin（=我们的默认 0.5），把它当对齐参数，见 §3.12。

v1 先保证正确 + 形状 static。

> **规律（踩过两次）：trip count 是 host 侧常量时，不要用 `fori_loop`。**
> XLA 的循环边界挡住跨迭代融合，改成 Python 展开后它会把整条链融掉：
> `surface.py` 的 `dilate` **28.95 → 4.86 ms（6.0×）**，
> `multigrid.py` 的 `_smooth` 让 PB triplet 的 **GPU 工作量 219.4 → 176.4 ms、
> GPU 操作数 11,987 → 3,610**（RESULTS §14.3）。两处都是逐位相同的结果。
>
> 判断依据只有一条：**`sweeps` / `offsets` 这类次数是不是编译期已知**。是就展开。

> **实测：surface 不是瓶颈**（plan §12 的预判是错的）。**RTX 2080 Ti** / fp64 / S4(1835 原子, 96 Å box)：
> h=0.75 → T_surface 4.2 ms vs T_solve 125 ms（**3.2%**）；h=0.5 → 35.2 ms vs 710 ms（**4.7%**）。
> Pallas/tiling 优化按这个比例排在 MG 之后。
>
> **但粗网格下它会静默改变物理**：`ball_offsets(1.4 Å, h)` 在 h≥1.5 只返回 1 个偏移，
> 探针球退化成一个点 → 闭运算变恒等 → SES 塌回 vdW 表面。所以 **h ≤ 1.0 才是有效区间**，
> 网格收敛研究不能包含 h=1.5/2.0。

### 3.6 `pb/operator.py` —— matrix-free 7 点差分

```
(Au)ᵢ = Σ_{f∈6面} ε_f·(uᵢ − u_neighbor)/h² + κ̄²ᵢ·uᵢ        κ̄² = ε·κ²，见 §1
```

边界：单中心 Debye–Hückel（APBS `bcfl sdh`），边界点 r_b：

```
u_b = BJERRUM_VAC · (Q_net/ε_out) · exp(−κ(r_b − a)) / (r_b · (1 + κa))
      (κ=0 时退化为 BJERRUM_VAC·Q_net/(ε_out·r_b))
```

- `BJERRUM_VAC = 560.75 Å` 与 §3.4 的源项是**同一个因子**，两处都不能漏。
- `a` = 分子等效半径（离子排除球半径），取该 species 的最大原子中心距 + `ion_radius`。
- `exp(κa)/(1+κa)` 是 DH 的**离子尺寸修正**，APBS 的 `sdh`/`mdh` 都含它。
  对 a ≈ 20 Å、κa ≈ 2.5 的蛋白，漏掉它边界值差 `e^2.5/3.5 ≈ 3.5` 倍。
- 逐帧 Q_net 由净电荷算出（向量化）。
- 线性 PB 下 A 对称正定 → CG 与几何多重网格都适用。

### 3.7 `pb/solver.py`

**几何多重网格以 CG 预处理器的形式接入**（`pb/multigrid.py`，已实现），不是独立求解器；
**且默认按网格规模自动启用**（`PBParams.precond="auto"`，阈值 `MG_NODE_THRESHOLD = 1.5 M` 节点）。

> **MG 在粗网格上是负收益**，实测交叉点在 **0.9–2.1 M 节点**之间，**两张卡一致**：
>
> | 节点 | 配置 | fp32 MG / fp32 Jacobi (2080 Ti) | (5080) |
> |---|---|---|---|
> | 0.91 M | h=1.0 | 0.87× | **0.45×（慢 2.2 倍）** |
> | 2.15 M | h=0.75 | 1.27× | 1.36× |
> | 5.00 M | h=0.5 | 1.74× | **2.81×** |
>
> 幅度两卡相反：粗网格上**卡越快 MG 越吃亏**（粗层那几个小网格是 kernel 启动延迟主导，
> 设备越快固定延迟占比越高）；细网格上则相反。原先硬编码 `"mg"` 在 h=1.0 上会让程序慢一倍多。

> **为什么不能当独立求解器——实测会发散，而且恰好在生产配置上。**
> 33³、随机右端、ε 在球面上跳变，每个 V-cycle 的收敛因子（**这组换卡不变**）：
>
> | ε 跳变 | 1:2 | 1:10 | **1:80** |
> |---|---|---|---|
> | 大球 (r=6 Å) | 0.21 | 0.27 | **1.8（发散）** |
> | 小球 (r=2 Å) | 0.21 | 0.21 | **停滞在 3e-2 后上升** |
>
> ε_in=1 / ε_out=78.5 就是 1:80 那一列，和溶质大小无关。原因是标准的：
> **几何插值（与算子无关）无法表示法向导数在介电界面上跳 80 倍的解**，
> 粗网格修正在最要紧的地方是错的。均匀系数下 MG 完全正常（0.25/cycle），
> 所以传输算子本身没问题。
>
> **修法（懒且稳）：套进 CG。** CG 是极小化，不可能发散；坏的预处理器只会变慢。
> 同时保住了 MG 买到的 h 无关性。要真让 MG 独立求解，得上算子相关插值
> （Alcouffe–Brandt / de Zeeuw "black box" MG）或 Galerkin 粗算子，代码量大得多。
>
> 对称性（CG 的前提）成立：restriction = P^T/2³（变分选择）、Jacobi 光滑器对称、nu1 = nu2。

> **实测依据** · **RTX 2080 Ti** / fp64 / S4, 96 Å box, padding 20 Å，DST 已接入（目标机 5090 上绝对时间会变，迭代数不变）：
>
> | h | n | it_solvent | ms/iter | T_solve | 有效带宽 |
> |---|---|---|---|---|---|
> | 1.00 | 97 | 142 | — | — | |
> | 0.75 | 129 | 174 | 0.72 | 125 ms | |
> | 0.50 | 193 | 221 | 3.22 | 710 ms | ~214 GB/s（峰值 616 → **35%**）|
>
> 迭代数标度实测 **~(L/h)^0.64**，不是原先估的 (L/h)^1、500–1500 次。**条件数比预想的好。**
>
> **MG-PCG vs Jacobi-PCG 实测** · **RTX 2080 Ti**（S4, fp32, tol=1e-5, V(2,2)）——迭代数换卡不变，时间比会变：
>
> | h | n | Jacobi-PCG | MG-PCG | 迭代比 | 时间比 |
> |---|---|---|---|---|---|
> | 1.00 | 97 | 83 it / 14.4 ms | 15 it / 21.4 ms | 5.5× | **0.67×（更慢）** |
> | 0.75 | 129 | 90 it / 34.6 ms | 12 it / 29.5 ms | 7.5× | 1.17× |
> | 0.50 | 193 | 107 it / 114.7 ms | **9 it** / 72.2 ms | 11.9× | **1.59×** |
>
> **迭代数的 h 无关性拿到了**：Jacobi 83→90→107（越细越多），MG 15→12→**9**（越细越少）。
> 趋势 0.67→1.17→1.59，**越往细网格走越划算**。
>
> **但时间收益远小于迭代收益，我原先估的 ~7× 是错的。** 每个 V-cycle ≈ 8 个 Jacobi 迭代的
> 成本（6 层 × (nu1+nu2) 次光滑 + 传输 + 残差），11.9/8 ≈ 1.5，与实测吻合。
>
> 光滑次数扫描（h=0.5）：V(1,1) 14 it / 1.45×、V(2,2) 9 it / 1.57×、**V(3,3) 7 it / 1.68×**。
> 多光滑反而更好——省下的一次 CG 迭代（含整个 V-cycle + 算子 + 两个内积）比多两次光滑贵。
>
> 融合优化（35% 峰值 → 减少每次迭代的数组遍历）另有 2–3×，与 MG 正交，**现在是更大的单项收益**。
> APBS 本身是多重网格这点仍成立：比较时必须写明双方用的求解器。

- **v1 主路径：几何多重网格 V-cycle**。固定粗化序列（如 193→97→49→25→13；各层保持奇数、
  形状 static，vmap 友好），smoother = damped Jacobi 或红黑 Gauss–Seidel，各层 pre/post 各 2 次；
  最粗层 13³ 直接稠密求解或多跑几十次 smoother。目标 ~10 个 V-cycle 收敛。
- **参考实现：Jacobi-PCG**（diag = Σ_faces ε_f/h² + κ̄²）；`tol=1e-6`（相对残差），
  `max_iter=1500`。用于 T1/T2 正确性对照，以及给 MG 的收敛结果做交叉验证。
- **vmap 下的 while_loop 语义要写死**：`lax.while_loop` 被 vmap 后会变成「跑到 batch 内
  **全部** lane 收敛」，因此 (a) 报告的迭代数 = batch 内**最大值**，(b) 已收敛 lane 的更新
  必须 mask 掉，否则会在收敛后继续被数值噪声推动。或者直接改成固定 V-cycle 数 + 末尾查残差。

### 3.8 `pb/energy.py` —— 两次求解法

```
G_PB = ½ Σᵢ qᵢ·(u_solvent(rᵢ) − u_ref(rᵢ)) · kT        (kcal/mol)
```

`u_ref`（**已实现：`pb/dst.py`，默认 `ref_solver="dst"`**）：**同一 grid**、均匀 ε_in、κ=0，边界用**逐原子解析库仑求和** `u_b = BJERRUM_VAC·Σᵢqᵢ/(ε_in·|r_b−rᵢ|)`
（不是单中心单极近似——ref solve 的边界误差不会和 solvent solve 抵消；边界点只有 O(6n²) 个，直接算不贵）。
势在原子位置做 trilinear 插值。两次 solve 必须在同一 grid 上，grid 自能才会抵消。

> **ref solve 不需要迭代**：均匀 ε_in、κ=0 是**常系数 Poisson + Dirichlet**，用 DST（离散正弦变换，
> 由 `jnp.fft` 做奇延拓实现）在 O(n³log n) 内**直接精确求解**，零迭代。
>
> **实测收益比预估的大。** 参考方程才是两者中更难解的那个——它没有介电对比、也没有 κ̄² 质量项：
>
> | h | it_solvent | it_reference | ref/solv |
> |---|---|---|---|
> | 1.00 | 142 | 271 | 1.91 |
> | 0.75 | 174 | 348 | 2.00 |
> | 0.50 | 221 | 494 | **2.24** → 占全部迭代 **69%**，且随加密仍在涨 |
>
> 端到端（**RTX 2080 Ti**, fp64）：h=1.0 **1.78×**、h=0.75 **2.26×**、h=0.5 **2.14×**；
> G_PB 与 PCG 差 1e-4 kcal/mol（相对 1e-7）——精确解，不是近似。

### 3.9 `pb/batch.py` + `trajectory/`

```
solve_all(coords_chunk[B,N,3]) -> G_PB[B,3]      # 3 = complex / receptor / ligand
    = vmap(pipeline)(stack_species(chunk))       # [3B, ...] 一个 kernel，公共 grid（§2）
    # pipeline: charges → surface → MG solve(solvent) → DST solve(ref) → energy
```

外层 `lax.scan` microbatch 喂 chunk。因为 §2 改成公共 grid，**三个 species 共用同一套 static 形状、
只编译一次**，直接堆在 batch 维上。

**warm start 与 vmap 是互相打架的，必须写清楚：**

chunk 内的 B 帧被 vmap **并行**求解，所以 warm start 只能跨 chunk 传递。正确做法是
carry **整个 `φ[3B, ...]`**——第 k 条 lane 用上一个 chunk 第 k 条 lane 的解（即 **stride-B warm start**），
而不是「把最后一个 φ 传给下一 chunk」。

推论：**B 越大，相邻帧间隔越大（t 与 t−B），warm start 收益越弱**。B 和 warm start 是此消彼长的。
因此 M7 的 benchmark **必须把 B 作为一个轴扫**（B ∈ {1, 4, 16} × {cold, warm}），
只报一个 B 下的「迭代数下降 30%」没有意义。

### 3.10 `sa/` —— JAX Shrake–Rupley（zsasa 只是对拍参照物）

```
G_SA = γ·SASA + β        ΔG_SA = γ·(A_C − A_R − A_L) + β·(1 − 1 − 1) = γ·ΔSASA − β
```

**β 不抵消**：三个 species 各有一份常数项，差分后剩 `−β`。Amber `pbsa` 的 **INP=1**
默认 β=0 会掩盖这一点，换 INP=2（γ=0.0378, β=−0.5692）就会差 0.57 kcal/mol。
默认取 INP=1 的 (0.005, 0.0)，**论文里必须写明用的是哪一档**——γ 直接平移 ΔG_SA。
测试 `test_beta_does_not_cancel_in_the_difference` 专门卡这条。

#### 实现：`sa/jax_sr.py`，唯一后端

`ΔG_MM/PBSA = ΔE_MM + ΔG_PB + ΔG_SA`，三项独立相加，**SA 从不回馈 PB**。
所以 SA 是一个独立阶段——但它仍然进 JIT 图，理由是批处理和 `lax.scan`，不是耦合。

**zsasa 不是后端，不是依赖，不进包。** [zsasa](https://github.com/N283T/zsasa)
是 Zig 写的独立 CLI，在本项目里**只有一个职责：给 JAX 实现做性能/结果对拍**，
因此它住在 `tests/zsasa_ref.py`，找不到二进制就跳过测试。
**这和 APBS 是同一个待遇**（§0：「外部进程调用，不进包」）——参照物不是依赖。

`jaxpbsa.sa.sasa()` 因此**没有 `backend=` 形参**：只有一个实现，分派是多余的。

**对拍时走 JSON 而不是 PDB**：`calc` 的 JSON 输入有 `r` 字段（逐原子半径 Å），
可以把我们的 mbondi2 **原样**喂进去。走 PDB 则半径由 zsasa 的分类器决定：实测 S4 上
`CCD: 882 atoms classified, 953 fallback`，近半数原子是猜的，且 CCD 是联合原子半径——
那样 rᵢ 的单源当场就破了，**对拍也就不是对拍**。

zsasa 自己已对着解析解验过（孤立球 2e-16、两球重叠 9e-6），所以它做参照物是够格的。

#### 算法：JAX Shrake–Rupley

golden-spiral 点集（host 侧静态常量，全原子共用），dense N² 距离 + `top_k` 取 K 近邻，
`lax.map` 分块限显存。整个核可 `jit`/`vmap`，`sasa_core` 是图内入口。

**关键的一步代数变换：不显式构造采样点。** 字面照抄 SR 要建 `[N,K,3]` 的
$\mathbf p_{ik}=\mathbf x_i+R_i\mathbf u_k$ 再逐个测距。把它代进埋藏判据展开：

```
|p_ik − x_j|² = d_ij² + R_i² + 2 R_i · u_k·(x_i − x_j)

埋住 ⟺  2 R_i · u_k·(x_i − x_j)  <  R_j² − R_i² − d_ij²
        └──── u[P,3] @ disp[K,3]ᵀ 一次矩阵乘 ────┘   └── 每对一个常数 ──┘
```

每原子只剩一次 `[P,3]×[3,K]` 矩阵乘。**顺带白捡两件事**：对 uₖ 取极值
`min(u·disp) = −d`，得 j 能遮住 i 的必要条件是 `d < R_i + R_j` —— 所以远邻居、
padding（d²=∞ ⇒ rhs=−∞）、自身项（disp=0 ⇒ `0 < 0` 假）**全部自动失效，不用写掩码**。

**K 近邻是有损的，所以有硬闸。** 判据 `K ≥ maxᵢ |{j : d_ij < R_i + R_max}|`。
充分性：不在该集合里的原子距离 ≥ R_i + R_max，比集合里每一个都远，所以 top_k 取最近的
K 个必然把整个集合装下，而任何可能的遮挡者都在集合里。

> **不能换成更紧的 `|{j : d_ij < R_i + R_j}|`**（S4 上 116 vs 136，看着省 15%）：
> top_k 按**距离**排序，半径不进排序，一个近处的小原子（不遮挡）会挤掉一个远处的
> 大原子（真遮挡）。所以界必须用 R_max。

**为什么不自动探测**（两条都实测过，都不安全）：

| 来源 | 所需 k |
|---|---|
| CIF（真空最小化的制备态） | 131 —— 不是 MD 系综的构象，**欠 5** |
| MD 帧 0 | 126 —— **欠 10** |
| MD 40 帧（10 ns 轨迹）跨度 | **123 – 136** |

单帧探不出系综的上界，热运动会挤出更密的局部；而 k 是 static argnum，逐块重探还会
反复触发重编译。默认 **192** 对这条轨迹有 40% 余量，且这个数是**堆积密度**限的
（半径 6.4 Å 球内 ~0.1 原子/Å³ ≈ 110），换个折叠蛋白量级不变。

兜底的是闸本身，**它直接报出该填多少，一次重试必中**：

```
k_neighbors=96 不够, 本批需要 ≥ 134。真遮挡会被丢掉, 面积偏大 —— 所以这里报错而不是返回。
  重试: sasa(..., k_neighbors=134)
  注意这是**本批**的值; 逐块处理长轨迹时取各块最大, 否则 k 变化会触发重编译。
```

**闸是充分条件，会早报**（实测 S4 单帧，k=256 为基准）：

| k | 128 | 96 | 80 | 64 | 48 | 32 |
|---|---|---|---|---|---|---|
| rel err | 0 | 0 | 5e-5 | 1.2e-4 | 1.4e-3 | **1.4e-2** |

闸在 k<134 就拦，而面积到 k=80 才开始动 —— 保守约 1.7×，这是该错的方向。
注意误差**恒为正且单调**（丢遮挡 ⇒ 暴露变多），所以没有闸的话就是第 1 节那种
「不报错、只给看起来合理的错数字」：k=32 时 +1.4% 没有任何提示。

**验收**（`tests/test_sa.py`，三层断言，只有第一层是自洽的）：

| 对拍对象 | 结果 |
|---|---|
| 解析解（孤立球、两球重叠、远离可加） | 与 stage 1 同一批断言，全过 |
| zsasa 参照物 总 SASA，S4，P=960 / 4000 | rel **4.6e-5 / 5.8e-5**（T4 要求 < 1%） |
| zsasa 参照物 逐原子 | 中位 rel 7e-3 —— **预期对不上**，两边点集不同 |
| 暴力法（显式建 p_ik、显式测距） | rel < 1e-6 = 逐点判定完全相同；这是上面那步代数变换的**唯一**独立验证，解析球测试发现不了它（无邻居时判据退化） |

不要拿 mdtraj 当基准——它的球面采样点生成方式与 golden spiral 不同，逐原子对不上，
且它不接受逐原子自定义半径（只有元素级的 `change_radii`），无法表达 mbondi2 对氢的成键依赖。

**性能**（2080 Ti，S4 1835 原子，P=960）：

| | zsasa 参照物 (host) | jax_sr B=1 | B=4 | B=16 |
|---|---|---|---|---|
| 单 species | 41 ms/帧 | 11.6 | 8.3 | **7.2** |
| 三 species（ΔG_SA） | 131 ms/帧 | 25.0 | 16.7 | **14.4** |
| 峰值显存 | — | 553 MB | 554 | 554 |

SA 从「每帧多 59%」降到 **占总量 6%**（PB 222 ms/帧）。

**邻居搜索在逐原子块内做，不物化 `[N,N]`。** 早先版本先建整张 d2 再分块跑采样点，
显存就卡在 d2 上：

| N | d2 矩阵（旧版） | 旧版 ms/帧 |
|---|---|---|
| 1835 | 13 MB | 10.3 |
| 8000 | 244 MB | 54.5 |
| 16000 | 977 MB | 138.8 |
| 32000 | 3906 MB | **OOM**（11 GB 卡）|

而**时间对 N 几乎是线性的**（主成本是逐原子那次 `[P,K]` 矩阵乘）——
所以这堵墙是白挨的。挪进块里以后峰值是 `B × chunk × (P·K + N)`，
**N 的二次项没了**，S4 结果逐位不变（6342.998 Ų），代码还少一层。

`chunk` 的默认值**按 B 缩放**（定在 64M 元素 = 256 MB fp32）。分块的收益只在 B 小时
才值得拿显存换 —— 实测 S4：B=1 时 chunk 42→256 是 **14.4→10.0 ms（−31%）**，
而 B=16 时 6.96→6.74 几乎没差（批维已经把 GPU 喂饱了）。不除 B 就会在 B=16 上
白占 5.8 GB。

`ponytail:` 逐块 d2 仍是 O(N²) **计算量**（显存已经不是了），与 `mm/` 同一档天花板；
真到十万原子再换 cell-list。

#### SA ↔ PB 的边界：只共享物理定义，不共享离散化

两者的几何体**确实是同一个**：`surface.py` 的中间量 `dilate(vdw, ball(R_probe))`
就是 SR 采样的那张球面所围的体 `∪ B(xᵢ, rᵢ+r_p)`，PB 拿它 erode 成 SES 后就扔了。
但**不能因此把 SA 挂到 PB 的网格上**：

- PB 手里是**布尔占据**（scatter-max）。从布尔格点数边界面取面积是 Cauchy 阶梯：
  对球精确算，x 向边界面数 = 2πR²/h²，三方向合计 ×h² = **6πR²** vs 真值 4πR² ——
  **系统性 +50%**，不是调参能救的。
- 要取面积得走 level-set `φ = minᵢ(|x−xᵢ| − Rᵢ)`，`A = ∫δ_ε(φ)|∇φ|`。但 φ 是
  **另一个核**（逐节点对候选原子取 min），比布尔 scatter 贵，PB 根本不算它。
- 合了会把 SA 的精度绑死在 PB 的 h 上。ΔSASA = −1146 Å² 是三个 ~6000 Å² 的差，
  和 ΔG_PB 那个 890 抵成 −9.4 同构；h=0.5 的 level-set 误差按 1–3% 算，
  落到 ΔG_SA(−5.73) 上就是 10% 量级。SR 现在对 zsasa 是 6e-5。
- 而且不值：SA 只占每帧 6%。

**该共享的**（必须只有一个 source of truth）：

| 量 | 单源 |
|---|---|
| rᵢ | `openmm_io.assign_radii`（mbondi2），PB 与 SA 同吃一个数组 |
| r_p | `constants.PROBE_RADIUS = 1.4 Å` |

**不该共享的，一律不外泄到 SA**：PB voxel mask、atom→grid 映射、erosion 表示、
网格间距 h。这些是离散化细节，不是物理。

两边用不同的 r_p **不会报错**，只会悄悄给出偏掉的 ΔG_MM/PBSA（PB 的分子表面和
SA 的可及表面对应不同溶剂），所以
`tests/test_constants.py::test_pb_and_sa_share_one_probe_radius` 既比默认值、
也扫源码里的裸字面量。zsasa 走 JSON 而不是 PDB 同属这条——走 PDB 半径就由它自己的
分类器决定，rᵢ 的单源当场破掉。

#### MG 的两个可调项（RESULTS §14）

- **`PBParams.mg_min_n`**（新）：粗化到哪一层为止，接到 `build_levels(min_n=)`。
  生产网格 161×161×193 → 81×81×97 → 41×41×49 → 21×21×25 → 11×11×13 → 6×6×7，
  `min_n = 7/11/21/41` 分别停在最后四档。**默认 7（未改）** ——
  41 在 S4/h=0.5 上快 10%，但只测过一个体系。
- **`mg_coarse_sweeps` 必须跟 `mg_min_n` 一起调。** 旧结论「4 与 50 无差别」
  **只在终层 252 点时成立**；终层 41×41×49 上 sweeps=4 让迭代数从 29 涨到 **69**。
- **`_smooth` 已改成 Python 展开**（不用 `fori_loop`）—— 见 §3.5 的规律。

### 3.11 `analysis/`

```
ΔG_MM/PBSA[frame] = ΔE_coul + ΔE_LJ + ΔG_PB + ΔG_SA
输出 dataclass + mean/std/block-SEM（τ_int, ESS 属第二阶段 online monitor，接口留好）
```

### 3.12 `benchmark/` + `scripts/`

- `scripts/bench_gpu.py`：一键跑（自动检测 GPU、矩阵 B×h×species、输出 CSV + T_surface/T_solve/T_energy、迭代数、峰值显存）。
  **报告口径：秒不可跨卡比较，必须同时报 (a) 迭代数 和 (b) 占设备峰值带宽的百分比**（`jaxpbsa/benchmark/roofline.py`）。
  一个 2080 Ti、5080、5090、H100 的带宽和 fp64 比例都不同；只有迭代数和 %峰值是可移植的。
- `scripts/validate_apbs.py`：生成 PQR + apbs 输入文件，调 APBS，出 MAE/RMSE/max-dev/correlation 报告（plan §15 指标）。**三个必须锁死的参数**：
  - **`mg-manual` 单层网格**——APBS 默认的 `mg-auto` 是 focusing（两层网格），解与单层不可比。
  - **`dime` 必须是 `2^a·c+1` 形式**（65/97/129/161/193/225/257），不是「任意奇数」。
    GridSpec 的 nxyz 直接从这个集合里选（§2），对拍时就不用插值。
  - **`srfm`** 与 §3.5 的选择一致：默认 `smol`，`mol` 作为对照跑一遍。
    其余逐项写出：`chgm spl0`、`bcfl sdh`、**`swin 0.5`（与 §3.5 默认一致，两边显式传同值；
    注意不要用 APBS 默认 0.3，在 h≥0.3 下它是不平滑，与我们实现不可比）**、
    `ion charge ±1 conc 0.15 radius 2.0`、
    `pdie 1.0`、`sdie 78.5`、`temp 298.15`，PQR 里的半径用同一份 mbondi2。

---

## 4. API（与 plan §26 一致）

```python
from jaxpbsa import JAXPBSA

analyzer = JAXPBSA.from_openmm(
    system=system, topology=topology,
    ligand_atoms=ligand_idx,           # receptor = 补集
    grid_spacing=0.5, eps_solute=1.0, eps_solvent=78.5,
    ionic_strength=0.15, temperature=298.15, radii_model="mbondi2",
)

result = analyzer(coords_batch)        # [B,N,3] Å, 首次调用编译
# result: dataclass, 每个 field 都是 [B] 数组
#   e_coul_rl, e_lj_rl
#   g_pb_complex/receptor/ligand, delta_g_pb
#   sasa_complex/..., delta_g_sa
#   delta_g_mmpbsa, solver_iters {species: [B,2]}
```

**字段名以本节为准**，plan §26 已同步（原来是 `delta_e_coul` / `delta_e_lj`）。

### 在线入口的两个额外字段（`online.py`，逐帧标量而非 `[B]` 数组）

```text
margin_A    溶质膨胀面到网格边界的余量 (Å)。**负数 = 已经在丢原子** ——
            越界原子被 clip + 权重置零悄悄丢掉 (§3.4)，G_PB 偏小而不报错。
            注意 margin ≥ 0 只保证没丢原子；Dirichlet 边界的物理阈值更大，
            由 PBSAReporter 的 margin_min 管（S4 传 12 ≈ 1.5κ⁻¹）。
sa_ok       SA 是否成功。k 近邻不够时 sasa() 按设计 raise（不给偏大的面积），
            在线 catch 成 flag —— 一帧异常不能杀掉几小时的 MD。
            False 时 sasa_* / delta_g_sa / delta_g_mmpbsa 全为 NaN。
```

离线路径不产生这两个字段：网格按整条轨迹的共同包围盒建，越界在建网格时就排除了。

包结构见 plan §25（已去掉 `pb/dielectric.py`、`sa/zsasa_backend.py`；dielectric 合进 `pb/surface.py`，SA 只留 `sa/jax_backend.py`），`pyproject.toml` + pytest，开发用 `pip install -e .` 进 `openmm_dev`。

---

## 5. 测试矩阵（全部离线 CPU 可跑）

| # | 测试 | 方法 | 通过标准 |
|---|---|---|---|
| T1 | Born 离子 | 单电荷 q=+1, r=2 Å, ε=1/80, κ=0, h={1.0,0.5,0.25} | **误差随 h 单调下降**并收敛到解析值 −81.98 kcal/mol；h=0.25 时 < 3%。<br>实测：h=1.0 → 6.54%，h=0.5 → **4.16%**，h=0.25 → 1.65%。<br>（原写的「h=0.5 时 < 3%」是未经验证的估计，锐利介电边界下做不到；单点阈值也卡不住离散化，改成收敛性断言） |
| T2 | Debye 屏蔽 | 单离子 + 盐 | 远场拟合 κ⁻¹ = 7.86 Å (I=0.15 M) < 2% |
| T3 | MM cross | pdbfixer 构建的 peptide–peptide 体系 vs OpenMM CustomNonbondedForce | rel err < 1e-6 |
| T4 | SASA | vs mdtraj shrake_rupley，两边 n_points=4000、同半径表、同探针 | **总 SASA rel err < 1%**（逐原子 0.1% 不可达，见 §3.10） |
| T5 | batch 不变性 | 同帧单独算 vs batch 算 | **ΔG_PB rel err < 1e-6**（**不是** bitwise：XLA 在不同 B 下会重排 reduction，batched while_loop 的迭代数也随 B 变） |
| T6a | **MMPBSA.py 端到端对拍**（`scripts/validate_mmpbsa.py`，已实现）| S4 单帧，&pb 参数逐项锁死；prmtop 由 parmed 从规范 System XML 直出，不走 tleap | **只有 ΔE_MM 是硬断言**：ΔE_coul / ΔE_LJ 必须近似逐位一致（实测 5.3e-6 / 1.8e-4）。<br>ΔG_PB、ΔG_SA 是**方法比较不设门槛**（另一套离散/表面/泛函；LCPO ≠ Shrake–Rupley）——实测 5.0% / 0.3%，且两边各自都已网格收敛，见 RESULTS §12 |
| T6b | **APBS 单帧对拍** | 3 体系各 5–10 帧，`mg-manual` 参数逐项对齐（§3.12）| **绝对 G_PB^C/R/L：< 1%**；**验收卡在 ΔG_PB：< 1 kcal/mol 或 < 2%**。<br>这条门槛**只对 APBS 成立**——它与我们同方法族（同方程、同类离散、都是格点 FD），才能逐参数对齐。拿它卡 MMPBSA.py 是范畴错误。<br>（蛋白 G_PB ≈ −2000 kcal/mol，原定「< 0.5 kcal/mol」= 0.025%，两个独立 PB 码做不到；差值里误差会抵消，ΔG_PB 才是真正要用的量）|
| T7 | warm start | 相同 tol 下能量一致；**扫 B ∈ {1,4,16} × {cold,warm}** | 迭代数下降曲线记录化（B 越大收益越弱，见 §3.9），不设硬断言 |

**离线测试体系（4 个，覆盖 plan §16 规模梯度）**：

- S1：小肽–小肽（~100 原子，T1–T5 调试用），pdbfixer 从序列构建
- S2：protein(~60 res)–peptide(8 res)（~1.2k 原子，APBS 对拍主力），同上
- S3：protein(~200 res)–peptide(15 res, 高电荷)（~3.5k 原子，性能压测），同上
- **S4（真实 benchmark，用户指定）：Src SH2 domain–phosphotyrosyl peptide，PDB 1SPS/1SPR**，文件在 `data/raw/`
  - **complex = 1SPS 链 C + 链 F（不是 A + D）**。非对称单元有三份拷贝，肽链的解析程度差别很大：
    | 肽链 | 11 个残基中解析出 | 缺失 |
    |---|---|---|
    | D | 7 | E(−3), I5, Y6, L7；另 GLN(−1) 缺侧链原子 |
    | E | 6 | E(−3), P4, I5, Y6, L7 |
    | **F** | **10** | 仅缺 N 端 E(−3) |
    用 A+D 会让 pdbfixer 去建模 4 个残基、包括**整段 C 端 IYL**——而 C 端正是 SH2 specificity pocket
    的结合部分，建出来的构象不可信。**用 C + F**，并检查 F 链是否也有缺原子。
  - SH2 ~1040 重原子 + 肽 **EPQ-pY-EEIPIYL**（11 残基，PTR 二价磷酸，肽净电荷 ≈ −5 —— 即 plan §16 的高电荷 protein–peptide 典型）
  - 1SPR = peptide-free SH2（多 trajectory 协议预留，v1 single-trajectory 用不到）
  - **力场**：`amber/ff14SB.xml + amber/phosaa14SB.xml`（openmmforcefields 0.16.0 提供 PTR/SEP/TPO，phosaa14SB = 2024 CTCT 参数，PTR 为二价阴离子）。**注意**：phosaa 必须配**无前缀类型名**的 `ff14SB.xml`；`protein.ff14SB.xml`（`protein-N` 前缀）配不上。水/离子：`amber/tip3p_standard.xml`（含 NA/CL 模板）。S1–S3 同用 ff14SB —— 全部体系统一 amber
  - **制备**：`scripts/prep_s4.py`（**pdbfixer-free**，PDB 文本层修复：缺失重原子从同型残基内坐标重建、PTR 走 CONECT + loadHydrogenDefinitions、HIS→HIE、SS→CYX、真空最小化；pdbfixer 对 HETATM 修饰残基有四个坑，见脚本头注释）。产物 S4_complex.pdb：**1835 原子，净电荷 0 = receptor(+4) + ligand(−4)**，无序 N 端（链 C GLN1、链 F GLU(−3)）截断不建模
  - **MD**：`scripts/run_s4_md.py`（溶剂化 23,847 原子 → 最小化 → NVT 100 ps → NPT 1 ns → 生产 10 ns，298.15 K 与 PB 温度一致，DCD 每 1 ps = 10,000 帧）+ `scripts/strip_unwrap.py`（image_molecules + 剥水/离子 → `S4_dry.dcd`，原子数与 meta 交叉校验）。CPU smoke 已通过

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| ~~5090 fp64 吞吐 1/64~~ **已定案**：默认 fp32 | — | 实测 fp32 只让 G_PB 偏移恒定 0.035 kcal/mol（相对 3.7e-5），比离散误差低三个数量级。**注意 stencil 是带宽瓶颈不是算力瓶颈**，所以 fp64 的真实代价是"2× 内存流量"而非"1/64 算力"——这个推理换卡不变 |
| **dtype 静默提升回 fp64**（不报错，只是变慢一倍且测量全废） | 所有 fp32 结论作废 | 来源：numpy 标量（强类型，会提升 fp32）、任何不带 dtype 的 `jnp.zeros/full/where`（x64 下默认 float64）、`while_loop` carry 初值。**已发生过一次**（`__init__.py` 强制开 x64 导致两轮 fp32 测量全是 fp64）。守卫：`test_dtype_is_not_silently_promoted` |
| surface kernel 数据依赖 | vmap 失败或编译爆炸 | cell-list + MAX_PER_CELL 静态 padding；这是设计约束而非事后补丁 |
| 细网格迭代多（条件数 ~ (L/h)²） | 吞吐不达标 | **v1 直接上几何多重网格**（§3.7），Jacobi-PCG 只作正确性参考。等 PCG 测完再决定是来不及的 |
| APBS 对拍偏差来源多（表面定义、平滑窗口） | 验证结论不可信 | `validate_apbs.py` 逐参数锁定（chgm/srfm/bcfl 全部显式写出）；偏差按 grid-spacing 收敛性分析，不只看单点 |
| 轨迹 wrapped / PBC | MM cross 算错 | 文档明确假设 + `periodic` 最小镜像选项 |
| ~~开发机无 GPU~~ **已解决**：本机 **RTX 2080 Ti**，JAX 默认后端已是 gpu | — | 算法/正确性工作可直接在本机 GPU 迭代。**剩余限制**：11 GB 显存（h=0.5/fp64 每 species slot ~0.6 GB → B≈5）；Turing fp64 = 1/32，fp32 路线先在这里验证。**5090 只在最终吞吐数字时才需要** |
| 只有 Jacobi-PCG 时跑不赢 APBS 的多重网格 | 数字不够看（**但不是 headline 崩塌**，见 §3.7 实测） | MG 仍在 v1；M4 验收含「V-cycle 数 ≤ 15」 |
| **C/R/L 用各自 grid 导致 ΔG_PB 噪声大于信号** | 所有能量结果不可用 | 公共 grid（§2），T5/T6 覆盖 |
| **无 MD 轨迹可用** | M9 benchmark 无输入 | 新增 M0（§7），体系制备 + MD 排进里程碑 |

---

## 7. 里程碑与进度

| M | 内容 | 状态 | 验收 |
|---|---|---|---|
| **M0** | 体系制备：S4 = 1SPS **链 C+F**，全部缺失原子在 **PDB 文本层**补齐（PDBFixer 对 PTR 有四处障碍，见 `scripts/prep_s4.py` 文档字符串），ff14SB + phosaa14SB | **结构完成** `data/prepared/S4_complex.pdb`：1835 原子 = receptor 1655 + ligand 180，净电荷 +0.000 = +4(SH2) + (−4)(肽)。**轨迹未生成** | 溶剂化 → 平衡 → ≥10 ns → strip/unwrap → DCD ≥ 10,000 帧 |
| M1 | openmm_io（含 `from_mdtraj` 的 strip / unwrap） | 参数提取完成 | T3 前置 |
| M2 | mm cross | **完成** | T3 ✅（**归约必须升 fp64**，见 §1） |
| M3 | grid + charges + surface | **完成** | ρ/ε map 断言 ✅ |
| M4 | operator + PCG | **完成** | T1 ✅ T2 ✅ |
| M4.7 | **MG 预处理器** `pb/multigrid.py` + `precond="auto"` | **完成** | h=0.5 时 11 次 CG 迭代（Jacobi 要 193），迭代数随 h 下降 ✅；两卡交叉点一致 ✅ |
| M4.5 | **DST 直解参考方程** `pb/dst.py` | **完成** | 自逆 1.1e-16、反演算子 2.2e-15、G_PB 对 PCG 差 1e-4 kcal/mol ✅ |
| M4.6 | **fp32 精度路线** | **完成** | `test_dtype_is_not_silently_promoted` ✅ |
| M5 | G_PB 验证 | 网格收敛已验；误差预算已建立（离散化 ~1.1 kcal/mol 主导，配体为主） | T1 ✅；**外部对拍 T6 未做** |
| **M5.5** | **C/R/L 三体** `solve.triplet` | **完成** | ΔG_PB = +885.405；参考解 3 次降 2 次（`u_ref_C = u_ref_R + u_ref_L`，实测相对差 1.83e-07）✅；三者共用一份编译 ✅ |
| **M5.6** | **规范产物**（转换器/下游边界） | **完成** | `S4_complex.cif` + `S4_system.xml` + 双 sha256；`load_canonical()` 加载即校验；下游不再 import ForceField ✅ |
| **M5.7** | **跨设备验证** | **完成** | 2080 Ti 与 5080 上 G_PB / 迭代 / 真残差**逐位一致** ✅ |
| M6 | JIT + vmap batch | 单帧 vmap 通过，**未接 microbatch scan** | T5 ✅ |
| M7 | warm start | 未开始 | T7（必须扫 B×{cold,warm}） |
| M8 | SA + 汇总 | 未开始 | T4 |
| M9 | benchmark | `benchmark/roofline.py` 已有 | 报告须含迭代数 + %峰值带宽 |

**下一步**（全部依据实测，见 RESULTS.md）：
1. **ΔG_SA**（M8）—— 补完 ΔG_MM/PBSA 的最后一项。目前已能算到 `ΔE_MM + ΔG_PB = −61.81 kcal/mol`。
2. **外部对拍 T6** —— 唯一还没做的外部验证。
3. **M0 轨迹生成** —— 结构已规范化，但生产 MD 未跑。
4. **溶剂求解器的访存效率** —— 它占 82–85%，且换快卡后占比更高；但**必须按 profile 判断**，
   不能再用纸面流量反推（§0.16 的教训：三个猜想里两个零收益、一个不可行）。

**不在列表里**：batching（实测两卡均为负收益）、warm start（1.08×，与 MG 此消彼长）、
MG 粗层轮数（无差别）、算子融合的「2–3×」（依据已撤回）。

---

## 7.1 实测记录

> **所有性能与正确性数字都在 [`RESULTS.md`](./RESULTS.md)**，本文件不重复。
> 那里按「换卡不变 / 本机专属」分了组，并标注了测量硬件（**RTX 2080 Ti**，
> 目标机 5090 不是本机）。另含 §8「已抓到的 bug 与测量陷阱」——11 条**全部是静默失败**，
> 不报错、只给看起来合理的错数字，其中 2 条是**测量工具的 bug 伪装成物理结论**。

## 8. 待确认决策点（默认值如上，均可改）

1. **ε_in=1.0**（plan API 示例用了 1.0；文献 protein–peptide 常用 2–4）——按 plan 用 1.0，只做成参数。
2. **radii = mbondi2**（Amber PB 惯例，APBS 对拍时同样用这组半径写进 PQR）。
3. **h 默认 0.5 Å**（correctness-first；吞吐 benchmark 会覆盖 0.75/1.0）。
4. **SASA 自实现 Shrake–Rupley**（不引入 zsasa 依赖；它不是瓶颈就不再换）。
5. **精度：fp32 存储/求解 + fp64 归约**（实测定案，见 §1 和下面的实测表），默认 tol=1e-5。
   不是"全局 fp32"——R–L 交叉能和 G_PB 的求和必须升 fp64。
6. **求解器：几何多重网格为主路径**（§3.7），PCG 降级为参考实现。这是从原计划的 v1/v2 里提上来的，
   因为论文结论依赖它。
7. **ref solve 用 DST 直接求解**（§3.8），不迭代。
8. **C/R/L 公共 grid**（§2）——这条不是可选项，是正确性前提。
9. **ε map 默认平滑（`srfm smol`）**（§3.5），锐利 SES 只在 APBS 对拍时用。
10. **S4 用 1SPS 链 C+F**（§5），不是 A+D。
11. **`precond="auto"`**（§3.7）——按节点数选，阈值 1.5 M，两卡实测标定。
12. **规范产物是 `S4_complex.cif` + `S4_system.xml`**，不是重跑制备管道。
    `prep_s4.py` 是转换器，下游只读产物、不碰力场。理由见 RESULTS.md §0.18：
    制备管道曾**完全不幂等**（重跑一次 1835 原子全动，RMSD 0.78 Å，G_PB 差 54 kcal/mol）。
13. **ΔG_PB 的误差棒不能用绝对值的相对误差估**（§2 / RESULTS.md §0.17）——
    按 G_C 的 3.2% 估会得到 28 kcal/mol，实际是 1.1 kcal/mol，**差 25 倍**。
