# JAXPBSA（OpenMM-centric）开发计划

## 1. 项目定位

本项目的重点不是 GBSA，而是：

\[
\boxed{\textbf{JAX-accelerated MM/PBSA for OpenMM trajectories}}
\]

整体架构：

\[
\boxed{\text{OpenMM = production MD / system authority}}
\]

\[
\boxed{\text{JAX = batched PBSA analysis backend}}
\]

OpenMM 负责：

- `System`
- `Topology`
- force-field 参数
- PME / nonbonded production electrostatics
- constraints
- thermostat / barostat
- integration
- trajectory generation

JAX 负责：

- trajectory batching
- MM interaction-energy analysis
- PB electrostatic solvation
- SA / nonpolar term
- trajectory-scale convergence statistics
- 后续 online/hot sampling monitor
- 后续稀疏 MLP audit

\[
\boxed{\textbf{JAX does not enter the MD loop.}}
\]

> JAXPBSA 是 OpenMM trajectory 的**外部高吞吐分析后端**，不是 OpenMM force provider。
> 方向是 `OpenMM MD → frames → JAX-PBSA`；`openmm-jax`（JaxForce）一类插件解决的是
> **反方向**（JAX 势函数 → OpenMM Force → MD loop），本项目用不到。
> 耦合越弱越好：DCD / XTC / 内存 frame 均可；JAX 侧的全部输入只是
> `X ∈ R^{B×N×3}` + 一组静态参数 `{qᵢ, Rᵢ, ε_in, ε_out, I, …}`，
> 一次 `jit` 之后不断吞 trajectory batches。online 形态见 §21。

项目第一阶段不改变 PBSA 的物理定义，而是把传统逐帧 PBSA 重新组织成：

\[
\boxed{
\text{fixed topology}
+
\text{many snapshots}
\rightarrow
\text{JIT-compiled GPU-batched PB solves}
}
\]

---

# 2. 核心科学问题

传统 MM/PBSA：

\[
\Delta G_{\mathrm{bind}}
=
\left\langle
G_C-G_R-G_L
\right\rangle
\]

其中：

\[
G_X
=
E_{\mathrm{MM},X}
+
G_{\mathrm{PB},X}
+
G_{\mathrm{SA},X}
\]

因此：

\[
\Delta G_{\mathrm{bind}}
=
\Delta E_{\mathrm{MM}}
+
\Delta G_{\mathrm{PB}}
+
\Delta G_{\mathrm{SA}}
\]

本项目真正研究的是：

> 能否把同一 OpenMM topology 下的大量 trajectory snapshots 转换成 GPU-friendly batched PB workloads，使 PBSA 从离线、逐帧、昂贵的后处理变成 trajectory-scale，甚至 online observable？

重点不是新的 PB 理论，而是：

\[
\boxed{\textbf{PB execution model}}
\]

---

# 3. 为什么优先 PBSA，而不是 GBSA

对于高电荷 protein–peptide / AMP–Toll/TLR 体系：

- peptide 可能具有较高净正电荷；
- protein surface 存在明显 electrostatic patches；
- salt bridge 较多；
- desolvation penalty 很强；
- dielectric boundary 不规则；
- ionic screening 不能简单忽略。

GB 把 continuum electrostatics 压缩为 effective Born radius：

\[
G_{\mathrm{GB}}
=
G(\{\alpha_i\})
\]

对于高度带电且表面复杂的体系，Born-radius approximation 的误差容易被放大。

PB 则直接求解：

\[
-\nabla\cdot
\left[
\epsilon(\mathbf r)\nabla\phi(\mathbf r)
\right]
+
\kappa^2(\mathbf r)\phi(\mathbf r)
=
4\pi\rho(\mathbf r)
\]

线性 PB 情况：

\[
A(\mathbf R)\phi=b(\mathbf R)
\]

> **注意一处内在张力（审稿人会抓）**：上面用「肽净电荷高、salt bridge 多、desolvation 强」
> 论证要 PB 不要 GB，但 §4 又把第一版限定为**线性** PB —— 而 |u| > 1 的高电荷表面
> 恰恰是线性化最不准的区域。两个修法二选一：
>
> 1. 把**非线性 PB 明确写进 roadmap**，并说明架构不排斥它（算子上只多一个标量 sinh 项 +
>    inexact Newton 外循环；多重网格的 smoother 可以直接复用）；
> 2. 把本节的论证从「精度更高」改成「**不依赖 Born radius 的经验拟合**，
>    介电边界与离子屏蔽是显式求解而非参数化」。
>
> 建议两条都做：改论证 + 留 roadmap 一句话。

因此本项目：

\[
\boxed{\text{PBSA = main path}}
\]

GBSA 最多作为：

- cheap prefilter；
- benchmark baseline；
- fallback mode。

不是主要方法。

---

# 4. 第一阶段范围

第一版只做：

\[
\boxed{
\text{MM}
+
\text{linear PB}
+
\text{SA}
}
\]

暂时不做：

- nonlinear PB；
- new dielectric theory；
- ML dielectric；
- MLP-MD；
- JAX MD engine；
- force-consistent PB dynamics；
- PB force；
- differentiable MD coupling；
- full PB free-energy force field。

第一阶段只需要：

\[
\mathbf R
\rightarrow
G_{\mathrm{PB}}(\mathbf R)
\]

因为 MM/PBSA 是 energy analysis，不需要：

\[
-\nabla_{\mathbf R}G_{\mathrm{PB}}
\]

这大幅降低开发难度。

> **但不要把「不需要梯度」写成卖点**——autodiff 正是 JAX 相对手写 CUDA kernel 的核心理由。
> 「那你为什么不直接写 CUDA」是审稿人必问的问题。
>
> 应当补一句：**整条管线除了 surface 的 0/1 indicator 之外都是可微的**，
> 这个 indicator 是已知的唯一阻碍（可用 soft/sigmoid 版本替代）。
> 一句话的成本，保住「为什么用 JAX」这个论证，并为第二阶段的 PB force 留门。

---

# 5. OpenMM 作为系统来源

目标 API：

```python
analyzer = JAXPBSA.from_openmm(
    system=system,
    topology=topology,
    ligand_atoms=peptide_idx,     # receptor = 补集，不单独传
)
```

> 与 DESIGN.md §4 对齐：只传 `ligand_atoms`，receptor 取补集。
> 同时传两个会引入「两者是否互补、是否覆盖全部原子」的校验负担，没有收益。

OpenMM 负责提供：

## 5.1 Nonbonded parameters

从 `NonbondedForce` 提取：

```text
charge
sigma
epsilon
exceptions
exclusions
```

## 5.2 Topology information

提取：

```text
atom -> residue mapping
element
receptor atoms
peptide atoms
atomic radii assignment
```

## 5.3 PB-specific parameters

JAXPBSA 自己保存：

```text
solute dielectric
solvent dielectric
ionic strength
temperature
probe radius
grid spacing
boundary condition
atomic radii model
surface definition
```

OpenMM 不负责 PB solve。

---

# 6. MM 项

对于 single-trajectory protein–peptide MM/PBSA，bonded terms 通常抵消。

因此：

\[
\Delta E_{\mathrm{MM}}
=
E_{\mathrm{Coul}}^{R-L}
+
E_{\mathrm{LJ}}^{R-L}
\]

直接计算 receptor–ligand cross interaction。

无需完整重写：

```text
bond
angle
torsion
improper
```

除非以后需要 multiple-trajectory MM/PBSA。

因此第一版 MM 部分应该保持极简。

---

# 7. PB 数学核心

## 7.1 Charge assignment

从原子坐标：

\[
\mathbf R
=
\{\mathbf r_i\}
\]

和点电荷：

\[
\{q_i\}
\]

构造 grid charge density：

\[
\rho(\mathbf r)
\]

例如使用：

- trilinear interpolation；
- B-spline assignment；
- Gaussian smearing。

第一阶段优先选择与 reference solver 最容易对齐的 charge assignment（trilinear = APBS `chgm spl0`）。

> **单位陷阱**：在「Å + e + kT/e」这套无量纲化下源项是
> `b = 4π · (e²/4πε₀k_BT) · ρ = 4π · 560.75 · ρ[e/Å³]`，
> **不是 `4πρ`**。漏掉这个因子 G_PB 会小 560 倍。同一因子也出现在 DH 边界条件里。
> 见 DESIGN.md §1 / §3.4 / §3.6。

---

## 7.2 Dielectric map

构建：

\[
\epsilon(\mathbf r)
\]

其中：

\[
\epsilon=
\begin{cases}
\epsilon_{\mathrm{in}}, & \text{solute}\\
\epsilon_{\mathrm{out}}, & \text{solvent}
\end{cases}
\]

关键步骤：

\[
\mathbf R
\rightarrow
\text{molecular surface}
\rightarrow
\epsilon(\mathbf r)
\]

这部分预计是整个项目最难的 kernel 之一。

需要单独设计：

```text
surface construction
grid occupancy
dielectric boundary
ion-accessibility map
```

> **κ 的两个版本不要混用**：物理量 `κ² = 8π·l_B(ε_out)·I_n`（→ κ⁻¹ = 7.86 Å @ 0.15 M）
> 用于定 padding；**算子里用的是 `κ̄²(r) = ε(r)·κ²(r)`**（APBS 约定）。
> 写错会让体相屏蔽长度差 √78.5 ≈ 8.9 倍。见 DESIGN.md §1。
>
> 另外介电边界默认做一次平滑（APBS `srfm smol`）：MM/PBSA 看的是帧间差分，
> 锐利 SES 的 grid 噪声在帧间随机，会直接污染 ΔG_PB 的方差。

---

## 7.3 Linear PB operator

离散化后：

\[
A\phi=b
\]

JAX 中需要实现：

```text
PB matrix-free operator
```

而不是显式构造巨大 sparse matrix。

推荐：

\[
\phi
\mapsto
A\phi
\]

通过规则 stencil 完成。

---

## 7.4 Solver

**修正：几何多重网格是第一版主路径，CG/PCG 降级为参考实现。**

```text
v1 主路径  : geometric multigrid (V-cycle, 固定粗化序列, 形状 static, vmap 兼容)
v1 参考实现: Jacobi-PCG          (用于 T1/T2 正确性对照与交叉验证)
```

理由（详见 DESIGN.md §3.7 与 §19-M4）：APBS 本身就是多重网格，
用 CG 去比 MG 是在比算法而不是比 execution model，而且 h=0.5 下大概率比输。

reference potential 的求解可以进一步简化：均匀 ε_in、κ=0 是**常系数 Poisson + Dirichlet**，
用 DST 可直接精确求解，零迭代 —— 每帧的迭代式 solve 从 6 次降到 3 次。

最终：

\[
\phi
=
A^{-1}b
\]

reaction potential：

\[
\phi_{\mathrm{reac}}
=
\phi_{\mathrm{solvent}}
-
\phi_{\mathrm{reference}}
\]

PB polar solvation energy：

\[
G_{\mathrm{PB}}
=
\frac12
\sum_i
q_i
\phi_{\mathrm{reac}}(\mathbf r_i)
\]

---

# 8. Complex / receptor / ligand

trajectory 只保存 complex：

\[
X_C
\in
\mathbb R^{B\times N\times3}
\]

得到：

\[
X_R=X_C[I_R]
\]

\[
X_L=X_C[I_L]
\]

MM 项可以直接 cross-evaluate。

但是 PB 必须真正计算：

\[
G_{\mathrm{PB}}^C
\]

\[
G_{\mathrm{PB}}^R
\]

\[
G_{\mathrm{PB}}^L
\]

最后：

\[
\boxed{
\Delta G_{\mathrm{PB}}
=
G_{\mathrm{PB}}^C
-
G_{\mathrm{PB}}^R
-
G_{\mathrm{PB}}^L
}
\]

因此每个 snapshot 逻辑上需要三个 PB evaluations。

> **C / R / L 必须用同一个 grid**（同 origin、同 dime、同 h，由 complex 决定）。
> G_PB 的绝对值里含几十 kcal/mol 量级的 grid 自能离散误差，只有在同一 grid 上做
> C − R − L 差分才会抵消；三套各自的 grid 会让 ΔG_PB 的离散噪声**大于信号**。
> 这是 MM/PBSA 的经典坑，也是本项目最容易在「看起来没报错」的情况下出错的地方。
>
> 附带好处：三者形状一致 → 可直接堆成 `[3B, Nx, Ny, Nz]`，一次编译、一个 kernel。
> 见 DESIGN.md §2。

---

# 9. Batched PB

传统 workflow：

```text
frame 1 -> PB solve
frame 2 -> PB solve
frame 3 -> PB solve
...
```

JAX 目标：

\[
\boxed{
[B,3,N_x,N_y,N_z]
}
\]

或等价的 batched operator。

逻辑上：

\[
\{A_k\phi_k=b_k\}_{k=1}^{B}
\]

通过：

```python
jax.jit
jax.vmap
jax.lax.scan
```

进行批处理。

关键目标：

\[
\boxed{
\text{compile once, solve many snapshots}
}
\]

---

# 10. Grid 策略

第一版不追求自动 adaptive mesh。

先固定规则 grid。

推荐 benchmark：

```text
0.5 Å
0.75 Å
1.0 Å
```

对于 protein–peptide：

```text
129^3
161^3
193^3
```

作为典型规模测试。

需要研究：

\[
\text{accuracy}
\leftrightarrow
\text{throughput}
\leftrightarrow
\text{GPU memory}
\]

---

# 11. Warm start

相邻 trajectory frame：

\[
\mathbf R_{t+\Delta t}
\approx
\mathbf R_t
\]

因此：

\[
\phi_{t+\Delta t}
\approx
\phi_t
\]

可以使用上一帧解作为下一帧初值：

\[
\phi_{t+\Delta t}^{(0)}
=
\phi_t
\]

这可能显著减少 CG / PCG iteration 数。

这是 trajectory-specific acceleration，不是普通单结构 PB solver 能充分利用的优势。

应单独 benchmark：

```text
cold start
vs
warm start
```

> ## ⚠️ 实测后的定位下调（RESULTS.md §0.2）
>
> warm start 的收益**取决于求解器**，而 MG 接入后它已经很小：
>
> | 求解器 | 冷启动迭代 | warm 迭代 | 省下 |
> |---|---|---|---|
> | Jacobi-PCG | 193 | ~155 (−20%) | 38 次 |
> | **MG-PCG** | **11.12** | **10.12 (−9%)** | **1 次** |
>
> 端到端 75.83 → 70.43 ms/帧，**1.08×**。能量一致（差 1.45e-3 kcal/mol）。
>
> **warm start 与 MG 此消彼长**：MG 把迭代数压到个位数之后，warm start 能省的绝对量
> 就所剩无几。本节原先把它当作「轨迹特有、单结构 solver 拿不到」的**核心 novelty**，
> 按实测这个说法站不住 —— 它仍然是真实存在的轨迹特有优化，但只值 8%，
> 不足以单独支撑论文主张。§27.1 的加速来源排序已相应更新。
>
> 另：`lax.scan` 本身**没有**额外开销（scan 70.43 vs Python 循环 71.01 ms/帧，差 0.8%）。

> **warm start 与 batching 是互相打架的，benchmark 必须体现这一点。**
> chunk 内的 B 帧被 vmap **并行**求解，warm start 只能跨 chunk 传递：
> 第 k 条 lane 用上一个 chunk 第 k 条 lane 的解（**stride-B warm start**），
> 因此 carry 的是整个 `φ[3B, ...]`，不是「最后一帧的 φ」。
>
> 推论：**B 越大，t 与 t−B 间隔越大，warm start 收益越弱。**
> 所以 §17 的矩阵里 B 与 cold/warm 必须是**交叉扫**（B ∈ {1,4,16} × {cold, warm}），
> 只报单个 B 下的「迭代数下降 30%」没有意义。见 DESIGN.md §3.9。

---

# 12. Surface / dielectric-map acceleration

预计真正的瓶颈可能不是 linear solver，而是：

\[
\boxed{
\mathbf R
\rightarrow
\epsilon(\mathbf r)
}
\]

因此这一部分要单独 profiling。

第一版可采用：

```text
atom-centered sphere rasterization
fixed-shape grid
local bounding boxes
batched occupancy update
```

后续优化：

```text
coarse spatial binning
cell lists
GPU tiling
JAX Pallas
custom kernel
```

如果已有成熟 surface/SASA engine 可复用，只要能输出合适的 surface representation，也可先接入。

---

# 13. SA / nonpolar term

定义：

\[
G_{\mathrm{SA}}
=
\gamma A+\beta
\]

其中：

\[
A=\mathrm{SASA}
\]

SA 不作为第一阶段 novelty，但 **γ / β 的取值必须在论文里写明是哪一档**，因为它直接平移 ΔG_SA：

```text
Amber pbsa INP=1 : γ=0.005  β=0.0        <- DESIGN.md 采用这一档
Amber pbsa INP=2 : γ=0.0378 β=-0.5692
常见 MM-PBSA     : γ=0.00542 β=0.92
```

backend 设计（**两段式**，见 DESIGN.md §3.10）：

```text
jaxpbsa/sa/
├── __init__.py   接口: sasa / g_sa / delta_g_sa,  backend="zsasa"|"jax"
├── zsasa.py      stage 1 (已实现): 外部 CLI, host 侧, 临时替补
└── jax_backend   stage 2 (待做):   JAX Shrake-Rupley, 进图, 可 vmap/scan
```

接口**收数组不收文件**，就是为了 stage 1 → stage 2 的替换对调用点无感。
替补的第二职责是**充当 stage 2 的验收基准**（它已对着解析解验过）。

实测（S4，1835 原子）：zsasa 三个 species **131 ms/帧**，PB 是 222 ms/帧 ——
SA 让每帧多 59%，且它在 host 侧进不了 `lax.scan`。这就是 stage 2 的动机。

计算：

\[
\Delta G_{\mathrm{SA}}
=
G_{\mathrm{SA}}^C
-
G_{\mathrm{SA}}^R
-
G_{\mathrm{SA}}^L
\]

如果现成 SASA 足够快，就直接使用。

---

# 14. 数据布局

Topology-dependent arrays：

```text
charges             [N]
sigma               [N]
epsilon             [N]
atomic_radii        [N]

receptor_idx        [NR]
ligand_idx          [NL]

grid metadata
PB parameters
```

Trajectory：

```text
coordinates         [B, N, 3]
```

PB grid（**C/R/L 共用同一 grid 后，species 直接并进 batch 维**）：

```text
rho                 [3B, Nx, Ny, Nz]      3 = complex / receptor / ligand
epsilon_map         [3B, Nx, Ny, Nz]      (面上介电: eps_x/y/z 三张)
kappa_map           [3B, Nx, Ny, Nz]
phi                 [3B, Nx, Ny, Nz]
```

显存推论：**每帧占 3 个 species slot**。h=0.5 / 257³ / fp64 下，
5090 (32 GB) 的帧数上限约 B ≈ 5–8（S4 实际 box ~95 Å → 193³，上限更高），fp32 翻倍。见 DESIGN.md §2。

如果显存不足：

```text
microbatch frames
```

而不是一次加载整个 trajectory。

---

# 15. Reference validation

第一阶段必须有成熟 PB solver 作为 reference。

建议对比：

```text
Amber pbsa   <- 推荐: 同为 mbondi2 半径 + Amber 力场, 且是 MMPBSA.py 背后的求解器
APBS
```

**不能用 GB 代替**：Amber 的 GBSA 是 PB 的近似，拿它对拍等于什么都没验
（§3 论证要 PB 不要 GB，正是因为两者会差）。GB 另有用处 —— 把 §3 那个论证
**量化**成数字，但它是"我们要击败的对象"，不是"验证我们对不对的基准"。

验证：

\[
G_{\mathrm{PB}}^C
\]

\[
G_{\mathrm{PB}}^R
\]

\[
G_{\mathrm{PB}}^L
\]

\[
\Delta G_{\mathrm{PB}}
\]

控制参数必须一致：

```text
atomic radii
solute dielectric
solvent dielectric
salt concentration
temperature
grid spacing
boundary condition
surface definition
```

重点报告：

```text
MAE
RMSE
max absolute deviation
frame-wise correlation
```

PB 不要求 bitwise equality。

关键是明确 numerical agreement。

**现实的容差（见 DESIGN.md T6）**：蛋白的 G_PB 绝对值在 −2000 kcal/mol 量级，
两个独立 PB 实现的绝对值一致到 **1%** 就已经很好；亚 kcal/mol 的一致性是做不到的。
因此：

```text
绝对 G_PB^{C,R,L}   : < 1%          (报告用)
ΔG_PB              : < 1 kcal/mol 或 < 2%   (验收用)
```

差分量里离散误差会大幅抵消，**ΔG_PB 才是真正要用的量，验收应当卡在它上面**。

还要锁死三个 APBS 侧的设置，否则对拍无意义（DESIGN.md §3.12）：
`mg-manual`（不是默认的 mg-auto focusing）、`dime` 取 `2^a·c+1` 形式、`srfm` 与本实现一致。

---

# 16. Benchmark 体系

## A. 小型 protein–ligand

```text
~3k–5k atoms
```

用于：

- solver correctness；
- grid convergence；
- compile overhead。

## B. protein–short peptide

```text
~8k–15k atoms
```

作为主要性能 benchmark。

## C. 高带电 AMP–Toll/TLR

作为最终应用。

特点：

```text
high peptide net charge
surface binding
salt bridges
strong desolvation
multiple electrostatic basins
```

这是 PBSA 相比 GBSA 更有物理意义的体系。

---

## D. 实验对照集（**原计划缺失，建议补上**）

论文的主张是**吞吐**而非精度，这可以接受。但如果一个实验对照都没有，
「算得快的错数」是审稿人一定会提的质疑。

建议补一个小的公开 protein–peptide 集（几个有实测 Kd 的 PDBbind 条目），
出一张 **ΔG_MM/PBSA vs 实验 ΔG 的 correlation 图**。
成本很低（体系制备流程 M0 已经有了），收益很大：
它把「我们做得快」变成「我们做得快**而且没做坏**」。

---

# 17. Benchmark 矩阵

Frames：

```text
1
10
100
1,000
10,000
```

Batch：

```text
1
2
4
8
16
32
```

Grid：

```text
1.0 Å
0.75 Å
0.5 Å
```

设备：

```text
CPU
single GPU
```

比较：

```text
Amber PBSA / APBS
JAX CPU
JAX GPU
```

指标：

\[
T_{\mathrm{total}}
\]

\[
\text{frames/s}
\]

\[
\text{iterations/frame}
\]

\[
\text{GPU memory}
\]

\[
T_{\mathrm{surface}}
\]

\[
T_{\mathrm{solve}}
\]

\[
T_{\mathrm{energy}}
\]

必须拆分 profiling。

---

# 18. Compile amortization

JAX 总时间：

\[
T_{\mathrm{total}}
=
T_{\mathrm{compile}}
+
T_{\mathrm{surface}}
+
T_{\mathrm{solve}}
+
T_{\mathrm{energy}}
\]

定义 trajectory crossover：

\[
N^*
\]

当：

\[
N>N^*
\]

时：

\[
T_{\mathrm{JAXPBSA}}
<
T_{\mathrm{traditional\ PBSA}}
\]

PBSA 本身是 multi-frame workload，因此这是 JAX 的核心优势。

---

# 19. 第一阶段 milestone

## M0 — 体系制备与轨迹生成（**原计划缺失的一步**）

M1–M9 里没有任何一步产生轨迹，但 M9 要 10–10,000 帧。必须显式排进来：

```text
PDB 清洗 (剥 HOH/PO4)
pdbfixer 补氢 / 补缺失原子
force field (S4: charmm36_2024 因 PTR; S1-S3: amber14)
溶剂化 + 加盐
minimize -> NVT -> NPT
production MD (>= 10 ns)
strip 水与离子
unwrap (PBC)
-> DCD
```

这一步的耗时和踩坑量不低于 M4，不能默认它「不算工作量」。
`from_mdtraj` 负责 strip / unwrap 的校验。

---

## M1 — OpenMM extraction

完成：

```text
System parsing
Topology parsing
charges
LJ parameters
receptor/ligand indexing
radii mapping
```

---

## M2 — MM cross energy

实现：

\[
E_{\mathrm{Coul}}^{R-L}
\]

\[
E_{\mathrm{LJ}}^{R-L}
\]

---

## M3 — Single-frame PB grid

完成：

```text
charge assignment
surface construction
dielectric map
ion-accessibility map
```

先：

```text
CPU
float64
one frame
```

---

## M4 — Linear PB operator

实现 matrix-free：

\[
A\phi
\]

并完成：

```text
geometric multigrid  <- v1 主路径，不是 v2
CG / PCG             <- 仅作正确性参考实现
```

**这是从 §7.4 的「后续」提前上来的。** 理由见 DESIGN.md §3.7：
h=0.5 / 241³ 下 Jacobi-PCG 约 0.5 s/solve，而 APBS 本身就是多重网格
（24 核 CPU 并行即可与之持平）。用 CG 去比 MG 是**比算法而不是比 execution model**，
而且大概率比输 —— §20 的 headline 会直接站不住。
验收：**V-cycle 数 ≤ 15**，且与 PCG 结果交叉一致。

---

## M5 — PB energy validation

完成：

\[
G_{\mathrm{PB}}
\]

并与 Amber PBSA / APBS 对齐。

---

## M6 — JAX JIT + batch

实现：

```text
[B, N, 3]
    ↓
batched PB
    ↓
[B]
```

---

## M7 — Warm start

实现 trajectory sequential warm-start solver。

比较：

```text
cold-start iterations
warm-start iterations
```

---

## M8 — SA backend

实现（**自实现 JAX Shrake–Rupley，不引入 zsasa**，与 §13/§25 一致）：

```text
sa/jax_backend (Shrake-Rupley)
```

得到完整：

\[
\Delta G_{\mathrm{MM/PBSA}}
\]

---

## M9 — Large-scale benchmark

测试：

```text
3 systems
10–10,000 frames
multiple grid spacings
CPU/GPU
```

---

# 20. 第一篇论文应该停在哪里

第一篇不需要 MLP。

核心只讲：

\[
\boxed{
\text{OpenMM-native trajectory input}
}
\]

\[
\boxed{
\text{JAX GPU-batched PB}
}
\]

\[
\boxed{
\text{MM/PBSA throughput acceleration}
}
\]

\[
\boxed{
\text{protein–peptide application}
}
\]

可能标题：

> **JAXPBSA: GPU-Batched Poisson–Boltzmann Binding Energy Analysis for OpenMM Molecular Dynamics Trajectories**

或：

> **Accelerating Trajectory-Scale MM/PBSA Analysis with JAX**

---

# 21. 第二阶段：Hot / online PBSA

如果 JAX PB 足够快：

```text
OpenMM MD
    │
    └── frame buffer
            │
            ▼
        JAXPBSA
            │
            ├── ΔE_MM(t)
            ├── ΔG_PB(t)
            ├── ΔG_SA(t)
            └── ΔG_MMPBSA(t)
```

JAXPBSA 不修改 MD Hamiltonian。

只是 online observable。

> online 的形态是 **asynchronous analysis worker**，不是把 PBSA 做成 OpenMM `Force`：
>
> ```text
> OpenMM MD (GPU 0) ──every N frames──> ring buffer ──> JAX PBSA worker (GPU 1)
> ```
>
> MD hot loop 与 PBSA analysis loop 并行——**analysis 不能拖生产 MD**。
> 这才是 §1 那条边界（JAX does not enter the MD loop）在第二阶段的自然延伸：
> 单 GPU 时要解决的是 device 坐标 → JAX device array 的数据交换（DLPack / PJRT），
> 仍然不是 `JaxForce`。

---

# 21.5 OpenMM-native 路线：`jax.export` → StableHLO → PJRT C++ plugin

**不是在 `CustomForce` 里写 JAX，而是把 JAX 编译出的 executable 当 OpenMM kernel 调。**

```text
Python/JAX 侧 —— 只建一次
──────────────────────────
pbsa(coords, params) → jax.jit → jax.export → StableHLO / compiled executable

OpenMM C++ plugin
──────────────────────────
PBSAForce → PBSAForceImpl → CudaCalcPBSAForceKernel → PJRT executable
         → JAXPBSA GPU kernels → ΔG_PBSA
```

上游正在做同一条路线：[openmm/openmm#5320](https://github.com/openmm/openmm/issues/5320)
（`JaxForce`，**open 原型**，非稳定标准 Force）。已有 benchmark：ANI2x 水体系上相对
NNPOps/PythonForce 快 2.3–3.4×。该 issue 自陈两个难点：**OpenMM–PJRT 边界**，
以及**动态 neighbor list**（"neighbor-list 和模型编译进同一个 executable，
需要在运行中协调 C++ 与 JAX 之间的重新 export"）。

## 为什么 JAXPBSA 比 ML potential 更适合走这条路

| 上游 `JaxForce` 的痛点 | JAXPBSA 的情况 |
|---|---|
| **动态 neighbor list** —— 关键阻碍 | **不存在**。网格、`ball_offsets`、MG 层级序列全是 host 侧常量，形状全程 static |
| 需要 force / backprop | **只要 energy**。`ForceImpl` 的 `includeForces/includeEnergy` 里直接 `if (!includeEnergy) return 0.0;`，不产生 −∇_R G_PBSA |
| 编译摊销不确定 | **已实测**：编译 8.5 s、稳态 76.8 ms/帧、**交叉点 ~100 帧**、10,000 帧时编译占 1.1%。正是 `jax.export`/PJRT 最喜欢的「同一个 executable 反复调」 |

**C/R/L 应当 export 成一个 executable，而不是三个 Force。** 这一点在 JAX 层**已经做到了**：
`solve.triplet` 是单个 jit 函数覆盖三个 species（把 R/L 补齐到 complex 的原子数），
**一次编译**。导出时原样导出即可，OpenMM 不需要知道内部是 sequential 还是 vmap。

辅助输出建议一并导出：`[G_C, G_R, G_L, ΔG_PB, ΔG_SA, ΔG_PBSA]`。

## Force group 模式

```python
pbsa_force.setForceGroup(31)
integrator.setIntegrationForceGroups(all_groups_except_31)   # 不参与积分
...
state = context.getState(getEnergy=True, groups={31})        # 需要时单独查询
```

OpenMM 明确支持「某个 force group 不参与积分但可随时单独查询」，正好对应 §23 的
PB-derived CV。

## 实测给出的四条约束

### 1. 拷贝**不是**我们的瓶颈 —— 与 ML potential 相反

1835 原子的坐标是 **22 KB**（1835 × 3 × 4 B）。PCIe 4.0 x16 约 25 GB/s → **~1 μs**。
而我们一次 ΔG_PB 是 **88.4 ms**（5080，三 species）。**拷贝占 1e-5。**

所以「OpenMM positions GPU → CPU copy → JAX GPU 很蠢」这个判断对**逐步调用的
ML potential** 成立，对我们**不成立**：我们每 N ps 才调一次，单次 88 ms 计算。
零拷贝仍然值得做（工程上更干净），但**它不该是决定要不要做这条路线的理由**。
真正的理由是避免 Python callback 的调度开销和 GIL。

> **2026-09-20 实测修正：上面的 `1e-5` 是纸面数，真实是 `1e-3`（RESULTS §16.5）。**
> 那个估计只算了溶质 fp32 的 22 KB；实测 host 往返是
> `getState`+`asNumpy` **0.95 ms** + 切片归位 **0.081 ms** ≈ **0.24%** of 431 ms ——
> 因为 `getState` 搬的是**全体系 fp64**（559 KB）外加同步开销。差 100 倍。
>
> 结论方向不变，但这个数把本节的立项论证钉死了：**OpenMM-native 插件能省的性能
> 上限就是 0.24%**。编译也不是它能省的 —— 编译是每进程一次（缓存命中 9.7 s），
> 不是每帧，而 PJRT 同样要在 Context 建立时 load executable。同卡争用也已实测
> 为干净串行（RESULTS §16.1），没有「调度不当」的损失可回收。
>
> **所以这条路线只能按 capability 立项**（PB 作为 force group 31 的可查询 CV），
> 不能按性能立项 —— 按 0.24% 论证，benchmark 一跑就打脸。

### 2. Host 侧的 SA 是硬阻碍 —— 这会改变 stage-2 的优先级

当前 SA 后端 zsasa 是**外部子进程**（§13），**进不了 PJRT executable**。
所以 **stage-2 的 JAX Shrake–Rupley 不是可选优化，而是这条路线的前置条件**。
在 RESULTS.md §9 的排序里它本来就是第一位，这里是第二个理由。

### 3. 在线场景：**动坐标，不动网格**

网格根本不需要变 —— 但接口必须把这件事约定死，因为 `grid.origin` 是闭包常量。

**实测**（S4，h=1.0，形状不变只平移原点）：

| 改什么 | 后果 |
|---|---|
| `grid.origin` 平移 3 Å | **重编译 2.59 s** |
| **坐标**平移 3 / 10 Å | **不重编译**，23.5 ms 稳态，G_PB 不变（−1267.6937 / −1267.6944）|

所以在线模式是：

1. **网格定死一次**（形状 + 原点都不动），padding 给足以覆盖构象涨落；
2. **每帧把坐标平移回盒心** —— 溶质在 MD 里会扩散漂移，但这是运行时操作，
   不触发重编译；
3. 接口**不暴露 `make_grid` 给调用方**，避免它在循环里被重新调用。

唯一真正需要改网格的情况是**构象尺度超出 padding**，靠一开始给足 padding 解决，
不是运行时问题。这也是上游 neighbor-list 问题的同构版本 —— 区别在于我们能靠
「预先定死」绕开，ML potential 不能。

### 4. 查询频率的定量上限（§22 要的数）—— 已实测

```
overhead = T_pbsa / (N · t_step)
```

**`t_step` 实测**（RTX 2080 Ti，S4 溶剂化 **20,637 原子**，PME + 1.0 nm cutoff，
HBonds 约束 + HMR，**4 fs**）：**247 μs/步 = 1397 ns/day**。

配 `T_pbsa = 222 ms`（同卡，PB 三 species，**尚未含 SA**）：

| 查询间隔 | 模拟时间 | overhead | §22 分档 |
|---|---|---|---|
| 每 100 步 | 0.4 ps | **898%** | 只能离线 |
| 每 200 步 | 0.8 ps | **449%** | 只能离线 |
| 每 1000 步 | 4 ps | 90% | 只能离线 |
| 每 2000 步 | 8 ps | 45% | 降低频率 |
| **每 5000 步** | **20 ps** | **18%** | 接近 practical |
| **每 10000 步** | **40 ps** | **9%** | **practical hot monitor** |
| 每 25000 步 | 100 ps | 3.6% | **always-on** |

**结论：PBSA 每 20–40 ps 查一次落在 practical 区间，每 100 ps 可以 always-on。**
这正好是 MM/PBSA 惯用的取帧频率（通常每 10–100 ps 存一帧），
所以 **online PB CV 不需要为采样频率做任何妥协**。

> **本表已被同卡实测确认（RESULTS.md §16）。** 上表是推算；2026-09-20 在 2080 Ti 上
> 真的把 PBSA 挂进 MD 循环跑了 200k 步 × 4 个配置：
>
> | 查询间隔 | 实测 overhead | 公式 `T_pbsa/(N·t_step)` |
> |---|---|---|
> | 每 10 ps | 30.6% / 31.4% | 31.1% |
> | 每 20 ps | 16.6% / 16.3% | 15.9% |
> | 每 50 ps | 9.8% / 7.1% | 6.5% |
>
> 差值 ±1–3 个百分点且非系统性（有一档实测**低于**公式），落在 MD 的跑间方差里。
> **MD 与 PBSA 干净串行，同卡争用没有额外代价** —— 所以 §21 的双 GPU async worker
> 至今没有证据支持：单卡换调度不改变 GPU 总工作量，能改变的只有换第二张卡。
>
> 两处口径要注意，否则会误读：实测是 **2 fs 无 HMR**（每模拟 ps 慢 2.2×）且在线默认
> `padding=30`（网格 9.77 M 节点，每帧 438 ms 而非 222）。两个 2× 抵消，按**模拟时间**
> 与上表重合（每 20 ps：18% vs 实测 16%；每 100 ps：3.6% vs 外推 3.2%）。
> 另：上面写的「20,637 原子」与当前 `data/md/S4_solvated.pdb` 的 **23,847** 对不上，
> 以文件为准。

> **修正记录 —— 错的是单位换算，不是估计值。**
>
> 本节先前写 `t_step ≈ 0.9 μs`，据此得出「每 1–4 ns 才能查一次」。追查如下：
>
> | | |
> |---|---|
> | 输入假设 | ~30k 原子、2 fs、约 200 ns/day（凭一般 MD 吞吐的印象）|
> | 这个假设本身 | **偏保守** —— 实测是 1397 ns/day |
> | 换算 | 步数/天 = 200 ns / 2 fs = 1.0e8；秒/步 = 86400 / 1.0e8 = **8.64e-4 s** |
> | 错在哪 | **8.64e-4 s = 0.864 毫秒 = 864 微秒**，被写成了 **0.86 微秒** |
> | 偏差 | **1005 倍，纯单位错误** |
>
> 若当时换算正确（864 μs），结论会是「每 1000–2000 步落进 5–10%」，
> 与实测口径（每 5000 步 18%）同量级，不会得出「每 1–4 ns」。
>
> **这个错误不会触发直觉警报**：0.86 μs/步对 GPU MD 听起来"像一台快机器"，
> 864 μs/步听起来"太慢" —— **错的那个反而更符合预期**。
> 与本项目其他静默失败同类：不报错，只给一个看起来合理的错数字。
>
> 教训不是"别用估计值"，而是：**跨数量级的单位换算必须落到纸面逐步验算**
> （ns/day → 步/天 → 秒/步 → μs/步 四级跳），
> 尤其当错误方向恰好符合预期时。
>
> 仍待确认：`T_pbsa` 尚未含 SA（stage-2 JAX SA 落地后需重算）；
> 500 步与 2000 步两次测量差 2 倍（565 vs 247 μs/步），说明还在预热，
> 长程 MD 的稳态值应更接近后者。

## 设计原则：planning is dynamic; execution is static

**不追求 shape-polymorphic 的 PB execution。** grid construction、multigrid
hierarchy、morphology stencil 全部视为 **Context specialization**：Context 建立
时由 host 侧 planner 算出具体 GridSpec 并解析出对应 executable，此后整段模拟
反复调用这一份 shape-specialized artifact，运行期 MD 帧只换坐标。

这不是偏好，是代码事实决定的：

- `build_levels()` 在 host 侧按具体 `(n_x, n_y, n_z)` 用 Python 循环搭 hierarchy
  （`multigrid.py`）。不同 shape 不只是某一维长度不同，是 **level 数、每一级
  shape 都可能不同**。
- `ball_offsets(radius, h)` 的长度是 `K(h,r) = #{n ∈ Z³ : |n|·h ≤ r}`
  （`surface.py`），probe/ion/smooth 三组 offsets 的静态长度都跟着 h 变。
  morphology 的性能正依赖这些 offset 在 host 侧静态展开、被 XLA 融合成
  `lax.slice` 链（`dilate` 28.95 → 4.86 ms 的教训，DESIGN §3.5）——
  把这部分改成动态长度等于亲手拆掉已有优化。

### universal artifact 的三条路全部否掉（量化）

| 方案 | 否决依据 |
|---|---|
| offsets pad 到 `h_min=0.25` 的最大长度 | m_max = ⌈1.4/0.25⌉ = 6 ⇒ 固定 (2·6+1)³ = **2197** candidate。真球内 K(0.25, 1.4) = **739**（3.0× 浪费）；h=0.5 时 K = **81**（**27×**）。且被 mask 掉的 candidate 破坏静态展开所换来的 XLA fusion |
| MG 各层 pad 到最大 shape + runtime mask | PB 本来就是 memory-heavy workload（solver 82–85% 时间是访存），把无效 grid volume 纳入 memory traffic 等于放大现有瓶颈 |
| 预编译 artifact pack | dime 从 APBS 集合取（33…385），三维自由组合本来就多。**更根本的：origin 取决于体系包围盒中点，连 (nx,ny,nz,h) 完全相同的两个体系 origin 都不同 —— origin 烤在 executable 里时，预编译包的命中率天生为零** |

所以缓存策略定为 **on-demand compile + cache**。固定编译成本对 online 不致命：
~8.5 s/份、~100 帧摊平、10k 帧时 1.1%。放进 OpenMM 就是 Context 建立时多几秒，
后面几小时到几天的模拟很快摊没。

### artifact cache 的键 —— 两条硬约束

**① 键必须完备，否则是正确性 bug，不只是 cache miss。** 手写结构键当场就能漏：

- **`grid.origin` 当前烤在 executable 里**（实测平移 3 Å → 重编译 2.59 s，
  本节约束 3）。键里没有 origin ⇒ 同 shape、不同体系的两次 Context 会
  **错误命中**，拿别人的 origin 沉默地算出一套看起来合理的数 ——
  与本项目其他静默失败同类：不报错，只给错数字。
- **手维护的键会腐烂**：`PBParams.mg_min_n` 本周才加，直接改变 level 数。
  每个 codegen-affecting 字段（`mg_nu` / `mg_coarse_sweeps` /
  `boundary_atom_block` / `precond` 的解析结果…）都是潜在的漏项，
  漏一个就是静默错配。
- batch 维是 vmap 的**静态**维：在线场景恒 B=1，可以写死；离线轨迹按变长分批，
  共享缓存时 B 必须进键或固定。
- 跨机器/跨版本复用还要求键含 jaxlib/XLA 版本与 GPU arch（sm_XX）。

**② 查找键从 trace 派生，不手工维护。** PBKernelKey 只作**日志与调试摘要**；
真正的查找键 = 编译产物（jaxpr/HLO）哈希 + 工具链/硬件标识。分两阶段：

- **现在（Python 路线）**：直接用 JAX 自带 persistent cache ——
  `enable_compilation_cache()` 已落地（编译 36.5 → 5.9 s，RESULTS §15.6），
  它的键就是产物哈希，天然完备。**没有自建 artifact store 的必要。**
- **PJRT plugin 阶段**：显式 artifact store、序列化 executable，查找键同上派生。

**origin 的归宿（必须在导出接口冻结前决定）**：把 origin 从闭包常量改成
**运行期 buffer 参数**。在线模式本来就约定「坐标归位回盒心」（约束 3），
origin 出运行期后键变短、跨 Context 复用率上升，预编译 pack 也才从零命中率
变成可行选项。**若不改，origin 必须进键** —— 二选一，不能都不做。

### 一个 Context = 三份 executable

非对称网格现为优化优先级①（RESULTS §15.7–15.8：又快 18%，与 Amber 差距
5.0% → 1.0%）且大概率成为默认（C/R 共盒 h=0.75 + 配体紧盒 h=0.25）。
所以 planner 的输出是 **{species → artifact}**，不是单 artifact；
编译 ~3×8.5 s，热进程 10k 帧占 1.3%（已实测）。

### 与 OpenMM 生命周期同构

平台层本来就在 Context 创建时做 per-Force kernel 的编译与加载，落点很自然：
planner + artifact lookup 放 `CudaCalcPBSAForceKernel::initialize()`，
per-frame 调用只喂坐标。**Context specialization 不是外来概念，
是 OpenMM 的既有模式。**

### 红线

只有两条不能接受：**每帧重编译**，或 **grid shape 随帧变化**。只要 trajectory
共用固定 grid，「每个体系 / 每个 h 一份 executable」不是问题 ——
坐标平移不重编译、G_PB 不变已有实测（约束 3）。

## 落地顺序

```text
1. stage-2 JAX SA        <- 前置条件, 否则整条流水线有一截在 host
2. 固定网格的在线接口     <- 预定 padding, 禁止中途改形状
3. origin 改运行期参数    <- 接口冻结前做; 否则 artifact 键不完备 = 静默错配
4. jax.export 单 executable  <- [G_C,G_R,G_L,ΔG_PB,ΔG_SA,ΔG_PBSA], 每 species 一份
5. PJRT C++ plugin       <- 跟随上游 #5320 的 boundary 方案, 不自己发明
   ^ planner + artifact lookup 放 kernel initialize(), 查找键从产物哈希派生
6. 零拷贝优化            <- 最后做, 实测只占 1e-5
```

**第 4 步应当跟随而非领先上游**：#5320 还是 open 原型，OpenMM–PJRT 边界的
方案没定稿。在它稳定前，Python callback 足够支撑 §23 的 sampling monitor 原型
（我们每 ns 才调一次，callback 的调度开销相对 88 ms 可以忽略）。
**Python callback 是原型手段，PJRT plugin 是产品形态** —— 两者不冲突。

---

# 22. Hot-path 性能目标

定义：

\[
\mathrm{overhead}
=
\frac{
T_{\mathrm{OpenMM+PBSA}}
-
T_{\mathrm{OpenMM}}
}{
T_{\mathrm{OpenMM}}
}
\]

建议：

```text
< 5%     always-on
5–15%    practical hot monitor
15–30%   lower evaluation frequency
> 30%    asynchronous / offline
```

> 这一节的可行性**完全取决于 §7.4 的求解器选择**。
> h=0.5 下 Jacobi-PCG 约 0.5 s/solve × 3 solve/frame ≈ 1.5 s/frame，
> 相对 OpenMM 的 overhead 会远超 30%，online monitor 直接不成立。
> 多重网格 + fp32 + warm start 才有可能进入「5–15% practical hot monitor」区间。

如果 PBSA 做不到 every-frame，也可以：

```text
every N ps
```

或：

```text
every B accumulated frames
```

批量执行。

---

# 23. Online sampling monitor

PBSA 不再只是最终平均分数。

持续监控：

\[
\Delta G_{\mathrm{PB}}(t)
\]

\[
\Delta G_{\mathrm{MM/PBSA}}(t)
\]

以及：

\[
\mu_t,\sigma_t,\tau_{\mathrm{int}},ESS
\]

比较 trajectory blocks：

\[
P_k(\Delta G_{\mathrm{PB}})
\]

与：

\[
P_{k+1}(\Delta G_{\mathrm{PB}})
\]

可以使用：

```text
block mean drift
Jensen-Shannon divergence
Wasserstein distance
autocorrelation
effective sample size
```

目标：

\[
\boxed{
\text{PBSA becomes a sampling diagnostic}
}
\]

---

# 24. 第三阶段：Sparse MLP audit

MLP 不进入 PB hot loop。

结构：

```text
OpenMM MD
    ↓
JAXPBSA
    ↓
detect new basin / PB anomaly / poor convergence
    ↓
select representative frames
    ↓
MLP single-point evaluation
```

定义：

\[
\Delta U_i
=
E_{\mathrm{MLP}}(x_i)
-
E_{\mathrm{MM}}(x_i)
\]

以及：

\[
w_i
=
e^{-\beta\Delta U_i}
\]

\[
ESS_{\mathrm{MLP}}
=
\frac{
(\sum_iw_i)^2
}{
\sum_iw_i^2
}
\]

目标不是让 MLP 替代 MM。

而是判断：

> OpenMM classical trajectory 是否覆盖了 MLP 认为合理的 configuration space。

最终分工：

\[
\boxed{\text{OpenMM = sampler}}
\]

\[
\boxed{\text{JAXPBSA = continuous electrostatic/sampling monitor}}
\]

\[
\boxed{\text{MLP = sparse high-cost auditor}}
\]

---

# 25. 软件结构建议

```text
jaxpbsa/
├── openmm_io/
│   ├── system.py
│   ├── topology.py
│   ├── parameters.py
│   └── radii.py
│
├── mm/
│   ├── coulomb.py
│   ├── lj.py
│   └── exceptions.py
│
├── pb/
│   ├── grid.py
│   ├── charges.py
│   ├── surface.py          # dielectric map 合并进来，不单开 dielectric.py
│   ├── operator.py
│   ├── solver.py           # multigrid 主路径 + PCG 参考实现
│   ├── energy.py
│   └── batch.py
│
├── sa/
│   └── jax_backend.py      # 自实现 Shrake-Rupley，不引 zsasa
│
├── trajectory/
│   ├── batch.py
│   ├── buffer.py
│   └── warm_start.py
│
├── analysis/
│   ├── mmpbsa.py
│   ├── statistics.py
│   └── convergence.py
│
├── benchmark/
│   ├── correctness.py
│   ├── grid_convergence.py
│   ├── throughput.py
│   └── online_overhead.py
│
└── cli.py
```

---

# 26. 最小 API

```python
from jaxpbsa import JAXPBSA

analyzer = JAXPBSA.from_openmm(
    system=system,
    topology=topology,
    ligand_atoms=peptide_idx,        # receptor = 补集
    grid_spacing=0.5,
    solvent_dielectric=78.5,
    solute_dielectric=1.0,
)

result = analyzer(coords_batch)
```

**已实现的等价入口**（`jaxpbsa.pb`，规范产物路径）：

```python
from jaxpbsa.openmm_io import load_canonical      # 只读 cif + 序列化 System, 不碰力场
from jaxpbsa.pb import PBParams, make_frame_solver, make_grid

d  = load_canonical()                              # 加载即校验 sha256
g  = make_grid(d["positions_A"][None], h=0.5, padding=20.0)   # 矩形, C/R/L 共用
sv = make_frame_solver(g, d["radii"], PBParams(precond="auto"))

sv(coords, q, radii)                               # 单帧, 单 species
sv.triplet(coords, q, rec_idx, lig_idx)            # ΔG_PB = G_C - G_R - G_L
sv.trajectory(traj, q, radii, warm=True, initial_state=st)   # -> (每帧能量, final_state)
```

返回（**字段名以 DESIGN.md §4 为准**，此处已对齐）：

```text
e_coul_rl                  # R-L cross，不是 delta_e_coul
e_lj_rl

g_pb_complex
g_pb_receptor
g_pb_ligand
delta_g_pb

sasa_complex / sasa_receptor / sasa_ligand
delta_g_sa

delta_g_mmpbsa
solver_iters               # {species: [B, 2]}，V-cycle / 迭代数
```

## 26.1 在线入口（`jaxpbsa.online`，§21 落地第 2 步）

离线三行是「帧管够、网格按整条轨迹建」；在线只有第 0 帧，网格必须**预先定死**，
每帧只动坐标（§21.5 约束 3）。所以是另一个入口，不是同一个函数加参数：

```python
import jaxpbsa
from jaxpbsa.online import OnlineMMPBSA, PBSAReporter

jaxpbsa.enable_compilation_cache()            # 把 __init__ 的预热编译压到 ~6 s

az = OnlineMMPBSA(system, topology,
                  solute_idx,                 # 溶剂化体系的**全局**索引(切水/离子)
                  ligand_local_idx,           # 切完之后 [0,N_solute) 的**局部**索引
                  ref_coords_A,               # 建网格 + 预热自检的参考帧
                  h=0.5, padding=30.0)        # padding 从轨迹量出来, 不是拍的
az(coords_A)                                  # [N_solute,3] Å -> 全部字段 + margin_A/sa_ok

sim.reporters.append(PBSAReporter(az, interval_steps=10000, solute_idx=solute_idx,
                                  out_csv="pbsa.csv", margin_min=12.0))
```

- **网格建一次**：`make_grid` 只在 `__init__` 里调，不暴露给调用方 —— 在循环里
  重建网格 = 每帧重编译。
- **归位用质心**（`recenter_com`），不是包围盒中点：中点由 6 个极端原子决定，
  远端侧链摆 1 Å 就移半个 h，按 RESULTS §15.4 那是峰峰 8.39 的摆放噪声直接进
  ΔG(t) —— §23 要测的正是这个量级。
- **同卡跑 MD 时必须** `XLA_PYTHON_CLIENT_PREALLOCATE=false`，否则 JAX 预占 75%
  显存，OpenMM 的 CUDA Context 起不来。
- 字段与守卫语义见 [ONLINE_PLAN.md](./ONLINE_PLAN.md) 与 DESIGN §4。

---

# 27. 开发优先级

优先级应当是：

\[
\boxed{1.\ \text{PB correctness}}
\]

\[
\boxed{2.\ \text{surface/dielectric-map construction}}
\]

\[
\boxed{3.\ \text{batched solver}}
\]

\[
\boxed{4.\ \text{warm start}}
\]

\[
\boxed{5.\ \text{trajectory throughput}}
\]

\[
\boxed{6.\ \text{online monitor}}
\]

而不是先优化 MM 或 SASA。

---

## 27.1 关于「Batch first」这个口号的诚实修正

**实测（RTX 2080 Ti，S4，详见 [RESULTS.md](./RESULTS.md) §6）：主 benchmark 工况下
batching 是负收益，不只是"几乎不加速"。**

| 相对 B=1 | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|
| h=0.5 (n=193) | 1.00× | **0.62×** | 0.63× | 0.63× |
| h=1.0 (n=97) | 1.00× | **2.60×** | 1.14× | 0.76× |

**不是显存**（峰值 2.78 GB / 11 GB）。生产网格上单帧已经喂饱设备，而批处理版 kernel
本身每帧效率低 1.6×（batch 维加在最前，打散了 stencil 的合并访存）。
粗网格上 B=2 确实有 2.60×——那是摊掉 MG 粗层的 kernel 启动延迟——但窗口极窄，B=4 就掉到 1.14×。

真实的加速来源按收益排序（**全部已实测，S4 / h=0.5，两张卡验证，见 RESULTS.md**）：

```text
1. fp32 + fp64 归约     <- 2.0-2.96x (2080Ti) / 2.5-4.1x (5080)
                           G_PB 偏移 +0.035, 离散误差是 40 量级
2. MG 预处理器          <- 2080Ti 1.74x / 5080 2.81x @ h=0.5
                           **粗网格上是负收益** -> precond="auto" 按节点数选
3. ref solve 用 DST     <- 2.14x       精确解; 参考方程本占 69% 迭代
4. tol 1e-7 -> 1e-5     <- 1.8x        两档 G_PB 只差 0.002
5. 矩形网格             <- 1.41x       节点 -30.4%
6. surface 静态 slice   <- 1.24x       dynamic_slice 阻止 XLA 融合; 逐位相同
7. 编译摊销             <- 100 帧是交叉点, 10000 帧时编译占 1.1%
8. warm start           <- 1.08x       MG 下迭代 11.12->10.12; 与 MG 此消彼长
9. batching             <- 0.63x(2080Ti) / 0.70x(5080)  两卡均为负收益
```

合计 **1800 -> 76.8 ms/帧（23.4×，单 species PB，2080 Ti）**；5080 上 **29.1 ms/帧**。
完整 ΔG_PB（C/R/L 三 species）：2080 Ti **222.5 ms/帧**，5080 **88.3 ms/帧**。

**三处与原先预判相反**：
- MG 不是 10–50×（实测 1.7–2.8×，每个 V-cycle ≈ 8 个 Jacobi 迭代的成本），
  而且**粗网格上是负收益**（0.45–0.87×）；
- warm start 不是核心 novelty（1.08×，MG 把迭代压到 11 后只能再省 1 次）；
- batching 在生产网格下是**负收益**，不是"几乎不加速"。

**benchmark 数据会自己验证这个排序**，所以论文口径应当提前改成
「trajectory-scale execution model（warm start + 编译摊销 + batched dispatch）」，
而不是把 batching 单独当作加速来源。否则 §17 的矩阵跑完会打脸 §29 的
"Batch first"。

---

# 27.5 Related work / 新颖性定位（**两份文档都缺这一节**）

GPU PB solver 已经存在，审稿人的第一个问题就是「和这些比新在哪」：

```text
AmberTools pbsa  (CUDA 版)
DelPhi GPU
APBS             (多重网格；有 GPU 分支)
gmx_MMPBSA       (并行封装，后端仍是 APBS / Amber pbsa)
```

它们全都是 **single-structure solver + 外层脚本循环帧**。
本项目可主张的差异化，按强度排序：

1. **compile-once / solve-many 的 execution model** —— 固定 topology + 变坐标，整条
   charge→surface→solve→energy 管线是一张编译好的图。实测编译 8.5 s、稳态 76.8 ms/帧，
   **交叉点在 100 帧附近**，10,000 帧时编译只占 1.1%。这是最稳的一条。
   C/R/L 三个 species 也**共用一份编译**（把 R/L 补齐到 complex 的原子数），
   而不是三种形状三次编译。
2. **warm start across trajectory frames** —— 单结构 solver 原理上拿不到的加速。
   但**实测只值 1.08×**（MG 下迭代 11.12→10.12）：它与求解器质量此消彼长，
   MG 把迭代压到个位数后就没多少可省了。**不足以单独支撑论文主张**，
   应作为 execution model 的一个组成部分而非 headline。
3. **算法与实现的组合收益** —— DST 直解参考方程、fp32+fp64 归约、矩形网格、
   静态 slice 形态学，合计 23.4×。这些不是「GPU 加速」而是「把同一物理算对且算省」。
3. **OpenMM-native**，不经过 PQR / 外部进程往返。
4. **可微就绪**（除 surface indicator 外全程可微，见 §4 的注记）——
   为第二阶段 PB force / online observable 留的路，是 CUDA 手写实现不具备的。

**不应该主张的**：「GPU 加速 PB」（已有）、「batched 所以快」（见 §27.1）。

---

# 28. 核心开发原则

## 不重写 OpenMM

不做：

```text
MD engine
integrator
thermostat
barostat
PME
constraints
full force field
```

## 不把 GB 当主线

GB 可以保留为：

```text
optional baseline
cheap prefilter
debug/reference mode
```

但论文主线应当是：

\[
\boxed{\textbf{PBSA}}
\]

## 不让 MLP 拖慢第一阶段

第一阶段：

```text
OpenMM + JAXPBSA
```

第二阶段：

```text
online PBSA
```

第三阶段：

```text
sparse MLP audit
```

---

# 29. 一句话项目定义

> **JAXPBSA is an OpenMM-native, JIT-compiled and GPU-batched MM/PBSA analysis engine designed to accelerate Poisson–Boltzmann trajectory analysis and ultimately enable PBSA as an online sampling observable for protein–peptide molecular dynamics.**

最终路线：

\[
\boxed{
\text{OpenMM production MD}
+
\text{JAX batched PBSA}
+
\text{optional sparse MLP validation}
}
\]

最核心的三个词（**"Batch first" 已按 §27.1 修正**——主工况下 batching 实测是 0.63× 负收益，
真正的差异化是 warm start + compile-once 的 trajectory-scale execution model）：

\[
\boxed{
\textbf{PB first. Trajectory-scale. OpenMM-native.}
}
\]
