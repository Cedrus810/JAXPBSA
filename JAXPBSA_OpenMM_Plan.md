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
