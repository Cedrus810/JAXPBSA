"""JAX Shrake–Rupley 后端（stage 2，DESIGN.md §3.10）—— **在图里**，可 jit/vmap。

SA 的**唯一**实现。对拍参照物 zsasa（`tests/zsasa_ref.py`，不是后端也不是依赖）
是外部进程：S4 上三个 species ~131 ms/帧，且进不了 `lax.scan`；这里 13.7 ms/帧。

**核心恒等式**（省掉了 [N,P,N] 的距离张量）。球 i 上的采样点
`p = x_i + R_i·u`（R_i = r_i + probe，u 是单位向量）被球 j 埋住的条件是
`|p − x_j|² < R_j²`，展开：

    |p − x_j|² = d_ij² + R_i² + 2 R_i · u·(x_i − x_j)

    埋住  ⟺  2 R_i · u·(x_i − x_j)  <  R_j² − R_i² − d_ij²
             └──── [P,K] 一次矩阵乘 ────┘   └── [K] 每对一个常数 ──┘

所以每个原子只是 `u[P,3] @ disp[K,3]ᵀ`，**不需要显式构造采样点坐标**。

同一个恒等式还免掉了截断判断：对 u 取极值 `min(u·disp) = −d`，可知
j 能埋住 i 的必要条件是 `|d − R_i| < R_j`，即 `d < R_i + R_j`。
所以**远处的邻居自动无效**，padding 项（d² = ∞ ⇒ rhs = −∞）也自动无效 ——
不用写掩码。自身项 disp = 0 同理（`0 < 0` 为假）。

**K 近邻是有损的**，所以 `sasa()` 会验，不够就报错，不静默给一个偏大的面积。

判据：`K ≥ maxᵢ |{j : d_ij < R_i + R_max}|`。**充分性**：不在该集合里的原子
距离 ≥ R_i + R_max，比集合里每一个都远，所以 top_k 取最近的 K 个必然把整个集合
装下 —— 而任何可能的遮挡者都在集合里（上面那条 `d < R_i + R_j` 的必要条件）。

**注意不能换成更紧的 `|{j : d_ij < R_i + R_j}|`**（S4 上 116 vs 136，看着省 15%）。
那个计数不是充分条件：top_k 按**距离**排序，一个近处的小原子（不遮挡）会挤掉一个
远处的大原子（真遮挡）。半径不进排序，所以界必须用 R_max。

**为什么不自动探测**（两条都试过，都不安全，实测）：

    来源                      所需 k
    CIF (真空最小化制备态)      131      ← 不是 MD 系综的构象, 欠 5
    MD 帧 0                    126      ← 欠 10
    MD 40 帧 (10 ns) 跨度   123 – 136

单帧探不出系综的上界，热运动会挤出更密的局部；而 k 是 static argnum，逐块重探还会
反复触发重编译。默认 192 对这条轨迹有 40% 余量，且这个数是**堆积密度**限的
（半径 6.4 Å 球内 ~0.1 原子/Å³ ≈ 110），换个折叠蛋白量级不变。
真正兜底的是下面这道闸 —— 它直接报出该填多少，一次重试必中。
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from .. import ACCUM_DTYPE
from .. import dtype as _dtype
from ..constants import PROBE_RADIUS


def golden_spiral(n: int) -> np.ndarray:
    """[n,3] 单位球面上的 golden-spiral 点集。host 侧常量, 所有原子共用一套。

    比随机点收敛快（离散度 ~1/n 而非 ~1/√n）, 且确定性 —— 同样输入逐位可复现。
    """
    if n < 1:
        raise ValueError(f"n_points 必须 ≥ 1, 得到 {n}")
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + math.sqrt(5.0)) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=-1)


def _frame(coords, R, u, k, chunk):
    """单帧 -> (逐原子面积 [N] Ų, 本帧实际所需的 k 标量)。

    **邻居搜索在逐原子块内做, 不物化 [N,N]。** 早先版本先建整张 d2 再分块跑采样点,
    结果显存卡在 d2 上: N=16k 时 977 MB/帧, N=32k 直接 OOM —— 而时间对 N 几乎是
    线性的(主成本是下面那次 [P,K] 矩阵乘)。挪进块里以后峰值是 chunk×(N + P·K),
    N 的二次项没了, 代码也少一层。
    """
    rmax = jnp.max(R)

    def per_atom(a):
        x_i, R_i, i = a
        d2 = ((x_i - coords) ** 2).sum(-1)  # [N]
        d2 = d2.at[i].set(jnp.inf)          # 自身不占近邻名额(它本来也无害: disp=0)
        neg, idx = jax.lax.top_k(-d2, k)    # top_k 取最大 -> -d2 最大 = d2 最小
        disp = x_i - coords[idx]            # [k,3]
        rhs = R[idx] ** 2 - R_i ** 2 + neg  # neg = -d², padding/远邻居自动 -> -inf
        exposed = ~jnp.any(2.0 * R_i * (u @ disp.T) < rhs, axis=-1)  # [P]
        # 所需 k 就在同一张 d2 上多一次归约, 不额外访存
        return jnp.count_nonzero(exposed), jnp.count_nonzero(d2 < (R_i + rmax) ** 2)

    # ponytail: chunk 同时限 d2([chunk,N]) 和采样点张量([chunk,P,K]); 后者大得多。
    n_exposed, k_need = jax.lax.map(
        per_atom, (coords, R, jnp.arange(coords.shape[0])), batch_size=chunk)

    # 面积算术升 fp64: 4πR² ~ 40 Ų 逐原子求和上千项, 且 count/P 要精确
    area = (4.0 * np.pi / u.shape[0]) * (
        R.astype(ACCUM_DTYPE) ** 2 * n_exposed.astype(ACCUM_DTYPE))
    return area, jnp.max(k_need)


@functools.partial(jax.jit, static_argnums=(3, 4))
def _batch(coords, R, u, k, chunk):
    return jax.vmap(_frame, in_axes=(0, None, None, None, None))(
        coords, R, u, k, chunk)


def sasa_core(coords, radii, probe_radius=PROBE_RADIUS, n_points=960,
              k_neighbors=192, chunk=None):
    """图内入口: [B,N,3] Å -> (逐原子面积 [B,N] Ų, 逐帧实际所需的 k [B])。

    **不做检查** —— 那要读回 host。批量分析走 `sasa()`, 它替你验。
    """
    dt = _dtype()
    c = jnp.asarray(coords, dt)
    r = jnp.asarray(radii, dt)
    if c.ndim != 3:
        raise ValueError(f"sasa_core 收 [B,N,3], 得到 {c.shape}")
    if c.shape[1] != r.shape[0]:
        raise ValueError(f"coords {c.shape} 与 radii {r.shape} 原子数不一致")
    k = min(int(k_neighbors), c.shape[1])
    if chunk is None:
        # 峰值 ≈ B × chunk × (P·K + N) 个元素；定在 64M = 256 MB fp32。
        # **必须除以 B**：分块的收益只在 B 小的时候才需要拿显存换 —— 实测 S4,
        # B=1 时 chunk 42→256 是 14.4→10.0 ms(-31%), 而 B=16 时 6.96→6.74
        # 几乎没差(批维已经把 GPU 喂饱了)。不除 B 就会在 B=16 上白占 5.8 GB。
        chunk = max(1, 64_000_000 // (c.shape[0] * (n_points * k + c.shape[1])))
    u = jnp.asarray(golden_spiral(n_points), dt)
    return _batch(c, r + jnp.asarray(probe_radius, dt), u, k, min(chunk, c.shape[1]))


def sasa(coords, radii, probe_radius=PROBE_RADIUS, n_points=960, per_atom=False,
         k_neighbors=192, chunk=None, check=True):
    """[B,N,3] 或 [N,3] -> [B] 或标量（`per_atom=True` 时 [B,N] / [N]）。

    `probe_radius` 默认取 `constants.PROBE_RADIUS` —— 与 PB 同一个数,
    这是 SA/PB 之间唯二该共享的东西之一 (另一个是 rᵢ)。
    """
    c = np.asarray(coords, dtype=np.float64)
    if not np.isfinite(c).all():
        raise ValueError("坐标含 NaN/Inf")
    r = np.asarray(radii, dtype=np.float64).reshape(-1)
    if not (r > 0).all():
        raise ValueError("半径必须为正")

    single = c.ndim == 2
    area, k_need = sasa_core(c[None] if single else c, r, probe_radius,
                             n_points, k_neighbors, chunk)
    if check:
        need = int(jnp.max(k_need))
        if need > min(int(k_neighbors), c.shape[-2]):
            raise ValueError(
                f"k_neighbors={k_neighbors} 不够, 本批需要 ≥ {need}。"
                f"真遮挡会被丢掉, 面积偏大 —— 所以这里报错而不是返回。\n"
                f"  重试: sasa(..., k_neighbors={need})\n"
                f"  注意这是**本批**的值; 逐块处理长轨迹时取各块最大, "
                f"否则 k 变化会触发重编译。")
    out = area if per_atom else area.sum(-1)
    if single:
        out = out[0]
    return out if per_atom or not single else float(out)
