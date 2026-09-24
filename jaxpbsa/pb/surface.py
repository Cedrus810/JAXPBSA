"""Surface / dielectric / ion-accessibility maps (DESIGN.md §3.5).

All kernels are shape-static and vmap-compatible:

  vdW occupancy : rasterize atom spheres by scatter (N x K candidate nodes,
                  K = |ball offsets| of the largest radius — cheap, no gather)
  SES           : morphological closing by the probe ball
                  ses = erode(dilate(vdw, R_probe), R_probe)
  ion exclusion : dilate(ses, R_ion); κ̄² = 0 inside it
  dielectric    : ε_in inside SES / ε_out outside, harmonic-mean smoothing
                  over a ball of radius `swin` (APBS srfm smol equivalent)
  ε faces       : harmonic mean of adjacent nodes (APBS convention)
"""
from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np

from .grid import GridSpec


def ball_offsets(radius: float, h: float) -> jnp.ndarray:
    """Integer offsets o with |o|·h ≤ radius, always including (0,0,0). Host-side."""
    if radius <= 0:
        return jnp.zeros((1, 3), dtype=jnp.int32)
    m = int(np.ceil(radius / h))
    r2 = (radius / h) ** 2 + 1e-9
    offs = [
        (i, j, k)
        for i in range(-m, m + 1)
        for j in range(-m, m + 1)
        for k in range(-m, m + 1)
        if i * i + j * j + k * k <= r2
    ]
    return jnp.asarray(offs, dtype=jnp.int32)


def rasterize_spheres(
    coords: jnp.ndarray,  # [N,3] Å
    radii: jnp.ndarray,  # [N] Å
    grid: GridSpec,
    offsets: jnp.ndarray,  # [K,3] host 静态, 半径 ≥ max(radii) + h
) -> jnp.ndarray:  # bool [nx,ny,nz]
    origin = jnp.asarray(grid.origin, coords.dtype)
    n = jnp.asarray(grid.shape)
    cell = jnp.rint((coords - origin) / grid.h).astype(jnp.int32)  # 最近节点
    cand = cell[:, None, :] + offsets[None, :, :]  # [N,K,3]
    nodes = origin + cand.astype(coords.dtype) * grid.h
    d2 = ((coords[:, None, :] - nodes) ** 2).sum(-1)  # [N,K]
    inb = jnp.all((cand >= 0) & (cand < n), axis=-1)
    take = inb & (d2 <= radii[:, None] ** 2)
    # jit 下不可布尔花式索引: clip 索引 + scatter-max(take) —— 越界项 take=False 无效果
    cand = jnp.clip(cand, 0, n - 1)
    occ = jnp.zeros(grid.shape, dtype=bool)
    return occ.at[(cand[:, :, 0], cand[:, :, 1], cand[:, :, 2])].max(take)


def dilate(m: jnp.ndarray, offsets: jnp.ndarray) -> jnp.ndarray:
    """Binary dilation by the structuring element `offsets`.

    **偏移是 host 侧常量, 所以用静态 `lax.slice` 展开。** 原先写成 `fori_loop` +
    `dynamic_slice`(起点是 tracer), XLA 无法融合 —— 257 个偏移就是 257 个独立
    kernel, 每个都完整读一遍网格再写回。改静态起点后 XLA 把整条 max 链融掉:
    S4/h=0.5 上形态学从 **28.95 ms 降到 4.86 ms(6.0×), 结果逐位相同**。

    分组展开(chunk)试过, **完全没有额外收益** —— 静态起点之后 XLA 本来就跨整条
    Python 展开链融合, 所以这里就是一个朴素循环。

    (另一条试过但**不可行**的路: 把 ball(R) 拆成 m 次 ball(R/m) 膨胀。连续空间里
     ball(a)⊕ball(b)=ball(a+b), 格点上不成立 —— h=0.5、R=2.0 拆两次得到的等效
     结构元只有 185 个点 vs 目标 257, 对称差 72, 会改变表面定义。见 split_ball。)
    """
    offs = np.asarray(offsets)
    pad = int(np.abs(offs).max())
    padded = jnp.pad(m, pad, mode="constant", constant_values=False)
    nx, ny, nz = m.shape
    acc = m
    for o in offs:
        sx, sy, sz = int(pad - o[0]), int(pad - o[1]), int(pad - o[2])
        acc = jnp.maximum(acc, jax.lax.slice(
            padded, [sx, sy, sz], [sx + nx, sy + ny, sz + nz]))
    return acc


def erode(m: jnp.ndarray, offsets: jnp.ndarray) -> jnp.ndarray:
    return ~dilate(~m, offsets)


def split_ball(radius: float, h: float, parts: int = 2):
    """把 ball(R) 的膨胀拆成 `parts` 次 ball(R/parts) 的膨胀。

    连续空间里 Minkowski 和满足 ball(a) ⊕ ball(b) = ball(a+b), 所以
    dilate(X, ball(R)) = dilate(...dilate(X, ball(R/m))..., ball(R/m))。
    偏移数从 K ~ (4/3)π(R/h)³ 降到 m·K/m³ = K/m²。

    **格点上这个等式只是近似**: 离散球的 Minkowski 和比目标离散球略大/略方。
    返回 (offsets, 实际等效结构元, 与目标球的对称差), 由调用方决定能否接受。
    """
    sub = ball_offsets(radius / parts, h)
    acc = {(0, 0, 0)}
    for _ in range(parts):
        acc = {(a[0] + o[0], a[1] + o[1], a[2] + o[2])
               for a in acc for o in np.asarray(sub).tolist()}
    target = {tuple(o) for o in np.asarray(ball_offsets(radius, h)).tolist()}
    return sub, acc, (acc ^ target)


def harmonic_smooth(
    eps: jnp.ndarray,
    offsets: jnp.ndarray,
    outside_value: float,
) -> jnp.ndarray:
    """ε_s[p] = K / Σ_{q∈ball(p)} ε[q]⁻¹, 界外按 ε_out 计."""
    pad = int(np.abs(np.asarray(offsets)).max())
    padded = jnp.pad(1.0 / eps, pad, mode="constant", constant_values=1.0 / outside_value)

    def body(i, acc):
        o = offsets[i]
        start = [pad - o[0], pad - o[1], pad - o[2]]
        return acc + jax.lax.dynamic_slice(padded, start, list(eps.shape))

    inv_sum = jax.lax.fori_loop(0, offsets.shape[0], body, jnp.zeros_like(eps))
    return offsets.shape[0] / inv_sum


def ses_level(
    coords: jnp.ndarray,  # [N,3] Å
    radii: jnp.ndarray,  # [N] Å
    grid: GridSpec,
    probe_radius: float,
    sas_offsets: jnp.ndarray,  # 半径 ≥ r_max + probe + reach, host 静态
    reach_offsets: jnp.ndarray,  # 半径 reach, host 静态
    reach: float,  # Å, ≥ probe + 2h
) -> jnp.ndarray:
    """SES 的连续水平集 G(p), **G ≥ 0 = 溶剂**, 界面附近 ≈ 到 SES 的有符号距离。

    d(c) = min_a(|c − x_a| − r_a − R_p) 在 SAS 外侧是**精确**欧氏距离, 所以自由探针
    中心 c 周围 d(c) 以内全是自由中心, ball(c, R_p + d(c)) ⊂ 溶剂(无假阳性):

        G(p) = max_{格点 c: d(c) ≥ 0, |p − c| ≤ reach} (R_p + d(c) − |p − c|)

    二值的 erode(dilate(vdw)) 用格点球采样探针: h=0.75 时沿轴只够到 0.75 Å
    (R_p=1.4), 表面按方向偏最多 ~0.65 Å。这里接触面上误差是 O(l²/s) (~0.1 Å)。
    盒内没被任何原子覆盖的格点 d 取下界 `reach`, 只会低估 G, 不改符号。
    """
    origin = jnp.asarray(grid.origin, coords.dtype)
    n = jnp.asarray(grid.shape)
    cell = jnp.rint((coords - origin) / grid.h).astype(jnp.int32)
    cand = cell[:, None, :] + sas_offsets[None, :, :]  # [N,K,3]
    nodes = origin + cand.astype(coords.dtype) * grid.h
    d = jnp.sqrt(((coords[:, None, :] - nodes) ** 2).sum(-1)) - (radii[:, None] + probe_radius)
    inb = jnp.all((cand >= 0) & (cand < n), axis=-1)
    cap = jnp.asarray(reach, coords.dtype)
    d = jnp.where(inb, jnp.minimum(d, cap), cap)
    cand = jnp.clip(cand, 0, n - 1)
    dsas = jnp.full(grid.shape, cap, coords.dtype).at[
        (cand[:, :, 0], cand[:, :, 1], cand[:, :, 2])].min(d)

    neg = jnp.asarray(-(reach + probe_radius), coords.dtype)  # 深处的下限, 保持有限
    src = jnp.where(dsas >= 0, dsas + probe_radius, neg)
    offs = np.asarray(reach_offsets)
    lens = np.sqrt((offs.astype(np.float64) ** 2).sum(-1)) * grid.h
    pad = int(np.abs(offs).max())
    padded = jnp.pad(src, pad, mode="constant", constant_values=reach + probe_radius)
    nx, ny, nz = grid.shape
    acc = jnp.full(grid.shape, neg, coords.dtype)
    for o, ln in zip(offs, lens):  # 静态起点, 同 dilate: XLA 融合整条链
        sx, sy, sz = int(pad - o[0]), int(pad - o[1]), int(pad - o[2])
        acc = jnp.maximum(acc, jax.lax.slice(
            padded, [sx, sy, sz], [sx + nx, sy + ny, sz + nz]) - float(ln))
    return acc


def _fraction_faces(g: jnp.ndarray, eps_in: float, eps_out: float, axis: int):
    """面上 ε: 按水平集线性插值求边上溶质(G<0)占的比例 θ, 串联(调和)混合。"""
    gi = jax.lax.slice_in_dim(g, 0, g.shape[axis] - 1, axis=axis)
    gj = jax.lax.slice_in_dim(g, 1, g.shape[axis], axis=axis)
    ini, inj = gi < 0, gj < 0
    t = gi / jnp.where(ini != inj, gi - gj, 1.0)  # 穿越点距 i 的比例
    theta = jnp.where(ini & inj, 1.0, jnp.where(ini == inj, 0.0, jnp.where(ini, t, 1.0 - t)))
    theta = jnp.clip(theta, 0.0, 1.0)
    return 1.0 / (theta / eps_in + (1.0 - theta) / eps_out)


def build_maps(
    coords: jnp.ndarray,  # [N,3] Å
    radii: jnp.ndarray,  # [N] Å
    grid: GridSpec,
    eps_in: float,
    eps_out: float,
    probe_radius: float,
    ion_radius: float,
    kappa2_phys: float,  # 物理 κ² (Å⁻²), 见 constants.debye_kappa2
    swin: float = 0.0,
    raster_offsets: jnp.ndarray | None = None,
    probe_offsets: jnp.ndarray | None = None,
    ion_offsets: jnp.ndarray | None = None,
    smooth_offsets: jnp.ndarray | None = None,
    level_offsets: tuple | None = None,  # (sas_offsets, reach_offsets, reach) -> 分数面 ε
) -> dict:
    """Returns eps [nx,ny,nz], face maps (eps_x/y/z), kbar2, ses.

    `level_offsets` 给定时走分数面 ε(`ses_level` + `_fraction_faces`, swin 不用);
    否则是二值 SES + 节点调和平均。"""
    if level_offsets is not None:
        sas_o, reach_o, reach = level_offsets
        if ion_offsets is None:
            ion_offsets = ball_offsets(ion_radius, grid.h)
        g = ses_level(coords, radii, grid, probe_radius, sas_o, reach_o, reach)
        ses = g < 0
        eps = jnp.where(ses, eps_in, eps_out).astype(coords.dtype)
        kbar2 = eps * kappa2_phys * (~dilate(ses, ion_offsets))
        return {"eps": eps, "kbar2": kbar2, "ses": ses,
                **{k: _fraction_faces(g, eps_in, eps_out, a).astype(coords.dtype)
                   for a, k in enumerate(("eps_x", "eps_y", "eps_z"))}}
    if raster_offsets is None:
        raster_offsets = ball_offsets(float(np.max(np.asarray(radii))) + grid.h, grid.h)
    if probe_offsets is None:
        probe_offsets = ball_offsets(probe_radius, grid.h)
    if ion_offsets is None:
        ion_offsets = ball_offsets(ion_radius, grid.h)

    vdw = rasterize_spheres(coords, radii, grid, raster_offsets)
    ses = erode(dilate(vdw, probe_offsets), probe_offsets)
    ion_excluded = dilate(ses, ion_offsets)

    # 同 charges.py: 两个 Python float 的 where 在 x64 下会给 float64
    eps = jnp.where(ses, eps_in, eps_out).astype(coords.dtype)
    if smooth_offsets is None and swin > 0:
        smooth_offsets = ball_offsets(swin, grid.h)
    if smooth_offsets is not None and smooth_offsets.shape[0] > 1:
        eps = harmonic_smooth(eps, smooth_offsets, eps_out)

    # κ̄² = ε(r)·κ², 仅离子可及区非零 (DESIGN.md §1/§3.5; 平滑壳层里两者同步渐变)
    kbar2 = eps * kappa2_phys * (~ion_excluded)

    eps_x = 2 * eps[:-1, :, :] * eps[1:, :, :] / (eps[:-1, :, :] + eps[1:, :, :])
    eps_y = 2 * eps[:, :-1, :] * eps[:, 1:, :] / (eps[:, :-1, :] + eps[:, 1:, :])
    eps_z = 2 * eps[:, :, :-1] * eps[:, :, 1:] / (eps[:, :, :-1] + eps[:, :, 1:])
    return {"eps": eps, "eps_x": eps_x, "eps_y": eps_y, "eps_z": eps_z,
            "kbar2": kbar2, "ses": ses}
