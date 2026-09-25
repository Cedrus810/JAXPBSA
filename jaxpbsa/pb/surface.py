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
    reach: float,  # Å, = probe + 2h
    crease: bool = True,
) -> jnp.ndarray:
    """SES 的连续水平集 G(p), **G ≥ 0 = 溶剂**, 界面附近 ≈ 到 SES 的有符号距离。

    精确定义: F = 自由探针中心区, G(p) = R_p − dist(p, F)(p 在 SAS 内), R_p + d(p)(p 在外),
    d(c) = min_a(|c − x_a| − r_a − R_p)。任何已知的自由中心 y 与半径 R_p + d(y) 给出下界
    R_p + d(y) − |p − y| —— 只要 y 真的自由, 就**没有假阳性**。两类候选:

    1. 格点 c(d(c) ≥ 0)。接触面(最近自由点在单个 SAS 球面上)误差 O(h²)。
    2. `crease=True`: 每个格点取 SAS 意义下最近的两个原子, 解析求它们 SAS 交线圆上离该点
       最近的点 y*, 再对其余原子精确检查 y* 自由。凹面(reentrant, 最近自由点在交线圆上)
       只靠格点时是 O(h) 且恒偏溶质 —— 双球解析对照 h=0.5 平均 −0.10 Å, 0.25 −0.047 Å
       (`data/md/ses_level_audit.py`); 结合界面的缝隙几乎全是这种面。三球顶点处 y* 可能被
       第三球挡住而作废, 退回格点候选(仍是下界)。

    **盒外不是溶剂**: 在向外扩 pad = ceil(reach/h) 的网格上算完再裁回。原先用常数把盒外
    填成自由中心, 盒面切过溶质时(focusing 细盒)会凭空造出 R_p + reach 厚的假溶剂层。
    """
    dt = coords.dtype
    h = grid.h
    offs = np.asarray(reach_offsets)
    pad = int(np.abs(offs).max())
    ext_shape = tuple(int(k) + 2 * pad for k in grid.shape)
    origin = jnp.asarray([o - pad * h for o in grid.origin], dt)
    n = jnp.asarray(ext_shape)
    R = radii + probe_radius
    cell = jnp.rint((coords - origin) / h).astype(jnp.int32)
    cand = cell[:, None, :] + sas_offsets[None, :, :]  # [N,K,3]
    nodes = origin + cand.astype(dt) * h
    inb = jnp.all((cand >= 0) & (cand < n), axis=-1)
    # 截断到 reach − h 而不是 reach: 候选球以 rint 格心为心, 边缘节点可能漏掉(真 d 低至
    # reach − 0.87h), 截断值必须 ≤ 真 d 才保持下界。d > reach − h 的中心只影响 G > R_p − h 的
    # 深溶剂节点, 不碰界面带。
    cap = jnp.asarray(reach - h, dt)
    d = jnp.where(inb, jnp.minimum(jnp.sqrt(((coords[:, None, :] - nodes) ** 2).sum(-1))
                                   - R[:, None], cap), cap)
    cand = jnp.clip(cand, 0, n - 1)
    ix = (cand[:, :, 0], cand[:, :, 1], cand[:, :, 2])
    d1 = jnp.full(ext_shape, cap, dt).at[ix].min(d)

    neg = jnp.asarray(-(reach + probe_radius), dt)  # 深处的下限, 保持有限
    src = jnp.where(d1 >= 0, d1 + probe_radius, neg)

    if crease:
        aid = jnp.arange(coords.shape[0], dtype=jnp.int32)[:, None]
        none = jnp.full(ext_shape, -1, jnp.int32)
        a1 = none.at[ix].max(jnp.where(inb & (d < cap) & (d == d1[ix]), aid, -1))
        dx = jnp.where(aid == a1[ix], cap, d)
        d2 = jnp.full(ext_shape, cap, dt).at[ix].min(dx)
        a2 = none.at[ix].max(jnp.where(inb & (dx < cap) & (dx == d2[ix]), aid, -1))
        i1, i2 = jnp.maximum(a1, 0), jnp.maximum(a2, 0)
        x1, x2, r1, r2 = coords[i1], coords[i2], R[i1], R[i2]
        axes = [jnp.arange(k, dtype=dt) * h for k in ext_shape]
        q = origin + jnp.stack(jnp.meshgrid(*axes, indexing="ij"), -1)
        u = x2 - x1
        dd = jnp.sqrt((u ** 2).sum(-1))
        nv = u / jnp.maximum(dd, 1e-6)[..., None]
        t = (dd ** 2 + r1 ** 2 - r2 ** 2) / (2 * jnp.maximum(dd, 1e-6))
        rho2 = r1 ** 2 - t ** 2
        m = x1 + t[..., None] * nv
        w = q - m
        v = w - (w * nv).sum(-1, keepdims=True) * nv
        vn = jnp.sqrt((v ** 2).sum(-1))
        y = m + (jnp.sqrt(jnp.maximum(rho2, 0.0)) / jnp.maximum(vn, 1e-6))[..., None] * v
        # |q − y| ≤ 2h: 采样够密(沿圆间距 ≤ h), 且任何盖住 y 的原子一定在 q 的候选球里
        ok = ((a1 >= 0) & (a2 >= 0) & (rho2 > 0) & (vn > 1e-6)
              & (((q - y) ** 2).sum(-1) <= (2 * h) ** 2))
        de = jnp.sqrt(((coords[:, None, :] - y[ix]) ** 2).sum(-1)) - R[:, None]
        de = jnp.where(inb & (aid != a1[ix]) & (aid != a2[ix]), de, cap)
        ok = ok & (jnp.full(ext_shape, cap, dt).at[ix].min(de) >= 0)
        yx, yy, yz = (jnp.where(ok, y[..., k], jnp.asarray(1e4, dt)) for k in range(3))
        px, py, pz = jnp.meshgrid(*[grid.origin[k] + jnp.arange(grid.shape[k], dtype=dt) * h
                                    for k in range(3)], indexing="ij")

    lens = np.sqrt((offs.astype(np.float64) ** 2).sum(-1)) * h
    nx, ny, nz = grid.shape
    acc = jnp.full(grid.shape, neg, dt)
    for o, ln in zip(offs, lens):  # 静态起点, 同 dilate: XLA 融合整条链
        st = [int(pad + o[0]), int(pad + o[1]), int(pad + o[2])]
        sl = lambda a: jax.lax.slice(a, st, [st[0] + nx, st[1] + ny, st[2] + nz])
        acc = jnp.maximum(acc, sl(src) - float(ln))
        if crease:
            dist = jnp.sqrt((px - sl(yx)) ** 2 + (py - sl(yy)) ** 2 + (pz - sl(yz)) ** 2)
            acc = jnp.maximum(acc, probe_radius - dist)
    return acc


def gaussian_density(
    coords: jnp.ndarray,  # [N,3] Å
    radii: jnp.ndarray,  # [N] Å, 0 = 屏蔽原子
    grid: GridSpec,
    sigma: float,
    offsets: jnp.ndarray,  # 半径 ≥ 3σ·r_max + h, host 静态
) -> jnp.ndarray:
    """DelPhi 高斯溶质密度 ρ = 1 − Π_i (1 − g_i), g_i = exp(−|r − r_i|² / (σR_i)²)。

    Li, Li, Zhang, Alexov, JCTC 9, 2126 (2013); σ = 0.93。逐原子截断在 3σR_i(同 DelPhi),
    Π 在 log 空间 scatter-add。**电荷中心附近 ε 按 r² 升高, 内层 ~σR·√(ε_in/Δε) ≈ 0.2 Å,
    h=0.5 下自项严重欠解析** —— 见 `gauss_selfcorr` 与 RESULTS §18.11–18.12。
    """
    dt = coords.dtype
    origin = jnp.asarray(grid.origin, dt)
    n = jnp.asarray(grid.shape)
    cell = jnp.rint((coords - origin) / grid.h).astype(jnp.int32)
    cand = cell[:, None, :] + offsets[None, :, :]  # [N,K,3]
    nodes = origin + cand.astype(dt) * grid.h
    r2 = ((coords[:, None, :] - nodes) ** 2).sum(-1)
    s2 = (sigma * radii[:, None]) ** 2
    live = jnp.all((cand >= 0) & (cand < n), axis=-1) & (radii[:, None] > 0) & (r2 <= 9.0 * s2)
    g = jnp.exp(-r2 / jnp.where(live, s2, 1.0))
    lg = jnp.where(live, jnp.log(jnp.maximum(1.0 - g, 1e-30)), 0.0)
    cand = jnp.clip(cand, 0, n - 1)
    acc = jnp.zeros(grid.shape, dt).at[(cand[:, :, 0], cand[:, :, 1], cand[:, :, 2])].add(lg)
    return 1.0 - jnp.exp(acc)


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
    gauss: tuple | None = None,  # (offsets, sigma) -> DelPhi 高斯 ε
) -> dict:
    """Returns eps [nx,ny,nz], face maps (eps_x/y/z), kbar2, ses.

    `level_offsets` 给定时走分数面 ε(`ses_level` + `_fraction_faces`, swin 不用);
    否则是二值 SES + 节点调和平均。"""
    if gauss is not None:
        # ε = ρ ε_in + (1 − ρ) ε_out 在节点上, 面上调和平均(同 binary)。离子可及度直接用 1 − ρ。
        g_off, sigma = gauss
        rho = gaussian_density(coords, radii, grid, sigma, g_off)
        eps = (rho * eps_in + (1.0 - rho) * eps_out).astype(coords.dtype)
        kbar2 = (eps_out * kappa2_phys * (1.0 - rho)).astype(coords.dtype)
        sl = lambda a, lo, hi, ax: jax.lax.slice_in_dim(a, lo, hi, axis=ax)
        faces = {k: 2 * sl(eps, 0, eps.shape[a] - 1, a) * sl(eps, 1, eps.shape[a], a)
                 / (sl(eps, 0, eps.shape[a] - 1, a) + sl(eps, 1, eps.shape[a], a))
                 for a, k in enumerate(("eps_x", "eps_y", "eps_z"))}
        return {"eps": eps, "kbar2": kbar2, "ses": rho > 0.5, **faces}
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
