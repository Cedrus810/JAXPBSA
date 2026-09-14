"""SA 项（T4）。

**当前后端 zsasa 是临时替补**，stage 2 会换成 JAX Shrake–Rupley。
所以这些测试有两个职责：

1. 保证替补期间 SA 本身是对的（对解析解）；
2. **成为 JAX 版的验收基准** —— stage 2 写完后把 `BACKENDS` 加上 "jax"，
   同一批断言会同时跑在两个后端上，两者必须给出一致的结果。

zsasa 是外部可执行文件（PB 路径不需要它），找不到时跳过而不是失败。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from jaxpbsa.sa import BETA_INP1, GAMMA_INP1, delta_g_sa, g_sa, sasa

#: stage 2 落地后在这里加 "jax"，下面所有断言自动覆盖它。
BACKENDS = ["zsasa"]


def _available(backend):
    if backend != "zsasa":
        return True
    from jaxpbsa.sa.zsasa import find_binary
    try:
        find_binary()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.parametrize(
    "backend", [pytest.param(b, marks=pytest.mark.skipif(
        not _available(b), reason=f"{b} 后端不可用")) for b in BACKENDS])

PROBE = 1.4


def test_isolated_sphere_matches_analytic(backend):
    """孤立球: SASA = 4π(r+p)²。这是唯一有闭式解的情形。"""
    for r in (1.2, 1.7, 2.0):
        got = sasa(np.zeros((1, 3)), [r], backend=backend,
                   probe_radius=PROBE, n_points=4000)
        want = 4 * math.pi * (r + PROBE) ** 2
        assert abs(got - want) / want < 1e-6, (r, got, want)


def test_two_overlapping_spheres_match_analytic(backend):
    """两个等半径球重叠: 各被削去一个球冠, 冠高 h = R − d/2 (R = r+p)。"""
    r, d = 2.0, 4.0
    got = sasa(np.array([[0.0, 0, 0], [d, 0, 0]]), [r, r], backend=backend,
               probe_radius=PROBE, n_points=4000)
    R = r + PROBE
    want = 2 * (4 * math.pi * R ** 2 - 2 * math.pi * R * (R - d / 2))
    # Shrake–Rupley 是采样法, 4000 点下 ~1e-5 量级
    assert abs(got - want) / want < 1e-3, (got, want)


def test_far_apart_spheres_are_additive(backend):
    """相距极远时总面积 = 各自之和（无遮挡）。"""
    r = 1.7
    got = sasa(np.array([[0.0, 0, 0], [500.0, 0, 0]]), [r, r], backend=backend,
               probe_radius=PROBE, n_points=2000)
    assert abs(got - 2 * 4 * math.pi * (r + PROBE) ** 2) / got < 1e-6


def test_per_atom_sums_to_total(backend):
    rng = np.random.default_rng(0)
    c = rng.uniform(-6, 6, (25, 3))
    r = rng.uniform(1.2, 1.9, 25)
    per = sasa(c, r, backend=backend, per_atom=True, n_points=2000)
    tot = sasa(c, r, backend=backend, n_points=2000)
    assert per.shape == (25,)
    assert abs(per.sum() - tot) / tot < 1e-9


def test_batched_matches_per_frame(backend):
    rng = np.random.default_rng(1)
    traj = rng.uniform(-6, 6, (3, 12, 3))
    r = rng.uniform(1.2, 1.9, 12)
    batched = sasa(traj, r, backend=backend, n_points=2000)
    assert batched.shape == (3,)
    for i in range(3):
        one = sasa(traj[i], r, backend=backend, n_points=2000)
        assert abs(batched[i] - one) / one < 1e-12


def test_beta_does_not_cancel_in_the_difference(backend):
    """ΔG_SA = γ·ΔSASA − β。**三个 species 各有一份常数项, 差分后剩 −β。**

    Amber INP=1 的 β=0 会掩盖这一点; INP=2 (β=−0.5692) 就会差 0.57 kcal/mol。
    """
    rng = np.random.default_rng(2)
    c = rng.uniform(-8, 8, (16, 3))
    r = np.full(16, 1.7)
    rec, lig = np.arange(10), np.arange(10, 16)
    d0, areas = delta_g_sa(c, r, rec, lig, gamma=0.005, beta=0.0, backend=backend)
    d1, _ = delta_g_sa(c, r, rec, lig, gamma=0.005, beta=-0.5692, backend=backend)
    assert abs((d1 - d0) - 0.5692) < 1e-9, (d0, d1)
    # 与手工组合一致
    manual = 0.005 * (areas["complex"] - areas["receptor"] - areas["ligand"])
    assert abs(d0 - manual) < 1e-9


def test_g_sa_formula(backend):
    assert abs(float(g_sa(1000.0, 0.005, 0.0)) - 5.0) < 1e-12
    assert abs(float(g_sa(1000.0, 0.0378, -0.5692)) - (37.8 - 0.5692)) < 1e-9


def test_binding_buries_surface_on_s4(backend):
    """S4 实体: 结合必须**埋藏**表面 (ΔSASA < 0), 且量级合理。

    这条不是精度断言, 是防止索引/切片写反 —— 那种错误会让 ΔSASA 变正。
    """
    pytest.importorskip("openmm")
    from jaxpbsa.openmm_io import load_canonical
    d = load_canonical()
    dg, areas = delta_g_sa(d["positions_A"], d["radii"], d["receptor_idx"],
                           d["ligand_idx"], backend=backend, n_points=960)
    assert areas["delta_sasa"] < 0, "结合不可能增加溶剂可及面积"
    assert -2000 < areas["delta_sasa"] < -500, areas["delta_sasa"]
    assert areas["complex"] < areas["receptor"] + areas["ligand"]
    assert -20 < dg < 0, dg
