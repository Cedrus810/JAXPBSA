"""SA 项（T4）—— `jaxpbsa.sa`，JAX Shrake–Rupley。

断言分三层，**只有第一层是自洽的**：

1. **解析解**（孤立球、两球重叠、远离可加）—— 唯一有闭式解的情形。
2. **暴力法**（`test_burial_identity_against_brute_force`）—— 显式构造采样点
   `p_ik` 再逐个测距，验 `jax_sr` 那步「判据代数化成矩阵乘」的变换。
   解析球测试发现不了它：无邻居时判据两边都退化。
3. **zsasa 对拍**（`zsasa_ref`）—— 外部参照物，**不是后端也不是依赖**，
   找不到二进制就跳过。职责只有性能/结果对照。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import zsasa_ref
from jaxpbsa.constants import PROBE_RADIUS
from jaxpbsa.sa import BETA_INP1, GAMMA_INP1, delta_g_sa, g_sa, sasa


#: 探针半径的单源 —— 测试也不该自带一份（`test_pb_and_sa_share_one_probe_radius`）。
PROBE = PROBE_RADIUS


def _zsasa_available():
    """zsasa 是外部参照物, 不是依赖 —— 缺二进制跳过, 不算失败。"""
    try:
        zsasa_ref.find_binary()
        return True
    except FileNotFoundError:
        return False


def test_isolated_sphere_matches_analytic():
    """孤立球: SASA = 4π(r+p)²。这是唯一有闭式解的情形。"""
    for r in (1.2, 1.7, 2.0):
        got = sasa(np.zeros((1, 3)), [r],
                   probe_radius=PROBE, n_points=4000)
        want = 4 * math.pi * (r + PROBE) ** 2
        assert abs(got - want) / want < 1e-6, (r, got, want)


def test_two_overlapping_spheres_match_analytic():
    """两个等半径球重叠: 各被削去一个球冠, 冠高 h = R − d/2 (R = r+p)。"""
    r, d = 2.0, 4.0
    got = sasa(np.array([[0.0, 0, 0], [d, 0, 0]]), [r, r],
               probe_radius=PROBE, n_points=4000)
    R = r + PROBE
    want = 2 * (4 * math.pi * R ** 2 - 2 * math.pi * R * (R - d / 2))
    # Shrake–Rupley 是采样法, 4000 点下 ~1e-5 量级
    assert abs(got - want) / want < 1e-3, (got, want)


def test_far_apart_spheres_are_additive():
    """相距极远时总面积 = 各自之和（无遮挡）。"""
    r = 1.7
    got = sasa(np.array([[0.0, 0, 0], [500.0, 0, 0]]), [r, r],
               probe_radius=PROBE, n_points=2000)
    assert abs(got - 2 * 4 * math.pi * (r + PROBE) ** 2) / got < 1e-6


def test_per_atom_sums_to_total():
    rng = np.random.default_rng(0)
    c = rng.uniform(-6, 6, (25, 3))
    r = rng.uniform(1.2, 1.9, 25)
    per = sasa(c, r, per_atom=True, n_points=2000)
    tot = sasa(c, r, n_points=2000)
    assert per.shape == (25,)
    assert abs(per.sum() - tot) / tot < 1e-9


def test_batched_matches_per_frame():
    rng = np.random.default_rng(1)
    traj = rng.uniform(-6, 6, (3, 12, 3))
    r = rng.uniform(1.2, 1.9, 12)
    batched = sasa(traj, r, n_points=2000)
    assert batched.shape == (3,)
    for i in range(3):
        one = sasa(traj[i], r, n_points=2000)
        assert abs(batched[i] - one) / one < 1e-12


def test_beta_does_not_cancel_in_the_difference():
    """ΔG_SA = γ·ΔSASA − β。**三个 species 各有一份常数项, 差分后剩 −β。**

    Amber INP=1 的 β=0 会掩盖这一点; INP=2 (β=−0.5692) 就会差 0.57 kcal/mol。
    """
    rng = np.random.default_rng(2)
    c = rng.uniform(-8, 8, (16, 3))
    r = np.full(16, 1.7)
    rec, lig = np.arange(10), np.arange(10, 16)
    d0, areas = delta_g_sa(c, r, rec, lig, gamma=0.005, beta=0.0)
    d1, _ = delta_g_sa(c, r, rec, lig, gamma=0.005, beta=-0.5692)
    assert abs((d1 - d0) - 0.5692) < 1e-9, (d0, d1)
    # 与手工组合一致
    manual = 0.005 * (areas["complex"] - areas["receptor"] - areas["ligand"])
    assert abs(d0 - manual) < 1e-9


def test_g_sa_formula():
    assert abs(float(g_sa(1000.0, 0.005, 0.0)) - 5.0) < 1e-12
    assert abs(float(g_sa(1000.0, 0.0378, -0.5692)) - (37.8 - 0.5692)) < 1e-9


def test_binding_buries_surface_on_s4():
    """S4 实体: 结合必须**埋藏**表面 (ΔSASA < 0), 且量级合理。

    这条不是精度断言, 是防止索引/切片写反 —— 那种错误会让 ΔSASA 变正。
    """
    pytest.importorskip("openmm")
    from jaxpbsa.openmm_io import load_canonical
    d = load_canonical()
    dg, areas = delta_g_sa(d["positions_A"], d["radii"], d["receptor_idx"],
                           d["ligand_idx"], n_points=960)
    assert areas["delta_sasa"] < 0, "结合不可能增加溶剂可及面积"
    assert -2000 < areas["delta_sasa"] < -500, areas["delta_sasa"]
    assert areas["complex"] < areas["receptor"] + areas["ligand"]
    assert -20 < dg < 0, dg


def test_matches_zsasa_on_s4():
    """对 zsasa 参照物对拍（DESIGN.md §3.10 / T4）。zsasa 只做这件事。

    **只卡总面积 <1%**。逐原子对不上是预期的 —— 两边的球面采样点集不同
    （golden spiral vs zsasa 自己的），单个原子的暴露比例差在 ~1% 量级，
    实测 S4 上逐原子中位相对差 7e-3、总量相对差 6e-5（误差互相抵消）。
    """
    if not _zsasa_available():
        pytest.skip("zsasa 参照物不可用（它不是依赖，缺了不算失败）")
    pytest.importorskip("openmm")
    from jaxpbsa.openmm_io import load_canonical
    d = load_canonical()
    c, r = np.asarray(d["positions_A"], float), np.asarray(d["radii"], float)
    if c.ndim == 3:
        c = c[0]
    for n in (960, 4000):
        z = zsasa_ref.sasa_frame(c, r, n_points=n)
        j = float(sasa(c, r, n_points=n))
        assert abs(j - z) / z < 0.01, (n, z, j)


def test_rejects_insufficient_k_neighbors():
    """K 近邻不够时**必须报错**, 不能静默返回偏大的面积。

    S4 内部原子的 R_i+R_max 邻域里超过 128 个原子 —— 实测 CIF 131、MD 帧 123–136。
    没有这道闸, 换个更致密的体系就会悄悄多算表面。
    """
    pytest.importorskip("openmm")
    from jaxpbsa.openmm_io import load_canonical
    d = load_canonical()
    c, r = np.asarray(d["positions_A"], float), np.asarray(d["radii"], float)
    if c.ndim == 3:
        c = c[0]
    with pytest.raises(ValueError, match="k_neighbors"):
        sasa(c, r, n_points=960, k_neighbors=128)


def test_burial_identity_against_brute_force():
    """核心恒等式对拍暴力法: 显式构造采样点、显式算 |p−x_j|。

    `jax_sr` 从不构造采样点坐标 —— 它把埋藏判据代数化成一次 [P,3]×[3,K] 矩阵乘。
    这条测试就是那步代数变换的**唯一**独立验证; 变换写错的话解析球测试
    （无邻居, 恒等式两边都退化）根本发现不了。
    """
    from jaxpbsa.sa.jax_sr import golden_spiral
    rng = np.random.default_rng(7)
    x = rng.uniform(-4, 4, (20, 3))
    r = rng.uniform(1.2, 1.9, 20)
    R, u = r + PROBE, golden_spiral(1000)
    brute = 0.0
    for i in range(len(x)):
        p = x[i] + R[i] * u                                    # [P,3] 显式采样点
        d = np.linalg.norm(p[:, None, :] - x[None, :, :], axis=-1)  # [P,N]
        occ = (d < R[None, :]) & (np.arange(len(x)) != i)
        brute += 4 * math.pi * R[i] ** 2 * (~occ.any(1)).mean()
    got = float(sasa(x, r, probe_radius=PROBE, n_points=1000))
    # 判据本身是逐点布尔的, 所以**一个点判反**就是 4πR²/P ≈ 2e-4 的相对差。
    # 卡 1e-6 比那小 200 倍 = 两边埋藏判定逐点相同; 残差只是 R 存成 fp32。
    assert abs(got - brute) / brute < 1e-6, (got, brute)
