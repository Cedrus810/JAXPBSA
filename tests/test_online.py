"""ONLINE_PLAN §4 的三条断言(不需要跑 MD)+ S1 的 S4 专项一次性检查。

用 conftest 的 peptide_system(双链小肽, A=受体 B=配体): 单位/索引/符号错误
与体系大小无关, 小体系在 CPU 上也跑得完。测试 2 与计划的差别: 不用
`padding=2.0` 触发负 margin —— margin 正负实际押在 APBS dime 取整的运气上
(peptide 这种扁平体系取整后可能仍为正), 断言不建立在运气上, 改为构造一个
明确越界的帧(平移 150 Å)打同一个守卫。
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest
from openmm import unit

import jaxpbsa

jaxpbsa.enable_compilation_cache()  # 测试 3 独立再建一份相同的 frame solver,
# 持久缓存让它命中(5.9 s)而不是重编译 36 s

from jaxpbsa.mm import mm_cross  # noqa: E402
from jaxpbsa.online import OnlineMMPBSA, PBSAReporter, recenter_com  # noqa: E402
from jaxpbsa.openmm_io import assign_radii, load_canonical  # noqa: E402
from jaxpbsa.pb import PBParams, TripletSolver  # noqa: E402
from jaxpbsa.sa import BETA_INP1, GAMMA_INP1, delta_g_sa  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_analyzer(peptide_system, **kw) -> OnlineMMPBSA:
    system, topology, pos = peptide_system
    lig_local = np.array([a.index for a in topology.atoms()
                          if a.residue.chain.id == "B"])
    solute_idx = np.arange(system.getNumParticles())  # 体系本身就是干的双链
    kw.setdefault("padding", 12.0)  # 测试不需要 30
    kw.setdefault("padding_lig", 8.0)
    return OnlineMMPBSA(system, topology, solute_idx, lig_local, pos, **kw), pos


def test_recenter_phase_stable():
    """归位必须用质心: 扰动单个极端原子 1 Å, 归位平移量几乎不动(< 0.05 Å)。
    包围盒中点会移 0.5 Å ≈ 一个 h, RESULTS §15.4 的摆放噪声(ΔG_PB 峰峰 8.39)
    就这样逐帧进在线 ΔG(t) —— 这条不需要求解, 只查平移量本身。"""
    rng = np.random.default_rng(1)
    n = 1835
    coords = rng.normal(scale=10.0, size=(n, 3))
    masses = rng.uniform(1.0, 16.0, size=n)
    s0 = recenter_com(coords, masses, np.zeros(3))
    pert = coords.copy()
    pert[int(np.argmax(pert[:, 0])), 0] += 1.0  # x 最大者 = 包围盒的极端原子
    s1 = recenter_com(pert, masses, np.zeros(3))
    # 比的是**平移量**(shift = 归位后 − 原坐标), 不是逐原子坐标差 —— 受扰
    # 原子自己当然动了 1 Å, 要约束的是整个分子被挪了多少。包围盒中点这里是
    # 0.5 Å ≈ 一个 h。
    shift_change = (s1 - pert) - (s0 - coords)
    assert np.abs(shift_change).max() < 0.05


def test_translation_invariance(peptide_system):
    """契约 2: 平移是运行期操作 —— 随机平移 100 Å 后 ΔG_MM/PBSA 不变 (rtol 1e-5,
    fp32 下平移-归位的舍入差远小于此)。"""
    az, pos = _make_analyzer(peptide_system)
    out0 = az(pos)
    rng = np.random.default_rng(0)
    out1 = az(pos + rng.uniform(-100.0, 100.0, size=3))
    assert abs(out0["delta_g_mmpbsa"] - out1["delta_g_mmpbsa"]) \
        <= 1e-5 * abs(out0["delta_g_mmpbsa"])
    assert out1["margin_A"] > 0 and out1["converged"] and out1["sa_ok"]


def test_margin_guard(peptide_system, tmp_path):
    """契约 3: 越界必须被抓住, 不许静默丢原子。"""
    # 默认参数: 预热自检(P1-2)会拒收不收敛的配置, 守卫测试也要用能过自检的参数
    az, pos = _make_analyzer(peptide_system)
    solute_idx = np.arange(len(pos))

    bad = pos.copy()
    bad[0] += np.array([150.0, 0.0, 0.0])  # 一个原子飞出(配体跳 image / 解离的
    # 信号): 包围盒暴涨。注意**整体平移不行** —— 归位(契约 2)会把纯平移完全吸收,
    # 越界指的是构象(包围盒)超出网格, 不是位置。
    out_bad = az(bad)
    assert out_bad["margin_A"] < 0, "越界帧必须报负 margin, 而不是悄悄给偏小的 G_PB"
    out_good = az(pos)
    assert out_good["margin_A"] > 0 and out_good["margin_lig_A"] > 0
    # 配体紧盒单独守: 配体原子撑出紧盒(整体仍在 C/R 网格里)也必须报负
    lig0 = az._lig_local[0]
    bad_l = pos.copy()
    bad_l[lig0] += np.array([0.0, 20.0, 0.0])
    assert az(bad_l)["margin_lig_A"] < 0

    class _FakeSim:
        currentStep = 1000

    class _FakeState:
        def __init__(self, coords_A, time_ps):
            self._c, self._t = coords_A, time_ps

        def getPositions(self, asNumpy=False):
            return unit.Quantity(self._c / 10.0, unit.nanometer)  # Å→nm, 走真实换算

        def getTime(self):
            return unit.Quantity(self._t, unit.picosecond)

    rep = PBSAReporter(az, 500, solute_idx)  # 6 元组协议, OpenMM ≥ 7.7
    assert rep.describeNextReport(_FakeSim()) == (500, True, False, False, False, False)

    # 回归: steps 必须是「距下次报告还差几步」。恒返回 interval 的实现在这里会给
    # 500, 而 OpenMM 只触发 steps == min(全部 reporter) 的那些 —— 同时挂 DCD@500
    # 时本 reporter 就永远不触发(实测 150k 步 0 帧)。
    class _FakeSimMid:
        currentStep = 1200

    assert rep.describeNextReport(_FakeSimMid())[0] == 300

    with pytest.raises(RuntimeError, match="margin"):
        PBSAReporter(az, 500, solute_idx, on_violation="raise").report(
            _FakeSim(), _FakeState(bad, 100.0))

    csv_path = str(tmp_path / "flag.csv")
    rep_flag = PBSAReporter(az, 500, solute_idx, out_csv=csv_path)
    rep_flag.report(_FakeSim(), _FakeState(bad, 100.0))  # flag: 落盘继续, 不抛
    rep_flag.close()
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1 and float(rows[0]["margin_A"]) < 0


def test_warmup_rejects_bad_config(peptide_system):
    """P1-2: sa_k/padding/求解器这三类配置错误必须在 __init__ 死, 不等 MD 第
    一帧写进 CSV。sa_k=1 确定性触发 SA 分支(sasa 按设计 raise 而不是给偏大的
    面积); PB 用快参数 —— 反正 converged 检查也在同一条 raise 里。"""
    with pytest.raises(ValueError, match="预热自检.*SA"):
        _make_analyzer(peptide_system, sa_k=1,
                       pb_params=PBParams(precond="jacobi", max_iter=3))


def test_assembly_matches_manual(peptide_system):
    """在线路径 vs 手工拼法(validate_mmpbsa.py 的公式), 同一帧逐项相等。
    抓单位(nm/Å)、索引(全局/局部)、符号(β 差分后剩 −β)。"""
    az, pos = _make_analyzer(peptide_system)
    out = az(pos)

    system, topology, _ = peptide_system
    from jaxpbsa.openmm_io import extract_nonbonded
    mmp = extract_nonbonded(system)
    lig = np.array([a.index for a in topology.atoms() if a.residue.chain.id == "B"])
    rec = np.setdiff1d(np.arange(len(pos)), lig)

    # 手工路径: 质量自己从 System 提, 半径自己定, 独立建一份 TripletSolver
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(system.getNumParticles())])
    radii = assign_radii(topology)
    tri2 = TripletSolver(pos, masses, radii, rec, lig, PBParams(), h=0.5,
                         padding=12.0, h_lig=0.25, padding_lig=8.0)
    assert tri2.grid == az.triplet_solver.grid
    assert tri2.grid_lig == az.triplet_solver.grid_lig
    pb2 = tri2(pos, mmp.charge)
    c2 = recenter_com(pos, masses, np.zeros(3))

    mm2 = mm_cross(c2[None], mmp, lig, rec)
    dsa2, areas2 = delta_g_sa(c2, radii, rec, lig,
                              gamma=GAMMA_INP1, beta=BETA_INP1, k_neighbors=192)
    e_coul2 = float(np.asarray(mm2["e_coul_rl"]).reshape(-1)[0])  # 批维 [1]
    e_lj2 = float(np.asarray(mm2["e_lj_rl"]).reshape(-1)[0])
    total2 = e_coul2 + e_lj2 + float(pb2["delta_g_pb"]) + float(dsa2)

    for ours, ref in (("e_coul_rl", e_coul2),
                      ("e_lj_rl", e_lj2),
                      ("g_pb_complex", float(pb2["g_pb_complex"])),
                      ("g_pb_receptor", float(pb2["g_pb_receptor"])),
                      ("g_pb_ligand", float(pb2["g_pb_ligand"])),
                      ("delta_g_pb", float(pb2["delta_g_pb"])),
                      ("sasa_complex", float(areas2["complex"])),
                      ("sasa_receptor", float(areas2["receptor"])),
                      ("sasa_ligand", float(areas2["ligand"])),
                      ("delta_g_sa", float(dsa2)),
                      ("delta_g_mmpbsa", total2)):
        np.testing.assert_allclose(out[ours], ref, rtol=1e-10, atol=0.0)


def test_s4_canonical_one_time_checks():
    """S1 的 S4 专项检查(需要 load_canonical, 所以放测试而不是类里):
    切片净电荷、assign_radii 与 canonical radii 逐位相等。纯 host 侧, 无求解。"""
    d = load_canonical(root=ROOT)
    n = d["positions_A"].shape[0]
    q = d["charge"]
    assert round(q.sum()) == round(q[d["receptor_idx"]].sum()
                                   + q[d["ligand_idx"]].sum())
    radii = assign_radii(d["topology"], d["meta"]["radii_model"])
    assert np.array_equal(radii, d["radii"]), \
        "assign_radii 在 canonical topology 上的行为漂移 —— rᵢ 单源破了"


def test_triplet_shared_grid_matches_frame_solver(peptide_system):
    """`h_lig=None` 必须就是旧的公共网格 triplet, 只多一步质心归位 ——
    抓 TripletSolver 的建网格(center=质心)与接线。"""
    from jaxpbsa.openmm_io import extract_nonbonded
    from jaxpbsa.pb import make_frame_solver
    system, topology, pos = peptide_system
    q = extract_nonbonded(system).charge
    lig = np.array([a.index for a in topology.atoms() if a.residue.chain.id == "B"])
    rec = np.setdiff1d(np.arange(len(pos)), lig)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(len(pos))])
    radii = assign_radii(topology)
    tri = TripletSolver(pos, masses, radii, rec, lig, h=0.5, padding=12.0, h_lig=None)
    c = recenter_com(pos, masses, tri.grid.center)
    assert np.allclose(np.average(c, axis=0, weights=masses), tri.grid.center)
    ref = make_frame_solver(tri.grid, radii, PBParams()).triplet(c, q, rec, lig)
    out = tri(pos + 17.0, q)
    for k in ("g_pb_complex", "g_pb_receptor", "g_pb_ligand", "delta_g_pb"):
        np.testing.assert_allclose(float(out[k]), float(ref[k]), rtol=1e-5)


def test_asym_ligand_uses_its_own_com(peptide_system):
    """非对称网格下配体按**自己的**质心归位到自己的紧盒: 把配体整体挪离受体,
    G_L 不变(孤立溶剂化能与它在复合物里的位置无关)。若误用复合物质心, 配体
    相对紧盒平移 -> 相位变 / 越界, G_L 会动。"""
    from jaxpbsa.openmm_io import extract_nonbonded
    system, topology, pos = peptide_system
    q = extract_nonbonded(system).charge
    lig = np.array([a.index for a in topology.atoms() if a.residue.chain.id == "B"])
    rec = np.setdiff1d(np.arange(len(pos)), lig)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(len(pos))])
    tri = TripletSolver(pos, masses, assign_radii(topology), rec, lig,
                        padding=12.0, padding_lig=8.0)
    moved = pos.copy()
    moved[lig] += np.array([0.0, 3.3, -2.1])
    a, b = tri(pos, q), tri(moved, q)
    np.testing.assert_allclose(float(b["g_pb_ligand"]), float(a["g_pb_ligand"]), rtol=1e-5)
    assert a["margin_A"] > 0 and bool(a["converged"])
