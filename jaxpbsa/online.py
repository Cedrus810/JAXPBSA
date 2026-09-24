"""在线 MM/PBSA: 固定网格 + 每帧归位 (ONLINE_PLAN.md; plan §21–§23 落地第 2 步).

第 1 步(stage-2 JAX SA)已在 `sa/jax_sr.py`。本文件是**同步原型**:
MD 循环经 `PBSAReporter` 逐帧调用, analysis 与 MD 同卡分时 —— async worker /
双 GPU 明确不做, 等 `scripts/online_overhead.py` 的实测数说话(ONLINE_PLAN §5)。

## 三条契约 (ONLINE_PLAN §0, 违反任何一条都是静默错数)

1. **网格建一次, 全程不动**(`origin`/`shape`/`h`)。`make_grid` 只在 `__init__`
   里调一次, 不暴露给调用方 —— 在线循环里重建网格 = 每帧重编译。
2. **每帧只动坐标**: 切溶质 → **质心**归位到网格中心(见 `recenter_com`, 不是
   包围盒中点) → 喂求解器。坐标平移是运行期操作, 实测不触发重编译、`G_PB`
   不变(plan §21.5 约束 3: 平移 origin 2.59 s 重编译, 平移坐标稳态 23.5 ms
   且能量不变)。
3. **越界不静默**。超出 padding 的原子当前被 `clip + 权重置零` 悄悄丢掉
   (`pb/charges.py` `_weights` / `pb/surface.py` `rasterize_spheres`) —— 不报错,
   只给一个偏小的 `G_PB`。所以每帧输出 `margin_A`, 由 reporter 按
   `on_violation` 策略处理(在线跑几小时, 一帧异常不能杀 MD, 但必须留下标记)。
   注意 `margin_A ≥ 0` 只保证没丢原子; Dirichlet 边界够不够远是另一回事,
   由 reporter 的 `margin_min` 管(S4 传 12 ≈ 1.5κ⁻¹, RESULTS §15.8)。

## origin 为什么**不**改成运行期参数 (ONLINE_PLAN §1)

plan §21.5 说「origin 运行期化 **或** origin 进 artifact 键, 二选一」。Python
阶段后者已自动满足: `enable_compilation_cache()` 的查找键是编译产物(HLO)哈希,
而 origin 烤在 HLO 里 —— **键天然完备, 不存在跨体系错配**。运行期化的唯一收益
(跨体系复用同一份 executable)只在 PJRT artifact store 阶段存在。

**失效条件**: 一旦做显式 artifact store / 跨 Context 复用 executable, 必须回来
把 origin 改成运行期 buffer 参数(它有 5 处使用点, 其中 `shell_indices` 是 host
侧 numpy) —— 那时它是 PJRT 阶段的前置项, 不是现在。

## padding=30.0 的来源 (ONLINE_PLAN S0, 2026-09-20, S4 干轨迹 10 ns / 10000 帧)

(2026-09-24 起 C/R 网格默认 h=0.5(0.75 有 −7~−11 的真离散偏差, RESULTS §18.8)、配体单独紧盒, 见 `pb.TripletSolver`。padding
仍以 Å 计, 下面的涨落预算照旧成立; 网格形状与 margin 数字正是 h=0.5 时量的。)

在线只有第 0 帧, 必须显式补上构象涨落: 每轴包围盒半长的轨迹最大值比制备态
大 **6.4 Å**(比 frame 0 大 3.5), + 介电/离子膨胀 5.7(r_max 1.8 + probe 1.4 +
ion 2.0 + swin 0.5) + 边界物理需求 12(RESULTS §15.8: ~1.5κ⁻¹, 0.03 kcal/mol)
⇒ **需要 ≈24 Å**。实测 padding=30 建网格(225×193×225, dime 取整额外送 5–13 Å)
后, **质心归位**下 10 ns 全部 10000 帧 `margin_A ≥ +16.5`(包围盒归位是 +18.5,
质心偏离中点最多 4.6 Å 吃掉一截, 由 dime 余量吸收)。**换体系/更长轨迹要重量
这个数。**

## 配体紧盒 padding_lig=14 的来源 (2026-09-23, 同一条 10 ns 干轨迹)

非对称网格(`TripletSolver`)下配体按**自己的**质心归位到自己的紧盒。离线 8 Å
(≈1κ⁻¹, 与 20 Å 差 0.09 kcal/mol, RESULTS §15.8) 够用, 因为离线按整条轨迹的
范围建盒; 在线只有参考帧, 磷酸肽很软: 每轴 max|x−COM_L| 制备态
[16.0, 7.5, 12.3] → 轨迹最大 [17.8, 12.7, 16.1], **y 轴涨 5.2 Å**(1 ns 前缀只到
+3.8, 又是单调增的 max 统计量)。8 + 6 = **14**。换配体必须重量。

## 索引约定 (最容易错的地方 —— 两套索引混用不报错, 只给一个错的 ΔG)

- `solute_idx`: **溶剂化体系**的全局索引(切水/离子用)。
- `ligand_local_idx`: 切完之后 `[0, N_solute)` 里的局部索引。
- `receptor_local` 是补集, 在 `__init__` 里算, 不让调用方给。

## 不做清单 (ONLINE_PLAN §5, 免得后面重捡)

async worker/双 GPU(等 S6 实测) · `jax.export`/PJRT plugin(跟随上游 #5320)
· 零拷贝(占 1e-5) · §23 的 JSD/ESS/自相关(CSV 落盘离线算) · SA 融进同一个
jit(PB 占 91.5%, 先优化错的项) · triplet warm start(相隔 N ps 的帧构象无关,
1.08× 不值得) · origin 运行期参数(见上)。
"""
from __future__ import annotations

import time

import numpy as np
from openmm import unit

from .mm import mm_cross_prepared, prepare_cross
from .openmm_io import MMParams, assign_radii, extract_nonbonded
from .pb import PBParams, TripletSolver, recenter_com
from .sa import BETA_INP1, GAMMA_INP1, delta_g_sa

__all__ = ["OnlineMMPBSA", "PBSAReporter", "recenter_com"]


class OnlineMMPBSA:
    """`__call__(coords_A[N_solute,3] Å) -> dict`, 网格/参数全部在 `__init__` 冻结.

    配合 `jaxpbsa.enable_compilation_cache()` 使用可把 `__init__` 末尾的预热编译
    从 ~37 s 压到 ~6 s(RESULTS §15.6) —— 库不替调用方改全局 `jax.config`, 所以
    这行由调用方脚本自己写(与 `enable_compilation_cache` 的设计一致)。
    """

    def __init__(
        self,
        system,
        topology,
        solute_idx,
        ligand_local_idx,
        ref_coords_A,
        *,
        h: float = 0.5,
        padding: float = 30.0,
        h_lig: float | None = 0.25,
        padding_lig: float = 14.0,
        pb_params: PBParams | None = None,
        gamma: float = GAMMA_INP1,
        beta: float = BETA_INP1,
        sa_k: int = 192,
        sa_points: int = 960,
        radii_model: str = "mbondi2",
        net_charge: float | None = None,
    ):
        solute_idx = np.asarray(solute_idx, dtype=int)
        lig = np.asarray(ligand_local_idx, dtype=int)
        ref = np.asarray(ref_coords_A, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[1] != 3 or ref.shape[0] != solute_idx.size:
            raise ValueError(
                f"ref_coords_A 要 [{len(solute_idx)},3] Å, 得到 {ref.shape}")
        if lig.size == 0 or lig.max() >= solute_idx.size or lig.min() < 0:
            raise ValueError(
                f"ligand_local_idx 必须落在 [0, {solute_idx.size}) (切溶质**之后**的"
                "局部索引; 拿溶剂化体系的全局索引来用只会得到一个错的 ΔG)")
        rec = np.setdiff1d(np.arange(solute_idx.size), lig)
        if rec.size == 0:
            raise ValueError("ligand 覆盖了全部溶质原子, receptor 为空")

        # ---- S1: 参数装配, 全部 host 侧一次性 ----
        params = pb_params if pb_params is not None else PBParams(precond="auto")
        mmp = extract_nonbonded(system)  # 全溶剂化体系
        q = np.asarray(mmp.charge)[solute_idx]
        solute_mm = MMParams(  # exceptions 只服务 v1 cross-energy, 逐帧用不到
            charge=q, sigma=np.asarray(mmp.sigma)[solute_idx],
            epsilon=np.asarray(mmp.epsilon)[solute_idx], exceptions=[])
        # atom_indices 不能省: topology 是**溶剂化**体系, 尾部的 Na/Cl 不在
        # Bondi 表, 先算全体再切片会在切片之前就 raise(实测 S4 挂在 Na 23807)
        radii = np.asarray(assign_radii(topology, radii_model,
                                        atom_indices=solute_idx))[solute_idx]
        masses = np.array(
            [system.getParticleMass(i).value_in_unit(unit.dalton)
             for i in solute_idx], dtype=np.float64)

        # 一次性检查(不是每帧): 净电荷非整数值 = 切片索引错位的最便宜信号。
        # S4 之外更严格的对照(与 canonical 的净电荷/radii 逐位相等)在
        # tests/test_online.py 里 —— 那些需要 load_canonical, 类不该依赖它。
        if abs(q.sum() - round(q.sum())) > 1e-3:
            raise ValueError(
                f"溶质净电荷 {q.sum():+.6f} e 不是整数 —— solute_idx 切错的大概率"
                "是把水/抗衡离子切进来了或把溶质原子漏了")
        if net_charge is not None and round(q.sum()) != round(net_charge):
            raise ValueError(
                f"溶质净电荷 {q.sum():+.3f} e 与预期的 {net_charge:+.0f} e 不符"
                " —— 切片索引错位")

        self._tri = TripletSolver(ref, masses, radii, rec, lig, params, h=h,
                                  padding=padding, h_lig=h_lig,
                                  padding_lig=padding_lig)
        self._q = q
        self._radii = radii
        self._masses = masses
        self._rec_local = rec
        self._lig_local = lig
        self._cross = prepare_cross(solute_mm, lig, rec)
        self._sa_kw = dict(k_neighbors=int(sa_k), n_points=int(sa_points))
        self._gamma, self._beta = float(gamma), float(beta)
        # reporter 的单位哨兵: 帧的包围盒尺度不应偏离参考帧 3 倍以上
        # (nm/Å 忘 ×10 是 10 倍, 必被抓; ps 级构象变化不会)
        self._ref_extent = float(np.linalg.norm(ref.max(0) - ref.min(0)))

        # ---- 预热 + 自检: 编译(~秒级 ×5 份)挡在 MD 开跑之前, 否则第一次
        # report 阻塞 MD 十几秒, 还会污染 online_overhead 的测量。
        # 预热结果必须验(ONLINE_PLAN S1 的验收): sa_k/padding/求解器这三类
        # **配置**错误在 __call__ 里都是「打标记不抛」, 不在这里死, 就要等 MD
        # 跑起来第一帧写进 CSV 才看得见。运行期的帧异常仍走 reporter 的 flag。
        t0 = time.perf_counter()
        warm = self(ref)
        self.warmup_time_s = time.perf_counter() - t0
        problems = []
        if min(warm["margin_A"], warm["margin_lig_A"]) < 0:  # nan(共用网格) 不触发
            problems.append(
                f"padding 不足: 预热帧 margin_A = {warm['margin_A']:.2f} / "
                f"margin_lig_A = {warm['margin_lig_A']:.2f} Å, "
                "越界原子已被 clip+权重置零静默丢弃 —— 加大 padding")
        if not warm["converged"]:
            problems.append(f"PB 预热帧未收敛: solver_iters = "
                            f"{warm['solver_iters']} —— 查 tol/max_iter/precond")
        if not warm["sa_ok"]:
            problems.append(f"SA 预热帧失败: {warm.get('sa_error', '')}")
        if problems:
            raise ValueError(
                "OnlineMMPBSA 预热自检失败(配置错误要在 MD 开跑前死, 不等第"
                "一帧): " + "; ".join(problems))

    def __call__(self, coords_A) -> dict:
        """[N_solute,3] Å -> 全部字段(标量 float / bool / list)。一次性同步在返回前。"""
        c = np.asarray(coords_A, dtype=np.float64)
        if c.shape != (self._q.size, 3):
            raise ValueError(f"coords 要 [{self._q.size},3], 得到 {c.shape}")
        if not np.isfinite(c).all():
            raise ValueError("坐标含 NaN/Inf")  # sasa 也会抓, 但在这里抓得更早更明确
        # PB 的归位与 margin 在 TripletSolver 里(C/R 与 L 各按自己的质心、各自的网格)。
        # MM/SA 平移不变, 但 SA 走 fp32, 远离原点的原坐标会丢精度 —— 也归位到原点
        pb = self._tri(c, self._q)
        c = recenter_com(c, self._masses, np.zeros(3))
        margin = pb["margin_A"]
        mm = mm_cross_prepared(c[None], self._cross)  # [1,N,3] 入口

        sa_ok, sa_err = True, ""
        try:
            # sasa(k 不够)按设计 raise 而不是给偏大的面积 —— 在线 catch 成 flag
            dg_sa, areas = delta_g_sa(c, self._radii, self._rec_local,
                                      self._lig_local, gamma=self._gamma,
                                      beta=self._beta, **self._sa_kw)
        except ValueError as e:
            sa_ok, sa_err = False, str(e)
            dg_sa = float("nan")
            areas = {"complex": np.nan, "receptor": np.nan, "ligand": np.nan}

        # mm 出口带批维 [B=1], PB 出口是 0 维标量; 统一摊平后取 float
        e_coul = float(np.asarray(mm["e_coul_rl"]).reshape(-1)[0])
        e_lj = float(np.asarray(mm["e_lj_rl"]).reshape(-1)[0])
        d_pb = float(pb["delta_g_pb"])
        out = {
            "e_coul_rl": e_coul,
            "e_lj_rl": e_lj,
            "g_pb_complex": float(pb["g_pb_complex"]),
            "g_pb_receptor": float(pb["g_pb_receptor"]),
            "g_pb_ligand": float(pb["g_pb_ligand"]),
            "delta_g_pb": d_pb,
            "sasa_complex": float(areas["complex"]),
            "sasa_receptor": float(areas["receptor"]),
            "sasa_ligand": float(areas["ligand"]),
            "delta_g_sa": float(dg_sa),
            "delta_g_mmpbsa": e_coul + e_lj + d_pb + float(dg_sa),
            "solver_iters": [int(v) for v in np.asarray(pb["iters"])],
            "converged": bool(np.asarray(pb["converged"])),
            "margin_A": margin,
            "margin_lig_A": pb["margin_lig_A"],
            "sa_ok": sa_ok,
        }
        if not sa_ok:
            out["sa_error"] = sa_err
        return out

    # reporter/测试要看的只读状态
    @property
    def triplet_solver(self) -> TripletSolver:
        return self._tri

    @property
    def ref_extent(self) -> float:
        """参考帧包围盒对角线(Å) —— reporter 首帧的单位哨兵用。"""
        return self._ref_extent


class PBSAReporter:
    """OpenMM Reporter 协议(OpenMM ≥ 7.7: `describeNextReport` 返回 **6 元组**,
    末位 `enforcePeriodicBox=False` —— 拿未 wrap 的坐标, 分子天然完整, 漂移由
    归位 + margin guard 吸收; 老版本 5 元组会把配体切成两半, 不支持)。

    每帧开销自累计, `close()` 打印 —— 这是 `scripts/online_overhead.py` 之外
    免费的日常观测口径。
    """

    _CSV_FIELDS = ("step", "time_ps", "e_coul_rl", "e_lj_rl", "g_pb_complex",
                   "g_pb_receptor", "g_pb_ligand", "delta_g_pb", "sasa_complex",
                   "sasa_receptor", "sasa_ligand", "delta_g_sa",
                   "delta_g_mmpbsa", "iter_c", "iter_r", "iter_l", "converged",
                   "sa_ok", "margin_A", "margin_lig_A")

    def __init__(self, analyzer: OnlineMMPBSA, interval_steps: int, solute_idx,
                 out_csv: str | None = None, on_violation: str = "flag",
                 margin_min: float = 0.0):
        if on_violation not in ("flag", "raise"):
            raise ValueError('on_violation 必须是 "flag"/"raise"')
        self.analyzer = analyzer
        self.interval_steps = int(interval_steps)
        self.solute_idx = np.asarray(solute_idx, dtype=int)
        self.on_violation = on_violation
        # margin_A ≥ 0 只保证没丢原子; margin_min 是 **Dirichlet 边界的物理
        # 余量**(padding 预算里那 12 Å, RESULTS §15.8 的 ~1.5κ⁻¹)—— margin=+1
        # 的帧数值上不丢原子, 边界已经贴脸。S4 传 12; 0 只查丢原子。
        self.margin_min = float(margin_min)
        self._fh = open(out_csv, "w", buffering=1) if out_csv else None
        self._csv = None
        self.n_reports = 0
        self.n_flagged = 0
        self.total_pbsa_s = 0.0

    def describeNextReport(self, simulation):
        # (steps, positions, velocities, forces, energy, enforcePeriodicBox)
        #
        # **steps 是「距下次报告还差几步」, 不是固定间隔。** OpenMM 每轮都重新问
        # 一遍全部 reporter, 取 nextSteps = min(steps), 步进那么多, 然后只触发
        # steps == nextSteps 的那些(`Simulation._simulate`)。恒返回 interval 的话,
        # 只要同时挂了间隔更短的 reporter —— 生产里必然有 DCD —— 本 reporter 就
        # **永远不触发**: 实测 DCD@500 + 本 reporter@25000 跑 150k 步, 0 帧。
        # 单独挂时反而正常(nextSteps 就是它自己), 所以这个 bug 只在生产配置下现形。
        steps = self.interval_steps - simulation.currentStep % self.interval_steps
        return (steps, True, False, False, False, False)

    def _violation(self, out: dict) -> str | None:
        m = out["margin_A"]
        if m < min(self.margin_min, 0.0):
            return (f"margin_A = {m:.2f} Å < 0: 溶质已触到网格边界, 越界原子被 "
                    "clip+权重置零悄悄丢掉(pb/charges.py), 本帧 delta_g_pb "
                    "偏小, 不可用")
        if m < self.margin_min:
            return (f"margin_A = {m:.2f} Å < margin_min {self.margin_min:.1f}: "
                    "没丢原子, 但 Dirichlet 边界已贴脸 —— padding 预算里的物理"
                    "边界量(RESULTS §15.8), 本帧 ΔG_PB 的边界误差超标")
        if not out["converged"]:
            return f"PB 未收敛: solver_iters = {out['solver_iters']}"
        if out["margin_lig_A"] < 0:
            # 配体紧盒只查丢原子: 它按设计贴近边界, margin_min 不适用(见 TripletSolver)
            return (f"margin_lig_A = {out['margin_lig_A']:.2f} Å < 0: 配体构象撑出"
                    "了紧盒, 越界原子被悄悄丢掉, 本帧 G_L 不可用 —— 加大 padding_lig")
        if not out["sa_ok"]:
            return f"SA 失败: {out.get('sa_error', '')}"
        return None

    def report(self, simulation, state):
        pos_A = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
            dtype=np.float64)
        coords = pos_A[self.solute_idx]

        # 单位哨兵**只在首帧查**: nm→Å 忘 ×10 是 10 倍误差, 而单位错不可能
        # 中途出现(代码路径固定), 首帧必暴露。反过来, 配体解离/跳 image 会让
        # 包围盒**真的**涨 3 倍 —— 那是 margin guard 该 flag 的物理事件, 不能
        # 让哨兵在 flag 模式下把它当单位错杀掉几小时的 MD。
        extent = float(np.linalg.norm(coords.max(0) - coords.min(0)))
        ratio = extent / self.analyzer.ref_extent
        if self.n_reports == 0 and not (1 / 3 < ratio < 3):
            raise ValueError(
                f"包围盒尺度是参考帧的 {ratio:.1f}× —— 像是 nm/Å 单位错(×10)"
                f"或 solute_idx 切错, 拒绝写入: extent={extent:.1f} Å")

        t0 = time.perf_counter()
        out = self.analyzer(coords)
        self.total_pbsa_s += time.perf_counter() - t0
        out["step"] = int(simulation.currentStep)
        out["time_ps"] = float(state.getTime().value_in_unit(unit.picosecond))

        bad = self._violation(out)
        if bad:
            if self.on_violation == "raise":
                raise RuntimeError(f"[step {out['step']}] {bad}")
            # flag: 落盘继续 —— 列里的 margin_A/converged/sa_ok 就是标记。
            # 但**不能只落盘**: 没给 out_csv 时那就等于把违规静默丢掉(实测 S6
            # 的第一次运行正是这个配置, 68 帧有没有被 flag 当时无从得知)。
            # 首次打印 + close() 汇总, 让 flag 模式至少留下一条可见的痕迹。
            self.n_flagged += 1
            if self.n_flagged == 1:
                print(f"[PBSAReporter] 首个被 flag 的帧 step {out['step']}: {bad}"
                      + ("" if self._fh is not None else
                         "  (未给 out_csv, 逐帧标记不会落盘)"), flush=True)
        if self._fh is not None:
            if self._csv is None:
                import csv
                self._csv = csv.writer(self._fh)
                self._csv.writerow(self._CSV_FIELDS)
            it = out["solver_iters"]
            self._csv.writerow([
                out["step"], out["time_ps"], out["e_coul_rl"], out["e_lj_rl"],
                out["g_pb_complex"], out["g_pb_receptor"], out["g_pb_ligand"],
                out["delta_g_pb"], out["sasa_complex"], out["sasa_receptor"],
                out["sasa_ligand"], out["delta_g_sa"], out["delta_g_mmpbsa"],
                it[0], it[1], it[2], out["converged"], out["sa_ok"],
                out["margin_A"], out["margin_lig_A"]])
            self._fh.flush()  # 跑几小时崩了不能丢已算的帧
        self.n_reports += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.n_reports:
            per = self.total_pbsa_s / self.n_reports * 1e3
            flagged = (f", **{self.n_flagged} 帧被 flag**" if self.n_flagged
                       else ", 无 flag")
            print(f"[PBSAReporter] {self.n_reports} 帧, PBSA 累计 "
                  f"{self.total_pbsa_s:.2f} s, 均值 {per:.1f} ms/帧{flagged}")

    def __del__(self):
        try:
            if getattr(self, "_fh", None) is not None:
                self._fh.close()
        except Exception:
            pass
