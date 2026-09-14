#!/usr/bin/env python
"""cold vs warm start 的三路对照 (RESULTS.md §10)。

    python scripts/warm_start.py [h] [帧间步长Å] [帧数]
    python scripts/warm_start.py 0.5 0.05 8          # 默认

三条路径：

  A  lax.scan + warm   carry = 上一帧的内部势
  B  lax.scan + cold   **每帧都从零势出发**（不含任何来自其他帧或调用方的信息）
  C  Python 循环 + warm  反复调同一个单帧 JIT，势场留在设备端手动传递

**C 是用来分离「scan 本身的开销」的对照组**：若 A ≈ C，说明 scan 没有额外损失。
先前报过「scan 比单帧慢 7.6×」，那是测量 bug —— 每次调用重新定义 step 导致重复
tracing，热运行里混着编译时间。本脚本把编译与热运行分开计时。

轨迹用随机游走模拟相邻 MD 帧（真实轨迹见 M0，尚未生成）。
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jaxpbsa

h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
amp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
T = int(sys.argv[3]) if len(sys.argv) > 3 else 8
jaxpbsa.set_precision(32)

import jax, jax.numpy as jnp, numpy as np

from jaxpbsa.benchmark.roofline import device_info
from jaxpbsa.openmm_io import load_canonical
from jaxpbsa.pb import PBParams, make_frame_solver, make_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = load_canonical(root=ROOT)
q, radii, xyz = d["charge"], d["radii"], d["positions_A"]
rng = np.random.default_rng(0)
traj = xyz[None] + np.cumsum(rng.normal(0, amp, (T,) + xyz.shape), axis=0)

g = make_grid(traj, h, padding=20.0)
p = PBParams(swin=0.5, tol=1e-5, max_iter=8000, precond="auto")
sv = make_frame_solver(g, radii, p)
dt = jaxpbsa.dtype()
c, qq, rr = (jax.block_until_ready(jnp.asarray(a, dt)) for a in (traj, q, radii))

print(f"设备 {device_info()}   网格 {g.shape}   T={T}   帧间步长 {amp} Å")
print(f"起点 {d['meta']['canonical_structure']} sha256 {d['meta']['canonical_sha256'][:16]}…\n")


def timed(fn):
    """编译与热运行分开 —— 混在一起是先前把 scan 误判为慢 7.6× 的原因。"""
    t0 = time.perf_counter(); o = fn(); jax.block_until_ready(o); comp = time.perf_counter() - t0
    t0 = time.perf_counter(); o = fn(); jax.block_until_ready(o); hot = time.perf_counter() - t0
    return comp, hot, o


cw, hw, ow = timed(lambda: sv.trajectory(c, qq, rr, warm=True))
cc, hc, oc = timed(lambda: sv.trajectory(c, qq, rr, warm=False))

svp = make_frame_solver(g, radii, p, return_potential=True)
fn1 = svp.jitted


def pyloop():
    st = jnp.zeros(g.shape, dt)
    out = []
    for i in range(T):
        o = fn1(c[i], qq, rr, st)
        st = o["u_solvent"]          # 留在设备端, 不回 host
        out.append(o["g_pb"])
    return out


cp, hp, _ = timed(pyloop)

itw = np.asarray(ow[0]["iters_solvent"]); itc = np.asarray(oc[0]["iters_solvent"])
print(f"{'路径':<24}{'编译':>9}{'热运行':>12}{'ms/帧':>10}{'平均迭代':>10}")
for nm, comp, hot, it in (("A scan + warm", cw, hw, itw.mean()),
                          ("B scan + cold", cc, hc, itc.mean()),
                          ("C python循环 + warm", cp, hp, float("nan"))):
    print(f"{nm:<24}{comp:8.2f}s{hot * 1e3:11.1f}ms{hot / T * 1e3:9.2f}{it:10.2f}")

gw, gc = np.asarray(ow[0]["g_pb"]), np.asarray(oc[0]["g_pb"])
print(f"\nwarm 收益: 迭代 {itc.mean():.2f} -> {itw.mean():.2f} "
      f"({100 * (itw.mean() / itc.mean() - 1):+.0f}%)   时间 {hc / hw:.2f}x")
print(f"scan 开销: A vs C = {hw / hp:.3f}x  (接近 1 说明 scan 无额外损失)")
print(f"能量一致性 cold vs warm: 最大差 {np.abs(gw - gc).max():.2e} kcal/mol")
print(f"收敛: warm {bool(np.all(np.asarray(ow[0]['converged'])))}  "
      f"cold {bool(np.all(np.asarray(oc[0]['converged'])))}")
