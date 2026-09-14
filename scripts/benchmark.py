#!/usr/bin/env python
"""JAXPBSA 跨设备性能基准。

    python scripts/benchmark.py                    # 默认矩阵
    python scripts/benchmark.py --quick            # 只跑 h=0.5
    python scripts/benchmark.py --csv out.csv
    python scripts/benchmark.py --pdb my.pdb --ligand-chain B --receptor-chain A

报告分两类，**这是本脚本存在的理由**：

  换卡不变   迭代数、真残差、G_PB —— 纯线性代数/离散化，任何 GPU 上都一样。
             算法结论（MG 值不值、fp32 够不够）只能建立在这一类上。
  本机专属   秒、ms/帧、吞吐 —— 换卡必变，只在同一台机器内部比较才有意义。

已经踩过并在此规避的测量陷阱：

  1. 每个配置跑在**独立子进程**里。同进程顺序跑多个配置时，编译产物和显存
     单调累积，会让后面的配置变慢甚至假 OOM —— 曾经因此得出完全错误的
     batch 标度结论。
  2. 关闭 `XLA_PYTHON_CLIENT_PREALLOCATE`。默认预分配 75% 显存，两个进程
     并存时第二个会伪 OOM。
  3. 编译时间与稳态时间**分开报告**。第一次调用含 tracing + XLA 编译。
  4. 每次计时后 `block_until_ready`。JAX 是异步派发的。
  5. 输入预先放上设备，不把 host→device 传输计进 kernel 时间。
  6. 收敛判据是**真残差** ‖b−Au‖/‖b‖，不是 PCG 免费拿到的 sqrt(rᵀM⁻¹r)。
     后者是 M⁻¹-范数，换预处理器就换了判据，"相同 tol" 不是相同精度。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SELF = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(SELF))


# ---------------------------------------------------------------- worker ----
def worker(cfg: dict) -> dict:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import time

    import jaxpbsa

    jaxpbsa.set_precision(cfg["bits"])
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxpbsa.openmm_io import load_canonical
    from jaxpbsa.pb import PBParams, make_frame_solver, make_grid

    # 只读规范产物(cif + 序列化 System), **不碰力场** —— 见 load_canonical 的说明。
    d = load_canonical(root=cfg.get("root"))
    q, radii, xyz = d["charge"], d["radii"], d["positions_A"]

    grid = make_grid(xyz[None], cfg["h"], padding=cfg["padding"], cubic=cfg["cubic"])
    params = PBParams(
        eps_in=cfg["eps_in"], eps_out=cfg["eps_out"],
        ionic_strength_M=cfg["ionic_strength"], swin=cfg["swin"],
        tol=cfg["tol"], max_iter=cfg["max_iter"], precond=cfg["precond"],
        mg_nu=cfg["mg_nu"], boundary_atom_block=cfg["atom_block"],
    )
    fn = make_frame_solver(grid, radii, params).jitted
    dt = jaxpbsa.dtype()
    c, qq, rr = (jax.block_until_ready(jnp.asarray(a, dt)) for a in (xyz, q, radii))

    B = cfg["batch"]
    if B > 1:
        fn = jax.jit(jax.vmap(fn, in_axes=(0, None, None)))
        c = jnp.broadcast_to(c, (B,) + c.shape)

    t0 = time.perf_counter()
    out = fn(c, qq, rr)
    jax.block_until_ready(out["g_pb"])
    compile_s = time.perf_counter() - t0

    reps = cfg["reps"]
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(c, qq, rr)
    jax.block_until_ready(out["g_pb"])
    total = (time.perf_counter() - t0) / reps

    dev = jax.devices()[0]
    stats = dev.memory_stats() or {}
    g_pb = np.atleast_1d(np.asarray(out["g_pb"]))
    iters = np.atleast_1d(np.asarray(out["iters_solvent"]))
    relres = np.atleast_1d(np.asarray(out["relres_solvent"]))
    conv = np.atleast_1d(np.asarray(out["converged"]))
    # **不收敛或出现非有限值的结果一律作废**, 不能只打印出来还计进加速比 ——
    # 那等于把"精度变松了"当成"计算变快了"。
    if not bool(conv.all()):
        return {"error": f"未收敛 (真残差 {float(relres.max()):.2e} > tol)"}
    if not np.isfinite(g_pb).all() or not np.isfinite(relres).all():
        return {"error": "结果非有限 (NaN/Inf)"}
    return {
        "structure": d["meta"]["canonical_structure"],
        "sha": d["meta"]["canonical_sha256"],
        "device": getattr(dev, "device_kind", str(dev)),
        "backend": jax.default_backend(),
        "shape": "x".join(map(str, grid.shape)),
        "nodes": int(np.prod(grid.shape)),
        "n_atoms": len(q),
        # --- 换卡不变 ---
        "iters": int(iters.max()),
        "relres": float(relres.max()),
        "g_pb": float(g_pb[0]),
        # --- 本机专属 ---
        "compile_s": compile_s,
        "total_ms": total * 1e3,
        "per_frame_ms": total * 1e3 / B,
        "frames_per_s": B / total,
        "peak_gib": stats.get("peak_bytes_in_use", 0) / 2**30,
    }


# ---------------------------------------------------------------- driver ----
def run_cfg(cfg: dict, timeout: int) -> dict | None:
    """每个配置一个子进程 —— 见模块文档陷阱 1/2。"""
    p = subprocess.run(
        [sys.executable, SELF, "--worker", json.dumps(cfg)],
        capture_output=True, text=True, timeout=timeout,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    err = p.stderr
    if "RESOURCE_EXHAUSTED" in err or "OUT_OF_MEMORY" in err:
        return {"error": "OOM"}
    return {"error": (err.strip().splitlines() or ["failed"])[-1][:80]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT,
                    help="含 data/prepared/ 的项目根目录(规范产物在那里)")
    ap.add_argument("--h", nargs="+", type=float, default=[1.0, 0.75, 0.5])
    ap.add_argument("--batch", nargs="+", type=int, default=[1])
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--padding", type=float, default=20.0)
    ap.add_argument("--eps-in", type=float, default=1.0)
    ap.add_argument("--eps-out", type=float, default=78.5)
    ap.add_argument("--ionic-strength", type=float, default=0.15)
    ap.add_argument("--swin", type=float, default=0.5)
    ap.add_argument("--mg-nu", type=int, default=2)
    ap.add_argument("--atom-block", type=int, default=128)
    ap.add_argument("--max-iter", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--cubic", action="store_true", help="立方网格(对照); 默认矩形")
    ap.add_argument("--quick", action="store_true", help="只跑 h=0.5, fp32")
    ap.add_argument("--csv")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        print("RESULT " + json.dumps(worker(json.loads(a.worker))))
        return

    base = dict(root=os.path.abspath(a.root),
                padding=a.padding, eps_in=a.eps_in, eps_out=a.eps_out,
                ionic_strength=a.ionic_strength, swin=a.swin, tol=a.tol,
                max_iter=a.max_iter, mg_nu=a.mg_nu, atom_block=a.atom_block,
                reps=a.reps, cubic=a.cubic)

    hs = [0.5] if a.quick else a.h
    # "auto" 也进矩阵: 它按节点数在 jacobi/mg 之间选(阈值 MG_NODE_THRESHOLD),
    # 跑一遍才能验证阈值在**这台机器**上是否仍然选对 —— 交叉点按节点数在两卡上
    # 一致, 但两侧的幅度依赖设备, 所以换卡值得复核。
    combos = [(32, "auto")] if a.quick else [(64, "jacobi"), (64, "mg"), (64, "auto"),
                                             (32, "jacobi"), (32, "mg"), (32, "auto")]
    rows, dev = [], None
    for h in hs:
        for B in a.batch:
            for bits, precond in combos:
                cfg = {**base, "h": h, "batch": B, "bits": bits, "precond": precond}
                r = run_cfg(cfg, a.timeout)
                r = r or {"error": "no result"}
                rows.append({"h": h, "batch": B, "bits": bits, "precond": precond, **r})
                dev = dev or r.get("device")
                tag = f"h={h} B={B} fp{bits} {precond}"
                if "error" in r:
                    print(f"  {tag:<28} {r['error']}")
                else:
                    print(f"  {tag:<28} {r['shape']:>13}  {r['iters']:4d} it  "
                          f"relres {r['relres']:.1e}  {r['per_frame_ms']:8.1f} ms/帧  "
                          f"G_PB {r['g_pb']:11.4f}  峰值 {r['peak_gib']:.2f} GiB")

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\n没有成功的配置。")
        return

    # auto 是否选对: 对每个 (h, bits) 看 auto 的耗时是否等于 jacobi/mg 中较快的那个
    checks = {}
    for r in ok:
        checks.setdefault((r["h"], r["bits"], r["batch"]), {})[r["precond"]] = r
    lines = []
    for key, v in sorted(checks.items()):
        if not {"auto", "jacobi", "mg"} <= set(v):
            continue
        best = min(("jacobi", "mg"), key=lambda k: v[k]["per_frame_ms"])
        ratio = v[best]["per_frame_ms"] / v["auto"]["per_frame_ms"]
        lines.append(f"  h={key[0]} fp{key[1]}: auto {v['auto']['per_frame_ms']:7.1f} ms "
                     f"| 较快的是 {best} {v[best]['per_frame_ms']:7.1f} ms "
                     f"| {'✅ 选对' if ratio > 0.97 else '❌ 选错'} "
                     f"(jacobi {v['jacobi']['per_frame_ms']:.1f} / mg {v['mg']['per_frame_ms']:.1f})")
    if lines:
        from jaxpbsa.pb.energy import MG_NODE_THRESHOLD
        print(f"\n=== precond=\"auto\" 的判断（阈值 {MG_NODE_THRESHOLD:,} 节点）===")
        print("\n".join(lines))

    print(f"\n设备: {ok[0]['device']} ({ok[0]['backend']})   "
          f"体系: {ok[0]['n_atoms']} 原子   tol={a.tol} (真残差)")
    print(f"起点: {ok[0]['structure']} sha256 {ok[0]['sha'][:16]}…  (规范产物, 已校验)")
    def tag(r):
        return "h={} B={} fp{} {}".format(r["h"], r["batch"], r["bits"], r["precond"])

    print("\n=== 换卡不变（算法结论只能用这些） ===")
    print(f"{'配置':<26}{'网格':>14}{'迭代':>7}{'真残差':>11}{'G_PB':>13}")
    for r in ok:
        print(f"{tag(r):<26}{r['shape']:>14}{r['iters']:7d}"
              f"{r['relres']:11.1e}{r['g_pb']:13.4f}")
    print("\n=== 本机专属（换卡必变） ===")
    print(f"{'配置':<26}{'编译':>9}{'ms/帧':>11}{'帧/秒':>10}{'峰值显存':>11}{'相对基准':>10}")
    # 基准优先取 fp64+jacobi+B=1(完整矩阵时的参照物); 若该配置没跑(如 --quick),
    # 退回同 h 下 B=1 的第一个配置, 至少让 batch 标度可读, 而不是整列都是 "-"。
    base_t = {r["h"]: r["per_frame_ms"] for r in ok
              if r["bits"] == 64 and r["precond"] == "jacobi" and r["batch"] == 1}
    for r in ok:
        if r["h"] not in base_t and r["batch"] == 1:
            base_t[r["h"]] = r["per_frame_ms"]
    for r in ok:
        b = base_t.get(r["h"])
        rel = f"{b / r['per_frame_ms']:9.2f}x" if b else "        -"
        print(f"{tag(r):<26}{r['compile_s']:8.1f}s{r['per_frame_ms']:11.1f}"
              f"{r['frames_per_s']:10.2f}{r['peak_gib']:10.2f}G{rel}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {a.csv}")
    print("\n分阶段耗时: python scripts/profile_stages.py [h] [32|64]")


if __name__ == "__main__":
    main()
