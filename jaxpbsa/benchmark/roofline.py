"""Stage-resolved timing for the PB pipeline (DESIGN.md §3.12 / RESULTS.md).

**为什么重写**: 上一版按固定 `12 × n³ × itemsize × iters` 估算内存流量再反推
"占峰值带宽的百分比", 那个数不能用来论证优化空间 ——

  * 12 次数组遍历是对 **Jacobi-PCG** 的估计, 不代表 MG 的多层操作;
  * 完全没算 DST 的 FFT、表面核的 gather、参考边界的 [S, atom_block] 中间量;
  * 默认 itemsize=8, 但默认精度已经是 fp32;
  * 真实缓存流量与编译后的融合结果有关, 源码层数不出来。

所以"算子融合还有 2–3×"这个说法没有依据, 已从文档撤回。要判断值不值得写
专用 kernel, 得先有**可复现的分阶段测量**, 而不是纸面流量。

计时的三个陷阱(JAX 官方 benchmark 指南):
  1. 异步派发 —— 必须 `block_until_ready` 后再停表;
  2. 编译时间 —— 第一次调用含 tracing+compile, 要单独记录并丢弃;
  3. 主机-设备传输 —— 输入要预先放到设备上, 否则量到的是 PCIe。
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp


def timeit(fn, *args, reps: int = 5, warmup: bool = True):
    """Returns (compile_seconds, steady_seconds_per_call).

    第一次调用的耗时含 tracing + XLA 编译, 单独返回; 之后 `reps` 次取均值。
    每次都做设备同步, 不依赖返回值被消费。
    """
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    if not warmup:
        return compile_s, compile_s
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return compile_s, (time.perf_counter() - t0) / reps


def device_arrays(*arrays, dtype=None):
    """把输入预先放上设备, 免得把 PCIe 传输计进 kernel 时间。"""
    return [jax.block_until_ready(jnp.asarray(a, dtype)) for a in arrays]


def stage_report(stages: dict[str, tuple[float, float]]) -> str:
    """stages: name -> (compile_s, steady_s)。按稳态耗时排序输出占比。"""
    total = sum(v[1] for v in stages.values())
    lines = [f"{'阶段':<20}{'编译':>10}{'稳态':>12}{'占比':>8}"]
    for name, (c, s) in sorted(stages.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"{name:<20}{c:9.2f}s{s * 1e3:11.2f}ms{100 * s / total:7.1f}%")
    lines.append(f"{'合计':<20}{'':>10}{total * 1e3:11.2f}ms")
    return "\n".join(lines)


def device_info() -> str:
    d = jax.devices()[0]
    return f"{getattr(d, 'device_kind', str(d))} ({jax.default_backend()})"
