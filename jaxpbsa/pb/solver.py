"""Jacobi-PCG for the interior Dirichlet problem (v1 reference solver).

操作在整个 grid 数组上进行, 用 interior mask 把线性系统限制在内节点:
边界值固定为 u0 的 shell 值, 所有内积/更新都在 mask 子空间, 等价于只解
内部未知量。几何多重网格是 v1 主路径 (DESIGN.md §3.7), 本模块同时是它的
正确性对照; MG 下轮接入, 接口同为 (apply_a, b, u0, diag, mask)。
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def pcg_solve(
    apply_a: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    u0: jnp.ndarray,
    diag: jnp.ndarray,
    mask: jnp.ndarray,  # bool, 内节点 True
    precond=None,  # r -> z; None = Jacobi(diag). 必须对称正定, 否则 CG 失效
    tol: float = 1e-7,
    max_iter: int = 2000,
    atol: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (u, iters, rel_residual, converged). u0 的 shell 值即 Dirichlet 边界.

    **收敛判据是真残差, 而且它被*执行*而不只是*报告*。**

    ‖b − Au‖₂ ≤ tol·‖b·mask‖₂ + atol

    三件事必须写在一起, 少一件就会静默出错:

    1. **判据用真残差, 不用 PCG 免费拿到的 sqrt(rᵀM⁻¹r)。** 后者是 M⁻¹-范数,
       换预处理器(Jacobi → MG)甚至只缩放预处理器都会改变停止时机 —— 于是
       "相同 tol" 在两个求解器下不是相同精度, 任何加速比都可能只是"精度变松了"。

    2. **循环内的 r 是递推的, fp32 下会漂移**, 所以每轮外层重算 b − Au。
       递推残差达标而真残差没达标时, **带着剩余迭代预算重启**, 而不是
       "重算一下、打印出来、照样返回"。

    3. **零右端项要显式处理。** 全零电荷时 b_norm = 0, 纯相对判据会变成 0/0 = NaN,
       NaN > tol 是 False, 于是"收敛"并返回 u0。绝对项 `atol` 堵这个洞。
       (这说的是**源项全零**, 不是普通的净电荷为零体系 —— 后者 b 并不为零。)

    非收敛时返回 `converged = False`, 由调用方决定是报错还是降级; benchmark
    必须拒绝这类结果, 不能计进加速比。
    """
    b_int = b * mask
    # 收敛判据的分母, 固定为**源项**在内节点上的 2-范数 ‖b·mask‖₂。
    # 不用初始残差 ‖r₀‖: 那样 warm start 会被惩罚(起点越好, 要求降得越多),
    # 而用固定的源项范数, warm start 才能正确表现为"更少迭代达到同一精度"。
    b_norm = jnp.sqrt((b_int * b_int).sum())
    thresh = tol * b_norm + atol  # 绝对+相对, 对 b ≡ 0 也良定义
    inv_diag = jnp.where(mask, 1.0 / jnp.where(diag > 0, diag, 1.0), 0.0)
    apply_m = precond if precond is not None else (lambda r: r * inv_diag)
    dt = b.dtype

    def dot(x, y):
        return (x * y * mask).sum()

    def true_res(u):
        r = (b - apply_a(u)) * mask
        return jnp.sqrt(dot(r, r))

    def inner(u, budget):
        """从 u 出发跑 CG, 最多 budget 轮; 内层用递推残差, 外层负责真残差。"""
        r = (b - apply_a(u)) * mask
        z = apply_m(r)
        rz = dot(r, z)

        def cond2(st):
            _, _, _, _, _, k, nr = st
            return (k < budget) & (nr > thresh)

        def body2(st):
            u, r, z, p, rz, k, _ = st
            ap = apply_a(p) * mask
            pap = dot(p, ap)
            alpha = rz / jnp.where(pap == 0, jnp.ones_like(pap), pap)
            u = u + alpha * p * mask
            r = r - alpha * ap
            z = apply_m(r)
            rz_new = dot(r, z)
            beta = rz_new / jnp.where(rz == 0, jnp.ones_like(rz), rz)
            p = (z + beta * p) * mask
            return u, r, z, p, rz_new, k + 1, jnp.sqrt(dot(r, r))

        st = (u, r, z, z, rz, jnp.asarray(0, jnp.int32),
              jnp.sqrt(dot(r, r)).astype(dt))
        u, _, _, _, _, k, _ = jax.lax.while_loop(cond2, body2, st)
        return u, k

    def outer_cond(st):
        _, used, nr, prev = st
        # 三个退出条件缺一不可, 否则外层可能不前进:
        #   预算耗尽 / 已达标 / **无进展或非有限**(NaN 时 nr > thresh 为 False, 自然退出;
        #   停滞时 nr >= prev 为真, 也退出) —— 保证 while_loop 一定终止。
        progressed = nr < prev
        return (used < max_iter) & (nr > thresh) & progressed

    def outer_body(st):
        u, used, nr, _ = st
        # inner 内部重新算 r、z、p、rz —— 重启必须重置 Krylov 状态, 不能沿用旧的
        u_new, k = inner(u, max_iter - used)
        # 重算真残差: 递推残差达标不代表真残差达标(fp32 漂移)
        return u_new, used + jnp.maximum(k, 1), true_res(u_new).astype(dt), nr

    nr0 = true_res(u0).astype(dt)
    u, iters, nr, _ = jax.lax.while_loop(
        outer_cond, outer_body,
        (u0, jnp.asarray(0, jnp.int32), nr0, jnp.asarray(jnp.inf, dt)))
    relres = nr / jnp.where(b_norm > 0, b_norm, jnp.ones_like(b_norm))
    # 非有限一律判为失败: NaN <= thresh 是 False, 这里写显式一点
    return u, iters, relres, jnp.isfinite(nr) & (nr <= thresh)
