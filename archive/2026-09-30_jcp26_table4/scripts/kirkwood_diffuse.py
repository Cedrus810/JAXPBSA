"""外部基准: 漫射界面 Kirkwood 球(Ijaodoro et al., J. Comput. Phys. 545 (2026) 114452, §3.1)。
S(r) = 1 (r<2), 0 (r>5), 中间五次 C² 多项式(唯一: 1 − (10t³ − 15t⁴ + 6t⁵), t=(r−2)/3);
ε = S·1 + (1−S)·80; LPB, I = 0.15 M, κ̄² = ε_s κ² (1−S); 单位电荷在原点; 盒 [−a, a]³, 点电荷 DH 边界。
参考 LPB ΔE = −68.884489814969 kcal/mol(论文 1D 高阶解)。T、单位常数论文没给, 我们用 298.15 K 与库常数。
"""
import sys, numpy as np, jax.numpy as jnp
from jaxpbsa.constants import KT_TO_KCAL, debye_kappa2
from jaxpbsa.pb.grid import GridSpec
from jaxpbsa.pb.charges import assign_density, interpolate, source_term
from jaxpbsa.pb.operator import (apply_operator, jacobi_diagonal, interior_mask, shell_indices,
                                 coulomb_boundary_values, dh_boundary_values)
from jaxpbsa.pb.dst import solve_reference
from jaxpbsa.pb.multigrid import build_levels, make_preconditioner
from jaxpbsa.pb.solver import pcg_solve
REF = -68.884489814969
REF_NPB = -73.415356786147  # 同文献 Table 1/2 的 NPB 参考
ES, EM, I = 80.0, 1.0, 0.15


def S_of(r):
    t = np.clip((r - 2.0) / 3.0, 0, 1)
    return 1 - (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)


def run(h, a=10.0, x0=(0.0, 0.0, 0.0), face="mid", nonlinear=False):
    n = int(round(2 * a / h)) + 1
    g = GridSpec(origin=(-a, -a, -a), shape=(n, n, n), h=h)
    ax = -a + np.arange(n) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    x0 = np.asarray(x0, float)
    eps_at = lambda dx, dy, dz: EM * S_of(np.sqrt((X + dx - x0[0]) ** 2 + (Y + dy - x0[1]) ** 2 + (Z + dz - x0[2]) ** 2)) \
        + ES * (1 - S_of(np.sqrt((X + dx - x0[0]) ** 2 + (Y + dy - x0[1]) ** 2 + (Z + dz - x0[2]) ** 2)))
    if face == "mid":
        ex, ey, ez = eps_at(h / 2, 0, 0)[:-1], eps_at(0, h / 2, 0)[:, :-1], eps_at(0, 0, h / 2)[:, :, :-1]
    else:
        e = eps_at(0, 0, 0); hm = lambda p, q: 2 * p * q / (p + q)
        ex, ey, ez = hm(e[:-1], e[1:]), hm(e[:, :-1], e[:, 1:]), hm(e[:, :, :-1], e[:, :, 1:])
    k2 = debye_kappa2(I, ES, 298.15)
    kb = ES * k2 * (1 - S_of(np.sqrt((X - x0[0]) ** 2 + (Y - x0[1]) ** 2 + (Z - x0[2]) ** 2)))
    ex, ey, ez, kb = map(jnp.asarray, (ex, ey, ez, kb))
    c, q = jnp.asarray(x0[None]), jnp.asarray([1.0])
    b = source_term(assign_density(c, q, g), g)
    sh, fl = shell_indices(g)
    ub = dh_boundary_values(sh, c[0], 1.0, ES, float(np.sqrt(k2)), 0.0)
    u0 = jnp.zeros(g.shape).reshape(-1).at[fl].set(ub).reshape(g.shape)
    pre = make_preconditioner(build_levels(ex, ey, ez, kb, h))
    us, it, rr, ok = pcg_solve(lambda u: apply_operator(u, ex, ey, ez, kb, h), b, u0,
                               jacobi_diagonal(ex, ey, ez, kb, h), interior_mask(g), precond=pre, tol=1e-9, max_iter=10000)
    assert bool(ok)
    g_ion = 0.0
    if nonlinear:
        from jaxpbsa.pb.energy import npb_newton, npb_ion_energy
        us, n_newt, rr_n, ok_n = npb_newton(ex, ey, ez, kb, b, us, interior_mask(g), h, tol=1e-6,
                                            inner_tol=1e-3, max_newton=40, max_iter=10000, use_mg=True)
        assert bool(ok_n), (h, int(n_newt), float(rr_n))
        g_ion = float(npb_ion_energy(us, kb, interior_mask(g), h))
    one = [jnp.ones_like(ex), jnp.ones_like(ey), jnp.ones_like(ez)]; k0 = jnp.zeros(g.shape)
    u0r = jnp.zeros(g.shape).reshape(-1).at[fl].set(coulomb_boundary_values(sh, c, q, EM)).reshape(g.shape)
    ur = solve_reference(b, u0r, lambda u: apply_operator(u, *one, k0, h), EM, h)
    return 0.5 * float(interpolate(us - ur, c, g)[0]) * KT_TO_KCAL + g_ion


if __name__ == "__main__" and "--npb" in sys.argv:
    print(f"NPB 参考 {REF_NPB}", flush=True)
    for h in (1.0, 0.8, 0.5, 0.4, 0.25, 0.2):
        v = run(h, 20.0, nonlinear=True)
        vo = [run(h, 20.0, np.random.default_rng(s).uniform(0, h, 3), nonlinear=True) for s in range(2)]
        print(f"NPB a=20 h={h:<5} 电荷在格点: {v:10.4f}  误差 {100*(v/REF_NPB-1):+7.3f}%   偏离格点: "
              f"{', '.join(f'{x:.4f}' for x in vo)}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    print(f"参考 {REF}", flush=True)
    for a in (10.0, 20.0):
        for h in (1.0, 0.8, 0.5, 0.4, 0.25, 0.2, 0.1):
            if a == 20.0 and h < 0.2:
                continue
            v = run(h, a)
            vo = [run(h, a, np.random.default_rng(s).uniform(0, h, 3)) for s in range(2)] if h >= 0.2 else []
            extra = f"   偏离格点相位: {', '.join(f'{x:.4f}' for x in vo)}" if vo else ""
            print(f"a={a:<4} h={h:<5} 电荷在格点: {v:10.4f}  误差 {100*(v/REF-1):+7.3f}%{extra}", flush=True)
