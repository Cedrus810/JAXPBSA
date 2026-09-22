#!/usr/bin/env python
"""分阶段 profile PB 管线 (RESULTS.md §2)。

用法: python scripts/profile_stages.py [h] [precision]

编译、稳态、以及各阶段占比分别报告。所有输入预先放上设备; 每次计时都做设备同步。
"""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jaxpbsa
jaxpbsa.enable_compilation_cache()  # 编译 36.5 -> 5.9 s

h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
jaxpbsa.set_precision(int(sys.argv[2]) if len(sys.argv) > 2 else 32)

import jax, jax.numpy as jnp, numpy as np

from jaxpbsa.benchmark.roofline import device_arrays, device_info, stage_report, timeit
from jaxpbsa.constants import debye_kappa2
from jaxpbsa.openmm_io import load_canonical
from jaxpbsa.pb import PBParams, make_frame_solver, make_grid
from jaxpbsa.pb.charges import assign_density, interpolate, source_term
from jaxpbsa.pb.dst import solve_reference
from jaxpbsa.pb.multigrid import build_levels, make_preconditioner
from jaxpbsa.pb.operator import (apply_operator, coulomb_boundary_values,
                                 dh_boundary_values, interior_mask,
                                 jacobi_diagonal, shell_indices)
from jaxpbsa.pb.solver import pcg_solve
from jaxpbsa.pb.surface import ball_offsets, build_maps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 只读规范产物, 不碰力场(见 load_canonical)
_d = load_canonical(root=ROOT)
q_np, radii_np, xyz_np = _d["charge"], _d["radii"], _d["positions_A"]

g = make_grid(xyz_np[None], h, padding=20.0)
p = PBParams(swin=0.5, tol=1e-5, max_iter=8000, precond="mg")
dt = jaxpbsa.dtype()
coords, q, radii = device_arrays(xyz_np, q_np, radii_np, dtype=dt)
shell_xyz, shell_flat = shell_indices(g)
shell_xyz = jnp.asarray(shell_xyz, dt)
mask = interior_mask(g)
k2p = debye_kappa2(p.ionic_strength_M, p.eps_out, p.temperature_K)
off = dict(smooth_offsets=ball_offsets(p.swin, g.h),
           raster_offsets=ball_offsets(float(radii_np.max()) + g.h, g.h),
           probe_offsets=ball_offsets(p.probe_radius, g.h),
           ion_offsets=ball_offsets(p.ion_radius, g.h))

print(f"设备: {device_info()}   精度: {dt.__name__ if hasattr(dt,'__name__') else dt}")
print(f"网格: {g.shape} = {int(np.prod(g.shape)):,} 节点   h={g.h}   "
      f"边界点 {shell_xyz.shape[0]:,}   原子 {len(q_np):,}\n")

S = {}
maps_fn = jax.jit(lambda c, r: build_maps(c, r, g, p.eps_in, p.eps_out,
                                          p.probe_radius, p.ion_radius, k2p, **off))
S["surface (ε/κ̄² map)"] = timeit(maps_fn, coords, radii)
maps = maps_fn(coords, radii)

S["charges → b"] = timeit(jax.jit(lambda c, qq: source_term(assign_density(c, qq, g), g)),
                          coords, q)
b = source_term(assign_density(coords, q, g), g)

S["MG 建层"] = timeit(jax.jit(lambda m: build_levels(m["eps_x"], m["eps_y"], m["eps_z"],
                                                     m["kbar2"], g.h)), maps)
lv = build_levels(maps["eps_x"], maps["eps_y"], maps["eps_z"], maps["kbar2"], g.h)
pre = make_preconditioner(lv, nu1=p.mg_nu, nu2=p.mg_nu, coarse_sweeps=p.mg_coarse_sweeps)
A = lambda u: apply_operator(u, maps["eps_x"], maps["eps_y"], maps["eps_z"],
                             maps["kbar2"], g.h)
d = jacobi_diagonal(maps["eps_x"], maps["eps_y"], maps["eps_z"], maps["kbar2"], g.h)
# **溶剂求解要用生产路径的真实 Dirichlet 边界**, 不是零边界。
# 零边界会让 CG 从一个和生产完全不同的起点出发, 迭代数与耗时都不可比;
# 而且 S4 complex 的净电荷接近零, 边界贡献本来就小 —— 这会掩盖差异,
# 不能外推到带电的 receptor / ligand。
center = jnp.asarray(np.asarray(g.origin) + 0.5 * (np.asarray(g.shape) - 1) * g.h, dt)
a_ion = jnp.max(jnp.linalg.norm(coords - center, axis=1)) + p.ion_radius
kappa = float(np.sqrt(k2p)) if k2p > 0 else 0.0
ub = dh_boundary_values(shell_xyz, center, q.sum(), p.eps_out, kappa, a_ion)
u0 = jnp.zeros(g.shape, dt).reshape(-1).at[shell_flat].set(ub).reshape(g.shape)
S["溶剂求解 (MG-PCG)"] = timeit(
    jax.jit(lambda: pcg_solve(A, b, u0, d, mask, precond=pre, tol=p.tol,
                              max_iter=p.max_iter)))

S["参考边界 (库仑和)"] = timeit(
    jax.jit(lambda s_, c_, q_: coulomb_boundary_values(
        s_, c_, q_, p.eps_in, atom_block=p.boundary_atom_block)), shell_xyz, coords, q)
ubr = coulomb_boundary_values(shell_xyz, coords, q, p.eps_in,
                              atom_block=p.boundary_atom_block)
u0r = jnp.zeros(g.shape, dt).reshape(-1).at[shell_flat].set(ubr).reshape(g.shape)
ex = jnp.full((g.shape[0] - 1, g.shape[1], g.shape[2]), p.eps_in, dt)
ey = jnp.full((g.shape[0], g.shape[1] - 1, g.shape[2]), p.eps_in, dt)
ez = jnp.full((g.shape[0], g.shape[1], g.shape[2] - 1), p.eps_in, dt)
apply_ref = lambda u: apply_operator(u, ex, ey, ez, jnp.zeros(g.shape, dt), g.h)
S["参考求解 (DST)"] = timeit(
    jax.jit(lambda bb, uu: solve_reference(bb, uu, apply_ref, p.eps_in, g.h)), b, u0r)

u_s, _, _, _ = pcg_solve(A, b, u0, d, mask, precond=pre, tol=p.tol, max_iter=p.max_iter)
u_r = solve_reference(b, u0r, apply_ref, p.eps_in, g.h)
# **能量阶段要复现生产路径**: 溶剂势减参考势, 再用 fp64 归约 —— 不是单份势的插值求和。
from jaxpbsa import ACCUM_DTYPE
from jaxpbsa.constants import KT_TO_KCAL
S["能量 (插值+差分+fp64归约)"] = timeit(
    jax.jit(lambda us, ur, c, qq: 0.5 * jnp.sum(
        qq * (interpolate(us, c, g) - interpolate(ur, c, g)),
        dtype=ACCUM_DTYPE) * KT_TO_KCAL),
    u_s, u_r, coords, q)

print(stage_report(S))
c_tot, s_tot = timeit(make_frame_solver(g, radii_np, p).jitted, coords, q, radii)
print(f"\n端到端单帧: 编译 {c_tot:.2f}s, 稳态 {s_tot * 1e3:.2f} ms")
print(f"分阶段之和 {sum(v[1] for v in S.values()) * 1e3:.2f} ms "
      f"(差值 = 融合收益 + 未覆盖的碎片)")
