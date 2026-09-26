# JAXPBSA

GPU-batched Poisson–Boltzmann binding-energy analysis for OpenMM trajectories, in JAX.

**Status: early development.** This repo is a working testbed — APIs move, and the
packaging/CI story comes later. What is stable is the measurement discipline: every
performance claim in [`RESULTS.md`](./RESULTS.md) is measured, split into
*machine-invariant* and *machine-specific*, and reproducible from
`scripts/benchmark.py`.

[中文版 / Chinese](./README.zh.md)

---

## What it computes

```
ΔG_MM/PBSA  =  ΔE_MM  +  ΔG_PB  +  ΔG_SA          (−TΔS out of scope for phase 1)
                 │         │         │
       receptor–ligand   linear PB   nonpolar
       Coulomb + LJ      solver      γ·SASA + β
```

On **S4** (Src SH2 domain + phosphotyrosyl peptide, PDB 1SPS chains C+F, 1835 atoms),
20 frames spread over a 10 ns trajectory, default grids (C/R h = 0.5, ligand h = 0.25),
fp32, against Amber `MMPBSA.py` on the same frames:

| term | MMPBSA.py | jaxpbsa | Δ |
|---|---|---|---|
| ΔE_coul (R–L) | −826.83 | −826.86 | 3.5e-5 rel |
| ΔE_LJ (R–L) | −44.66 | −44.66 | 4.4e-5 rel |
| **ΔG_PB** | **780.52** | **784.85** | **+4.33 (0.55%)** |
| **ΔG_SA** | −5.79 | −5.78 | 2.6e-4 rel |
| **ΔG_MM/PBSA** | **−96.75** | **−92.45** | +4.30 |

> On a second system (1YCR, MDM2–p53) the same default gives ΔG_PB **284.25 vs 292.66
> (−2.9%)**; the source of the remaining few percent is not yet measured (RESULTS §18.8).
> The earlier S4 figure of 0.26% came from C/R at h = 0.75, which carries a real
> −7 to −11 kcal/mol discretisation bias that happened to offset the rest.

Until 2026-09-23 the default was one h = 0.5 grid shared by all three species, and
ΔG_PB came out **5% high** (885.4 vs 842.9 on the canonical frame). The cause: the
C/R discretisation errors cancel, so the *whole* error of ΔG_PB sits in the
isolated-ligand solve, which carries a ~31 kcal/mol bias at h = 0.5 (S4; 24.6 on 1YCR).
The default now gives the ligand its own tight h = 0.25 box and keeps C/R at h = 0.5.
C/R at h = 0.75 takes ~37% less time but is biased low by 6.8 (S4) / 11.1 (1YCR) kcal/mol even
with the sub-grid phase randomised per frame ([RESULTS §17–18](./RESULTS.md)).

The electrostatic cancellation is the sharpest check on the PB result: two numbers of
order 800 cancel to a few percent of either. A 5% error in ΔG_PB would leave tens of kcal/mol
of residue and change the answer entirely.

---

## Performance

RTX 2080 Ti, S4, h = 0.5 Å, fp32, true-residual tol = 1e-5. Single-species PB solve:

| step | ms/frame | cumulative |
|---|---|---|
| baseline (fp64, cubic grid, Jacobi-PCG, PCG reference) | 1800 | 1.0× |
| + DST direct reference solve | 840 | 2.1× |
| + true-residual criterion, tol 1e-5 | 769 | 2.3× |
| + multigrid preconditioner | 339 | 5.3× |
| + fp32 arrays (fp64 reductions) | 149 | 12.1× |
| + rectangular grid | ~100 | 18.0× |
| + blocked reference boundary | 96 | 18.8× |
| + static-slice morphology | 76.8 | 23.4× |
| + unrolled MG smoother (host-constant trip count) | **68.3** | **26.4×** |

Full ΔG_PB (three species: complex, receptor, ligand) — **~210 ms/frame** on a
2080 Ti with the default asymmetric grids on trajectory frames (old shared h = 0.5
grid: 197.5 ms on the canonical frame, 285 ms on a trajectory-wide grid). Truncating the MG hierarchy at 41×41×49 (`PBParams.mg_min_n=41`) gives
**177.0 ms/frame**; not the default yet, measured on S4 only (RESULTS §14.4).
The 5080 figure (88.4 ms/frame) predates the unrolled smoother and has not been
re-measured on that card.

**Cross-device validation.** Two GPUs, same canonical structure, identical results:

| | RTX 2080 Ti | RTX 5080 |
|---|---|---|
| ΔG_PB | +885.405 | **+885.405** |
| iterations (C/R/L) | 11 / 12 / 10 | **11 / 12 / 10** |
| true residual | 8.79e-06 / 4.76e-06 / 7.81e-06 | **identical** |
| end-to-end, single frame | 76.8 ms | 29.0 ms |

**Against Amber `MMPBSA.py`** — same S4 frames, throughput regime (frames are free,
the whole machine is yours). jaxpbsa on one 2080 Ti does **320 ms/frame**; `MMPBSA.py`
scales from 5806 ms/frame serial down to **292 ms/frame** — but only when given the
*entire* dual-socket server, **all 40 physical cores** (the machine has 80 threads;
hyperthreading does not scale and is excluded). That is the only point where Amber is
ahead, and barely: **parity is reached only at full-machine 40c/80t** — at 32 cores it
is already behind (340 ms), and in the online/latency regime (frames arriving one at a
time while the CPU is busy running the MD that produced them) jaxpbsa wins outright,
325 ms vs 5806 ms, **17.9×**. Do not quote a "26×" speedup — that compares against
`MMPBSA.py` on a single core, which is not how anyone would run it. Details and the
full table: [`RESULTS.md`](./RESULTS.md) §13.

---

## How it works

**Linear PB**, discretised on a regular grid, solved matrix-free:

- **C and R share one grid; the ligand gets its own.** G_PB absolute values carry tens
  of kcal/mol of grid self-energy error; only a common grid makes it cancel in C − R
  (but not fully at h = 0.75: 20 frames × random phase put C − R 7–11 below h = 0.5,
  so C/R stays at 0.5). The isolated
  ligand has nothing to cancel against, so it is solved in a tight h = 0.25 box
  (`TripletSolver`, `h_lig=None` restores the old single shared grid).
- **Every frame is recentred on its mass-weighted centroid** — C/R on the complex
  centroid, L on its own. Rigid-body drift otherwise sweeps the sub-grid phase and moves
  ΔG_PB by up to ±9 kcal/mol frame to frame (RESULTS §15.4, §17.2).
- **Multigrid as a CG *preconditioner*, not a solver.** Re-discretised MG *diverges* at
  the production ε jump of 1:80 (convergence factor 1.8 per V-cycle); geometric
  interpolation cannot represent a solution whose normal derivative jumps by 80× across
  the dielectric interface. CG minimises, so it cannot diverge — and the h-independence
  that multigrid buys survives (11 iterations at h = 0.5 vs 193 for Jacobi).
  Enabled automatically above 1.5 M grid nodes; below that MG is a *net loss*.
- **The reference equation is solved exactly by DST**, not iteratively. It is the harder
  of the two equations — no dielectric contrast, no κ̄² mass term — and accounted for
  **69%** of all iterations before this change.
- **fp32 arrays, fp64 reductions.** MD coordinates are fp32 anyway. Measured shift in
  G_PB: a constant **+0.035 kcal/mol**, against a discretisation error of **40**. But
  reductions must stay fp64: the R–L cross energy sums ~10⁶ signed pairs.

**Error budget**: the ligand's discretisation bias (31 / 24.6 kcal/mol at h = 0.5 on
S4 / 1YCR, ~1–3 left at h = 0.25) and the C/R bias at h = 0.75 (−6.8 / −11.1, real
discretisation error, not placement: RESULTS §18.8) — hence C/R at h = 0.5. Padding is converged on both systems (40/20 vs 20/8: 0.004). Do not estimate the
error bar on ΔG_PB from the relative error on the absolute values: that overstates it
by an order of magnitude.

---

## Layout

```
jaxpbsa/
  pb/        grid, charges, surface, operator, solver (PCG), multigrid, dst, energy
  mm/        receptor–ligand Coulomb + LJ cross terms
  sa/        nonpolar term — JAX Shrake-Rupley (zsasa is a test-only reference)
  openmm_io/ parameter extraction, radii, load_canonical
  benchmark/ stage-resolved timing helpers
  online.py  online entry: fixed grid + per-frame COM recentring, OpenMM reporter
scripts/
  prep_s4.py        converter: PDB → canonical artifact
  benchmark.py      cross-device benchmark matrix
  profile_stages.py per-stage timing
  crl.py            ΔG_PB = G_C − G_R − G_L
  warm_start.py     cold vs warm, three-way comparison
  online_overhead.py  MD + online PBSA on one GPU: contention measurement
  fit_grid.py       size the fixed online grid from a pilot trajectory
data/prepared/
  S4_complex.cif    topology + coordinates   ← canonical starting point
  S4_system.xml     serialised openmm.System
  S4_meta.json      sha256 of both, atom indices, provenance
```

### The canonical artifact

`scripts/prep_s4.py` is a **converter**: broken PDB records, the PTR phosphotyrosine,
hydrogen placement, force fields, residue naming — all of that is its problem and
nobody else's. It runs once and emits a self-contained OpenMM artifact. Everything
downstream reads only that and never touches a force field:

```python
from jaxpbsa.openmm_io import load_canonical
d = load_canonical()          # verifies both sha256 on load
```

Pinning the artifact rather than trusting the recipe is deliberate. The pipeline *is*
now bit-reproducible, but only for this OpenMM version: `Modeller.addHydrogens` places
hydrogens at random positions and fixes them with an internal minimisation. Before this
was pinned, two runs of the same script differed by **RMSD 0.78 Å** — enough to move
G_PB by 54 kcal/mol.

---

## Usage

```python
from jaxpbsa.openmm_io import load_canonical
from jaxpbsa.pb import PBParams, TripletSolver, make_frame_solver, make_grid

d   = load_canonical()
tri = TripletSolver(traj, masses, d["radii"],        # traj [T,N,3] Å (or one frame):
                    d["receptor_idx"], d["ligand_idx"])  # grids sized from it, recentred
tri(coords, q)          # ΔG_PB = G_C − G_R − G_L (+ margin_A), any rigid translation

# single-species building blocks
g  = make_grid(d["positions_A"][None], h=0.5, padding=20.0)
sv = make_frame_solver(g, d["radii"], PBParams(precond="auto"))
sv(coords, q, radii)                              # one frame, one species
sv.trajectory(traj, q, radii, warm=True,          # returns (per-frame energies,
              initial_state=state)                #          final_state)
```

Compilation is amortised over frames: ~8.5 s to compile, 76.8 ms/frame steady state,
**crossover at ~100 frames**, 1.1% overhead at 10,000 frames. The three species share a
single compilation.

```bash
python scripts/benchmark.py --csv out.csv    # h × {fp32,fp64} × {jacobi,mg,auto}
python scripts/crl.py                        # ΔG_PB (C/R 0.5, L 0.25; `0.5 32 shared` = old)
python scripts/profile_stages.py 0.5 32      # per-stage timing
pytest -q                                    # 42 tests
```

### Online: ΔG_MM/PBSA(t) from a running simulation

Offline builds the grid from the whole trajectory. Online has only frame 0, so the grid
is **fixed up front** and every frame is recentred into it — translating coordinates does
not recompile, moving `grid.origin` does (2.59 s, measured). See
[`ONLINE_PLAN.md`](./ONLINE_PLAN.md).

```python
import jaxpbsa
from jaxpbsa.online import OnlineMMPBSA, PBSAReporter

jaxpbsa.enable_compilation_cache()

az = OnlineMMPBSA(system, topology,
                  solute_idx,        # global indices into the solvated system
                  ligand_local_idx,  # local indices, [0, N_solute) after the slice
                  ref_coords_A,      # warm-up frame (runs the self-check)
                  pilot_coords_A=pilot)  # [T,N_solute,3] Å trial trajectory: sizes the grid

sim.reporters.append(PBSAReporter(az, interval_steps=10000, solute_idx=solute_idx,
                                  out_csv="pbsa.csv"))   # margin_min defaults to 1.5·κ⁻¹
```

The grid is **sized from data, not a magic padding**: per axis, half-extent =
max|x − COM| over the pilot frames + reach (r_max + probe + ion + swin) + 1.5·κ⁻¹, rounded up
to the next APBS dime (ligand box: + 1.0·κ⁻¹). Conformational fluctuation does not
generalise, so either a pilot trajectory or an explicit `fluctuation_allowance=` (Å) is
**required** — a single frame under-sizes S4 by 3.79 Å (`RESULTS.md` §16.9), and the old
fixed `padding=30` cost 26.9% extra time for identical ΔG (§16.8). On S4, a 1 ns pilot is
enough. The chosen sizes are in `az.sizing`.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # required when MD shares the GPU:
                                             # JAX otherwise grabs 75% of VRAM and
                                             # OpenMM's CUDA context fails to start
python scripts/online_overhead.py --steps 50000
```

Recentring uses the **mass-weighted centroid**, not the bounding-box midpoint: the midpoint
is set by six extremal atoms, so one side chain swinging 1 Å shifts the solute by half a
grid spacing, and the placement sensitivity of ΔG_PB is **8.39 kcal/mol peak-to-peak**
(`RESULTS.md` §15.4) — the same order as what an online sampling monitor is meant to
measure. Each frame also reports `margin_A` (distance left to the boundary of the tighter
of the two grids; negative
means atoms are being silently dropped) and `sa_ok`.

---

## Reading the numbers

`RESULTS.md` splits every measurement into two classes, and **the split is by mechanism,
not by unit**:

- **Machine-invariant** — iteration counts, energies, true residuals. Also *ratios driven
  by iteration count*: warm start is 1.07× on both GPUs tested.
- **Machine-specific** — wall-clock, throughput, memory. Also *ratios driven by fixed
  latency*: multigrid on a coarse grid is 0.87× on a 2080 Ti but 0.45× on a 5080,
  because its coarse levels are launch-latency bound and a faster card wastes more.

Algorithmic conclusions rest only on the first class. Report efficiency as a fraction of
device peak bandwidth, not seconds, when comparing across machines.

---

## Known state

**Done** — MM cross terms, PB solver (PCG + multigrid preconditioner + DST reference),
surface/dielectric construction, C/R/L triplet with reference-field reuse
(`u_ref_C = u_ref_R + u_ref_L`, potentials not energies — the reference energy contains
R–L cross terms), trajectory interface with warm start and cross-chunk state, SA via
JAX Shrake–Rupley, canonical artifact, cross-device validation.

**Not done** — solver memory-access efficiency (82–85% of runtime, and a larger share on
faster cards), re-measuring the online overhead table (§16) on the new default grids,
and checking the asymmetric-grid result on a second system.

Since the last revision of this section: SA in JAX (stage 2), external validation against
Amber `pbsa` (`RESULTS.md` §12), the 10 ns production trajectory, the online entry
(`jaxpbsa/online.py`), and — 2026-09-23 — asymmetric grids as the default plus centroid
recentring on the offline path, both through `TripletSolver` (`RESULTS.md` §17).

**Deferred with a measurement behind it** — BinaryCIF. The canonical CIF quantises
coordinates at 1e-4 Å; measured effect on ΔG_PB is **1e-4 kcal/mol**, four orders of
magnitude under the discretisation error. BinaryCIF would not help anyway: its standard
coordinate encoding is fixed-point with the same quantisation, so it compresses the
representation rather than improving precision. Revisit if systems reach 10⁵–10⁶ atoms;
for trajectories the answer is DCD/XTC, not BCIF. See `RESULTS.md` §9.6.

**Ruled out by measurement**, with numbers in `RESULTS.md` §10: batching (a net loss on
both GPUs at production grid size), warm start as a headline claim (1.07×), multigrid
coarse-sweep tuning (no effect), decomposing the morphological ball, and GB as a
validation reference for PB — GB *is* an approximation to PB, so agreeing with it proves
nothing.

See [`CHANGELOG.md`](./CHANGELOG.md) for the 13 correctness bugs found (all silent — no
error, just a plausible wrong number), 8 measurement-methodology lessons, and 6 design
predictions that measurement overturned.

---

## License

Copyright © 2026 Cedrus810. This project is licensed under the
[GNU Affero General Public License v3.0](./LICENSE) (**AGPL-3.0-only**).

The license is applied as a temporary protective measure; the copyright holder
reserves the right to relicense future versions.
