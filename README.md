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

On **S4** (Src SH2 domain + phosphotyrosyl peptide, PDB 1SPS chains C+F, 1835 atoms,
h = 0.5 Å, fp32):

| term | kcal/mol |
|---|---|
| ΔE_coul (R–L) | −894.80 |
| ΔE_LJ (R–L) | −52.41 |
| **ΔE_MM** | **−947.21** |
| **ΔG_PB** | **+885.41** |
| **ΔG_SA** | **−5.74** |
| **ΔG_MM/PBSA** | **−67.54** |

> **⚠ ΔG_PB is under review — a fix is measured but not yet the default.**
> Because `δG_C ≈ δG_R` cancels, the *entire* discretization error of ΔG_PB is the
> isolated-ligand solve, and that solve is far from converged at h = 0.5 Å.
> A per-species grid (C/R at h = 0.75, ligand in a tight box at h = 0.25) gives
> **ΔG_PB = 851.7 instead of 885.4** — closing the gap to Amber `pbsa` (842.9) from
> **5.0% to 1.0%** — and is **18% faster** (162 vs 197 ms/frame), because the current
> code spends its grid on C and R where the error cancels anyway.
> Measured on one S4 frame; see [RESULTS §15](./RESULTS.md). The table below is the
> current code's output, not a converged result.

The electrostatic cancellation is the sharpest check on the PB result: two numbers of
order 890 cancel to **−9.4 (1.1%)**. A 5% error in ΔG_PB would leave tens of kcal/mol
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

Full ΔG_PB (three species: complex, receptor, ligand) — **197.5 ms/frame** on a
2080 Ti. Truncating the MG hierarchy at 41×41×49 (`PBParams.mg_min_n=41`) gives
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

- **One grid shared by complex / receptor / ligand.** G_PB absolute values carry tens
  of kcal/mol of grid self-energy error; only a common origin/shape/spacing makes it
  cancel in C − R − L. Measured: refining h 0.75 → 0.5 moves G_C by +31.35 and G_R by
  **+31.36** — cancelling to 0.01 kcal/mol.
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

**Error budget** at h = 0.5 (Richardson-extrapolated): discretisation **~1.1 kcal/mol**,
dominated by the *ligand* — the complex's and receptor's errors cancel almost exactly,
so refinement should be driven by the ligand, not the complex. Do not estimate the
error bar on ΔG_PB from the relative error on the absolute values: that gives
28 kcal/mol instead of 1.1, a factor of **25**.

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
from jaxpbsa.pb import PBParams, make_frame_solver, make_grid

d  = load_canonical()
g  = make_grid(d["positions_A"][None], h=0.5, padding=20.0)   # shared by C/R/L
sv = make_frame_solver(g, d["radii"], PBParams(precond="auto"))

sv(coords, q, radii)                              # one frame, one species
sv.triplet(coords, q, rec_idx, lig_idx)           # ΔG_PB = G_C − G_R − G_L
sv.trajectory(traj, q, radii, warm=True,          # returns (per-frame energies,
              initial_state=state)                #          final_state)
```

Compilation is amortised over frames: ~8.5 s to compile, 76.8 ms/frame steady state,
**crossover at ~100 frames**, 1.1% overhead at 10,000 frames. The three species share a
single compilation.

```bash
python scripts/benchmark.py --csv out.csv    # h × {fp32,fp64} × {jacobi,mg,auto}
python scripts/crl.py 0.5 32                 # ΔG_PB
python scripts/profile_stages.py 0.5 32      # per-stage timing
pytest -q                                    # 40 tests
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
                  ref_coords_A,      # builds the grid + runs the warm-up self-check
                  h=0.5, padding=30.0)

sim.reporters.append(PBSAReporter(az, interval_steps=10000, solute_idx=solute_idx,
                                  out_csv="pbsa.csv", margin_min=12.0))
```

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
measure. Each frame also reports `margin_A` (distance left to the grid boundary; negative
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
zsasa, canonical artifact, cross-device validation.

**Not done** — solver memory-access efficiency (82–85% of runtime, and a larger share on
faster cards), asymmetric grids (measured faster *and* closer to Amber, not yet the
default), the online contention measurement (`scripts/online_overhead.py` is written, the
numbers in plan §21.5 are still extrapolated), and the same COM recentring on the offline
path (`ONLINE_PLAN.md` §7 — required before any online-vs-offline per-frame comparison).

Since the last revision of this section: SA in JAX (stage 2), external validation against
Amber `pbsa` (`RESULTS.md` §12), the 10 ns production trajectory, and the online entry
(`jaxpbsa/online.py`) all landed.

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
