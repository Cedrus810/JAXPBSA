# JAXPBSA

GPU Poisson–Boltzmann / MM-PBSA binding-energy analysis in JAX. Two modes:

- **offline** — batch over OpenMM trajectories: `ΔG_PB = G_C − G_R − G_L` per frame,
- **online** — ΔG_MM/PBSA(t) computed frame-by-frame *while the MD runs*, on the same GPU.

**Status: research code.** APIs move; packaging/CI come later. What is stable is the
measurement discipline: every number in [`RESULTS.md`](./RESULTS.md) is measured, split
into *machine-invariant* and *machine-specific*, and reproducible from
`scripts/benchmark.py`.

## What it computes

```
ΔG_MM/PBSA  =  ΔE_MM  +  ΔG_PB  +  ΔG_SA          (−TΔS out of scope)
                 │         │         │
       receptor–ligand   linear PB   γ·SASA + β
       Coulomb + LJ      solver
```

Validated against Amber `MMPBSA.py` on the same frames. On **S4** (Src SH2 +
phosphotyrosyl peptide, 1835 atoms; 20 frames over 10 ns; current defaults C/R
h = 0.5, ligand h = 0.25):

| term | MMPBSA.py | jaxpbsa | Δ |
|---|---|---|---|
| ΔE_coul (R–L) | −826.83 | −826.86 | 3.5e-5 rel |
| ΔE_LJ (R–L) | −44.66 | −44.66 | 4.4e-5 rel |
| **ΔG_PB** | **780.52** | **784.85** | **+0.55%** |
| **ΔG_SA** | −5.79 | −5.78 | 2.6e-4 rel |
| **ΔG_MM/PBSA** | **−96.75** | **−92.45** | +4.30 |

On a second system (1YCR, MDM2–p53) the same defaults give ΔG_PB 284.25 vs 292.66
(−2.9%); the remaining few percent are not yet decomposed (RESULTS §18.8). The
electrostatics are the sharp end: two ~800 kcal/mol numbers cancel to a few percent,
and the whole error sits in the isolated-ligand solve (`δΔG_PB ≈ −δG_L`) — which is why
the ligand gets its own tight grid.

## Features

- **Linear PB**, matrix-free PCG on a regular grid; geometric multigrid used as a CG
  *preconditioner* (standalone MG diverges at the production 1:80 dielectric jump);
  the uniform-dielectric reference equation is solved **exactly by DST**.
- **fp32 arrays, fp64 reductions** — bit-identical ΔG_PB and iteration counts across GPUs.
- **Dielectric models** (`PBParams(surface=...)`): `binary` SES (default), `fraction`
  (level-set), `gaussian` (DelPhi), `gaussian_gap` (heterogeneous Gaussian PB, aligned to
  JCP 545 (2026) 114452), plus **nonlinear PB** (`nonlinear=True`, damped Newton with a
  convex-energy Armijo line search).
- **Asymmetric grids by default**: C/R share one h = 0.5 box, the ligand solves in its
  own tight h = 0.25 box.
- **Every frame recentred on its mass-weighted centroid** — rigid-body drift no longer
  sweeps the sub-grid phase (±9 kcal/mol effect, RESULTS §15.4).
- **Nonpolar term** via JAX Shrake–Rupley SASA.
- **Online mode**: grid sized from a pilot trajectory, per-frame centroid recentring,
  per-frame boundary-margin guard, OpenMM reporter writing CSV.

## Installation

Requires Python ≥ 3.10, JAX with a CUDA backend, and (for the online mode / system prep)
OpenMM and mdtraj:

```bash
pip install -e ".[openmm,traj,test]"
# JAX CUDA wheel per https://jax.readthedocs.io (e.g. pip install -U "jax[cuda]")
```

## Usage

**Offline** (system prep once via `scripts/prep_s4.py`, then everything downstream reads
the pinned artifact — never a force field):

```python
from jaxpbsa.openmm_io import load_canonical
from jaxpbsa.pb import PBParams, TripletSolver, make_frame_solver, make_grid

d   = load_canonical()                    # verifies sha256 on load
tri = TripletSolver(traj, masses, d["radii"],          # traj [T,N,3] Å (or one frame)
                    d["receptor_idx"], d["ligand_idx"])  # grids sized from it
tri(coords, q)                            # dict: ΔG_PB = G_C − G_R − G_L (+ margin_A)

# single-species building block
g  = make_grid(d["positions_A"][None], h=0.5, padding=20.0)
sv = make_frame_solver(g, d["radii"], PBParams(precond="auto"))
sv(coords, q, radii)                      # one frame, one species (u_prev= for warm start)
```

**Online** (ΔG(t) from a running simulation; design in
[`ONLINE_PLAN.md`](./ONLINE_PLAN.md)):

```python
import jaxpbsa
from jaxpbsa.online import OnlineMMPBSA, PBSAReporter

jaxpbsa.enable_compilation_cache()

az = OnlineMMPBSA(system, topology,
                  solute_idx,             # global indices into the solvated system
                  ligand_local_idx,       # local indices after the solute slice
                  ref_coords_A,           # warm-up frame (runs the self-check)
                  pilot_coords_A=pilot)   # [T,N_solute,3] Å: sizes the grid

sim.reporters.append(PBSAReporter(az, interval_steps=10000, solute_idx=solute_idx,
                                  out_csv="pbsa.csv"))
```

The online grid is sized from data, not a magic padding: per axis, half-extent =
max|x − COM| over the pilot frames + reach + 1.0·κ⁻¹. A pilot trajectory (or an explicit
`fluctuation_allowance=`) is required — a single frame under-sizes S4 by 3.79 Å
(RESULTS §16.9). The per-frame flag threshold is separate and physical (0.1·κ⁻¹, where
the measured ΔG_PB boundary error is still ≤ 0.073 kcal/mol, §18.17). When MD and PBSA
share one GPU: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`, or OpenMM's CUDA context
fails to start.

```bash
python scripts/benchmark.py --csv out.csv   # h × {fp32,fp64} × {jacobi,mg,auto}
python scripts/crl.py                       # ΔG_PB, C/R 0.5 + L 0.25 (current defaults)
python scripts/profile_stages.py 0.5 32     # per-stage timing
pytest -q
```

## Performance

| measurement | number | notes |
|---|---|---|
| single-species PB solve, end-to-end | 68.3 ms (2080 Ti) / 29.0 ms (5080) | S4, h = 0.5, fp32; bit-identical results |
| full ΔG_PB (C + R + L) | ~210 ms/frame (2080 Ti) | asymmetric defaults, trajectory frames |
| binary surface on RTX 5090 | 59.1 / 53.8 ms/frame | S4 / 1YCR (RESULTS §18.14) |
| online end-to-end incl. MD on one GPU | 128.7 ms/frame (5080) | §18.15 |
| vs Amber `MMPBSA.py` | 17.9× in the online/latency regime | parity in throughput only when Amber gets all 40 cores (§13) |

The same code produces **bit-identical** ΔG_PB (+885.405), iteration counts and true
residuals on a 2080 Ti and a 5080 (RESULTS §0.2). Do not quote a "26×" speedup — that
compares against `MMPBSA.py` on a single core, which is not how anyone runs it (§13).

## Layout

```
jaxpbsa/
  pb/          grid, charges, surface (binary/fraction/gaussian/gaussian_gap),
               operator, solver (PCG), multigrid, dst, energy (LPB + NLPB)
  mm/          receptor–ligand Coulomb + LJ cross terms
  sa/          nonpolar term — JAX Shrake–Rupley (zsasa is a test-only reference)
  openmm_io/   parameter extraction, radii, load_canonical
  benchmark/   stage-resolved timing helpers
  online.py    online entry: fixed grid + per-frame centroid recentring + reporter
scripts/
  prep_s4.py, prep_1ycr.py, prep_pdb25.py   system prep → canonical artifacts
  benchmark.py, profile_stages.py           benchmarking / per-stage timing
  crl.py, validate_mmpbsa.py                ΔG_PB and side-by-side vs MMPBSA.py
  online_overhead.py, fit_grid.py           online contention; pilot-based grid sizing
  run_s4_md.py, analyze_md.py, warm_start.py, gpu_bench.py, …
data/prepared/
  S4_complex.cif + S4_system.xml + S4_meta.json   pinned canonical artifact (sha256)
```

System preparation (`prep_*.py`) is a one-shot converter — broken PDB records, force
fields, hydrogen placement are its problem and nobody else's. The artifact is pinned
rather than re-derived because `Modeller.addHydrogens` is not deterministic across
OpenMM versions: two runs once differed by RMSD 0.78 Å, enough to move G_PB by
54 kcal/mol.

## Documentation

- [`RESULTS.md`](./RESULTS.md) — every measurement; machine-invariant vs machine-specific.
- [`CHANGELOG.md`](./CHANGELOG.md) — 13 silent correctness bugs, 8 methodology lessons,
  6 design predictions overturned by measurement.
- [`DESIGN.md`](./DESIGN.md) — design decisions and their rationale.
- [`ONLINE_PLAN.md`](./ONLINE_PLAN.md) — online mode design and decisions.
- [`docs/README-2026-09.md`](./docs/README-2026-09.md) — the previous README (snapshot).

**Ruled out by measurement** (don't re-try without new evidence, RESULTS §10): batching
(net loss at production grid size), warm start as a headline claim (1.07×), MG
coarse-sweep tuning, decomposing the morphological ball, and GB as a validation
reference for PB (GB *is* an approximation to PB — agreeing proves nothing).

## License

[MIT](./LICENSE) © 2026 Cedrus810.
