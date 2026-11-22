# BKT-plasma2D

Monte Carlo simulation of a two-dimensional two-component Coulomb gas (charges ±1, hard disks of diameter σ)
with an additional short-range interaction mediated by Majorana modes. The goal is the
Berezinskii–Kosterlitz–Thouless (dielectric → conducting) transition and the liquid–gas behaviour as functions of
density ρ\*, temperature T\* and coupling E0\*.

## What is on `main`

| Path | Contents |
|---|---|
| `11.22 Final Version/` | Current simulation code (Julia 1.12.6). Copied unchanged from `11.7 Code Final Version` on the `David` branch (commit 375289c); the file hashes equal the `source_sha256` values recorded in the 11.8 scan metadata. |
| `lit/` | `references.bib` (BibTeX for the papers behind the model, methods and analysis: Kosterlitz–Thouless 1973, Nelson–Kosterlitz 1977, Perram–de Leeuw 1981, Caillol–Levesque 1986, Orkoulas–Panagiotopoulos 1994 and 1996, Henderson 1975, Golub–Welsch 1969, Spitzner et al. 2018); PDF of Orkoulas & Panagiotopoulos, J. Chem. Phys. **104**, 7205 (1996); notes on the Majorana Monte Carlo model. |

Analysis and earlier code live on other branches:
- `kai`: `9.6_Summary`, `9.21_Density Correlation` (S_NN, S_QQ, 1/ε), `11.8_BKT_Transition` (BKT criterion on the 11.8 scan).
- `David`: development history from `5.27_Ewald` to `11.7 Code Final Version`, including `11.3 Free Energy Reproduction`.

## Model

The sampled weight is exp(−βU) with

    βU = κ U1 + U2,   κ = 1/T*

- **U1**: Ewald sum of −Σ<sub>i<j</sub> q<sub>i</sub> q<sub>j</sub> ln r<sub>ij</sub> in the unit box (self, real-space and
  reciprocal terms), plus the polarization term (π/2)|P|² with P = Σ<sub>i</sub> q<sub>i</sub> R<sub>i</sub> evaluated
  from coordinates wrapped into [0, 1). Ewald parameters: α = 4, n_cut = 1, m_cut = 5.
- **U2**: Σ<sub>i<j</sub> −ln cosh[(E0/k<sub>B</sub>T) e<sup>−r<sub>ij</sub>/ξ<sub>F</sub></sup>] with ξ<sub>F</sub> = 5σ, summed
  over periodic images (minimum image exact, further images from a cubic-spline table with total error ≤ 5×10⁻⁵).
  The constant −ln 2 per pair is dropped. U2 is switched off when E0\* = 0.
- **Hard core**: r<sub>ij</sub> ≥ σ.

Reduced units: L = 1, ρ\* = Nσ²/L², a = σ/L = √(ρ\*/N), T\* = k<sub>B</sub>T/K, E0\* = E0/K.

Moves per sweep: N single-particle displacements (uniform, ±0.05 L per coordinate) and 0.02 N rigid cluster
translations (Orkoulas & Panagiotopoulos 1994; bonds r < 1.5σ, step ±0.5 L, rejected on contact).

## Drivers

| Driver | Purpose | Run length |
|---|---|---|
| `Output_Coordinate_Dielectric.jl` | Scan of (ρ\*, T\*, E0\*) states. Writes `series.f64` (per sample: U1, U2, βU, then \|ρ_m\|² and \|Q_m\|² for the 10 shells m² = 1, 2, 4, 5, 8, 9, 10, 13, 16, 17), `coordinates.f64` (100 frames per run), `charges.f64`, `states.csv`, `progress.csv`, `metadata.toml` (schema 2). | E0\* = 0: 5,000 burn-in + 10,000 sweeps, sample every 10. E0\* > 0: 20,000 + 400,000, every 200. |
| `Output_FreeEnergy.jl`, `Free_Energy.jl` | Free energy at fixed T\*, E0\* for a list of ρ\* by λ-integration from the hard-disk fluid to the full interaction on 10 Gauss–Legendre nodes. Used with N = 50 (larger N exceeds the 3 h job limit). | 50,000 or 1,000,000 burn-in by node, 1,000,000 sweeps, sample every 100. |

`Setup.jl`, `Plasma_Interaction.jl`, `Monte_Carlo.jl` and `one_over_epsilon0.jl` are shared by both drivers.

## Observables

Stored values are raw sums (no 1/N), averaged over the directions of each shell, k = 2π|m|/L:

    S_NN(k) = < |Σ_j exp(2πi m·R_j)|² >        S_QQ(k) = < |Σ_j q_j exp(2πi m·R_j)|² >
    1/ε(k)  = 1 − κ S_QQ(k) / (2π m²)

The BKT transition is where 1/ε(k_min) = 4T\* (universal Nelson–Kosterlitz jump).
The text in `metadata.toml` defines S with a 1/N; the numbers in `series.f64` do not include it.

## Running

Install the packages once (from `11.22 Final Version/`):

    julia --project=. -e 'using Pkg; Pkg.instantiate()'

Local test of the scan driver:

    julia --project=. Output_Coordinate_Dielectric.jl --write-jobs --N 50 --rho 0.001,0.01 --T 0.05,0.1 --E0 0,0.25 --replicates 1
    julia --project=. Output_Coordinate_Dielectric.jl --job 0 --output-root Data

Cluster (HTCondor):

1. Copy `params_Dielectric.toml` (or `params_FreeEnergy.toml`) to `Cluster_Tasks/<task>/params.toml` in the project
   root (no spaces in `<task>`) and edit the grid. Axis syntax: a list `[0.1, 0.2]`, a single number,
   `"start:step:stop"`, or `"log:start:stop:points_per_decade"`.
2. From `11.22 Final Version/`: `julia --project=. Make_Jobs.jl ../Cluster_Tasks/<task>/params.toml`.
   This writes `<driver>_jobs.csv` and an `upload/` folder with everything the cluster needs (runs are packed into jobs of about 3 h, estimated).
3. Upload the contents of `upload/` to a dated folder inside `BKT-plasma2D/` on the cluster, next to
   `julia-1.12.6-linux-x86_64.tar.gz`.
4. `condor_submit Dielectric.sub` (or `FreeEnergy.sub`). Progress and merge commands are at the top of the `.sub` file.

The step-by-step notes inside `params_*.toml` (in Chinese) still refer to the folder by its old name,
`11.7 Code Final Version`.
