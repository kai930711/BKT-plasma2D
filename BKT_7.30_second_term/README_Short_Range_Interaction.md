# Full pair interaction (logarithmic + short-range "cosh" term)

This extends the logarithmic-plasma-only Monte Carlo simulation to
optionally include the full physical pair interaction

```
V_{zeta,zeta'}(r) = -zeta*zeta' * K * ln(r/xi)
                     - T * ln[2 * cosh((E0/T) * exp(-r/xi_F))]
```

The two terms are handled by completely different (and separately
switchable) code paths:

| Term | Long/short range | Depends on charges | Summation | Code |
|---|---|---|---|---|
| `-zeta*zeta'*K*ln(r/xi)` (**U1**) | long-ranged | yes (`zeta_i * zeta_j`) | Ewald | `Ewald_Method_pbc.py` (unchanged) |
| `-T*ln[2 cosh(...)]` (**U2**) | short-ranged | no | minimum-image, no periodic images | `Short_Range_Interaction.py` (new) |

`U1` is exactly the pre-existing Ewald calculation and was **not modified**.
`U2` is new and is evaluated with a plain minimum-image sum since it decays
exponentially and does not need lattice summation.

## Dimensionless energies used by the sampler

```
beta*U = kappa*U1 + U2
```

For a single pair, define

```
x_ij = (E0/T) * exp(-r_ij / xi_F) = E0_over_T * exp(-r_ij / xi_F)

beta*V2(r) = -ln[2 cosh(x)]              (exact, physical)
V2_shifted(r) = -ln cosh(x)              (shifted: drop the -ln(2) constant)
```

Summed over all pairs `i < j`:

```
U2_exact   = sum_{i<j} beta*V2(r_ij)      = U2_shifted - [N(N-1)/2] * ln(2)
U2_shifted = sum_{i<j} V2_shifted(r_ij)
```

`-ln(2)` per pair is configuration-independent at fixed `N`, so it cancels
out of every `ΔU2` used by the Metropolis rule. The sampler therefore always
uses **`U2_shifted`**, and `U2_exact` is only needed when you want the
absolute energy matching the original equation (e.g. for reporting free
energies, not for driving the Markov chain).

## Numerical stability

`-ln(cosh(x))` is *never* evaluated directly (it overflows once `x` is a
few hundred). Instead:

```
ln cosh(x) = logaddexp(x, -x) - ln(2)
-ln cosh(x) = ln(2) - logaddexp(x, -x)
```

`numpy.logaddexp` is used, which is stable for arbitrarily large `|x|`.

## Units / conventions

Positions `R_all` are fractional coordinates in `[0, 1) x [0, 1)`, i.e. the
box length is `L = 1`, exactly as in the existing Ewald code
(`Ewald_potential.md`, `Setup.py`). Every length-like parameter you pass to
the new functions — `r`, `xi_F`, `L` — must be expressed in these same box
units (physical length divided by the physical box length `L_phys`):

```
xi_F_box = xi_F_physical / L_phys
```

`Short_Range_Interaction.py`'s functions accept an explicit `L` keyword
(default `1.0`) so the minimum-image convention generalizes to a
non-unit box if you ever need it; `Monte_Carlo.run_mc` itself keeps the
existing simulation on the unit-box convention (`L=1.0`) to stay consistent
with `Setup.py`'s hardcore-overlap check, which assumes `L=1`.

If your externally-known parameter is `xiF_over_xi` (i.e. `xi_F` expressed
in units of the logarithmic-term's own length scale `xi`), convert it
yourself before calling this code:

```
xi_F_box = xiF_over_xi * xi_box
```

This code makes **no assumption** that `xi` equals the hard-core radius
`a` — that equivalence is not baked in anywhere here. If it is a
convention you use elsewhere, compute `xi_F_box` from it explicitly at the
call site.

## New file: `Short_Range_Interaction.py`

- `minimum_image_displacement(dR, L=1.0)` / `minimum_image_distance(Ri, Rj, L=1.0)`
  — minimum-image displacement/distance, generalized to box length `L`.
- `energy2_pair(r, E0_over_T, xi_F)` — shifted pair energy `V2_shifted(r)`
  for one pair (or elementwise over an array of `r`). Numerically stable.
- `energy2(R_all, E0_over_T, xi_F, L=1.0)` — total `U2_shifted`, summed once
  over all `i < j` using minimum-image distances.
- `energy2_diff(R_all, i, R_i_old, R_i_new, E0_over_T, xi_F, L=1.0)` — `O(N)`
  change in `U2_shifted` from moving particle `i`; used by the Metropolis
  step instead of recomputing the full `O(N^2)` sum on every trial move.
- `energy2_exact_from_shifted(U2_shifted, N)` — recovers
  `U2_exact = U2_shifted - [N(N-1)/2] * ln(2)`.

## Modified file: `Monte_Carlo.py`

`run_mc1(...)` (the original logarithmic-only driver) is **unchanged** and
still works exactly as before — it's what `run_mc(..., include_log=True,
include_cosh=False, ...)` reproduces bit-for-bit (same RNG stream, same
accept/reject decisions).

A new driver `run_mc(...)` was added alongside it:

```python
run_mc(
    N, q, kappa, a, alpha, n_cut, m_cut,
    step_size, n_sweeps, sample_every,
    n_burnin=0, seed=0,
    include_log=True,      # include U1 (logarithmic Ewald term)
    include_cosh=False,    # include U2 (short-range cosh term)
    E0_over_T=0.0,         # E0/T, dimensionless
    xi_F=1.0,              # decay length of U2, in box units
    L=1.0,                 # box length (positions are fractional coords)
)
```

Acceptance rule:

```
ΔβU = kappa*ΔU1 + ΔU2
accept with probability min(1, exp(-ΔβU))
```

`ΔU1` is computed by re-running the (unmodified) full Ewald sum, exactly as
`run_mc1` already did. `ΔU2` is computed in `O(N)` via `energy2_diff`, not
by recomputing the full pairwise sum.

Returns:

```python
snapshots, U1_trace, U2_shifted_trace, U2_exact_trace, beta_U_total_trace, acceptance_rate = run_mc(...)
```

where `beta_U_total_trace[k] = kappa*U1_trace[k] + U2_shifted_trace[k]` is
the configuration-dependent quantity that actually drives the sampler, and
`U2_exact_trace[k] = U2_shifted_trace[k] - [N(N-1)/2]*ln(2)`.

Modes:

| `include_log` | `include_cosh` | Behavior |
|---|---|---|
| `True` | `False` | logarithmic term only (= `run_mc1`) |
| `False` | `True` | short-range term only |
| `True` | `True` | full interaction |

## Modified file: `Size_Effect_Data.py`

`main()` now calls `run_mc` instead of `run_mc1` and accepts
`include_cosh=False, E0_over_T=0.0, xi_F=1.0` (all defaulted so the output
`fig2b_summary.csv` is byte-for-byte identical to before unless you opt in).

The original Fig. 2(b) Coulomb observable

```
U_over_Ne2 = U1/N - 0.5*log(a)
```

is **always** computed from `U1` alone — it is never redefined to include
`U2` and keeps its original column name. When `include_cosh=True`, four new
columns are appended instead: `U2_shifted_mean`, `U2_exact_mean`,
`beta_U_total_mean`, `beta_U_total_std`.

## Tests: `Test_Short_Range_Interaction.py`

Covers:

- minimum-image distance across a box boundary (two particles near opposite
  edges resolve to a short distance, not the naive long one), including a
  non-unit `L`;
- `energy2_diff` agrees with `energy2(after) - energy2(before)` over 200
  random single-particle trial moves on an evolving configuration;
- numerical stability of `energy2_pair` for very large `x` (`E0_over_T=1e6`);
- `energy2_exact_from_shifted` bookkeeping;
- `beta_U_total_trace == kappa*U1_trace + U2_shifted_trace` and
  `U2_exact_trace == U2_shifted_trace - [N(N-1)/2]*ln(2)` throughout a run;
- `run_mc(include_log=True, include_cosh=False)` reproduces `run_mc1`
  exactly (same acceptance rate, same `U1` trace, same snapshots);
- `E0_over_T=0` gives `U2_shifted == 0` everywhere and produces an
  identical trajectory/acceptance to the log-only run (i.e. it truly does
  not affect Metropolis decisions, not just "small effect");
- `include_log=False, include_cosh=True` runs and gives `U1 == 0`,
  `beta_U_total == U2_shifted`.

Run:

```bash
python3 Test_Short_Range_Interaction.py
```

or with pytest:

```bash
pytest Test_Short_Range_Interaction.py -v
```

## Quick-test command

```bash
python3 -c "
import numpy as np
from Monte_Carlo import run_mc

N = 4
q = np.array([1.0, -1.0, 1.0, -1.0])
a = 0.05

snapshots, U1, U2s, U2e, betaU, acc = run_mc(
    N=N, q=q, kappa=2.0, a=a, alpha=np.sqrt(np.pi), n_cut=1, m_cut=3,
    step_size=0.1, n_sweeps=10, sample_every=2, n_burnin=2, seed=0,
    include_log=True, include_cosh=True, E0_over_T=1.0, xi_F=0.2,
)
print('acceptance_rate', acc)
print('beta_U_total', betaU)
"
```

## Summary of exactly which original files were changed

- **`Monte_Carlo.py`** — modified. Added an import of the three functions
  needed from the new module, and added a new function `run_mc(...)`
  after the existing, untouched `run_mc1(...)`. No line inside
  `initialize_positions` or `run_mc1` was changed.
- **`Size_Effect_Data.py`** — modified. Swapped the `run_mc1` import/call
  for `run_mc` with all-new parameters defaulted to reproduce the original
  behavior (`include_cosh=False`), and added optional extra CSV columns
  for the full interaction. The original `U_over_Ne2` column/definition is
  untouched.
- **`Ewald_Method_pbc.py`** — **not modified** (per requirement 1).
- **`Setup.py`, `Regeneration_of_Size_Effect.py`, `Visualization.py`** —
  **not modified**.

## New files added

- **`Short_Range_Interaction.py`** — the new short-range interaction module.
- **`Test_Short_Range_Interaction.py`** — the test script described above.
- **`README_Short_Range_Interaction.md`** — this file.
