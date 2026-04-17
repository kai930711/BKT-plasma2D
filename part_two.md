# Part Two Progress

## 2026-04-17: Step 1 - Inverse Dielectric Formula

Task from `# Revised Next Steps`:

> Figure out the exact formula for the inverse dielectric function `1/epsilon_0` used by Orkoulas & Panagiotopoulos, including its normalization in terms of `T*`, density, box size, charges, and dipole fluctuations.

Sources checked:

- Local reference: `lit/7205_1_online.pdf`.
- Cross-check: Aupic and Urbic, "A structural study of a two-dimensional electrolyte by Monte Carlo simulations", Eq. (4), which states that the low-`k` averaging follows Orkoulas & Panagiotopoulos: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4288541/>.
- Cross-check: Aupic and Urbic, "Thermodynamics and structure of a two-dimensional electrolyte by integral equation theory", Eq. (20), which gives the same estimator: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4032415/>.

### Formula

For a neutral two-dimensional Coulomb gas in a square periodic box of side length `L`, define the Fourier component of the microscopic charge density as

```math
q_{\mathbf{k}}
= \sum_{j=1}^{N} z_j \exp(-i \mathbf{k}\cdot\mathbf{r}_j),
```

where `z_j = +1` for positive charges and `z_j = -1` for negative charges. The allowed wave vectors in a square periodic box are

```math
\mathbf{k}
= \frac{2\pi}{L}(n_x,n_y),
\qquad n_x,n_y \in \mathbb{Z}.
```

If `q_k` is defined with dimensionless charge signs `z_j`, the dimensional form of the inverse dielectric estimator is

```math
\epsilon_0^{-1}
= \lim_{\mathbf{k}\to 0}
\left[
1
- \frac{2\pi q^2}{D k_B T k^2 L^2}
\left\langle q_{\mathbf{k}} q_{-\mathbf{k}} \right\rangle
\right].
```

In the reduced variables used for the standard two-dimensional Coulomb gas,

```math
T^* = \frac{k_B T D}{q^2},
\qquad
L^* = \frac{L}{\sigma},
\qquad
\mathbf{r}_j^* = \frac{\mathbf{r}_j}{\sigma},
\qquad
\mathbf{k}^* = \sigma \mathbf{k},
```

the same estimator becomes

```math
\epsilon_0^{-1}
= \lim_{\mathbf{k}^*\to 0}
\left[
1
- \frac{2\pi}{(k^*)^2 T^* (L^*)^2}
\left\langle q_{\mathbf{k}^*} q_{-\mathbf{k}^*} \right\rangle
\right],
```

with

```math
q_{\mathbf{k}^*}
= \sum_{j=1}^{N} z_j
\exp(-i \mathbf{k}^*\cdot\mathbf{r}_j^*).
```

For a neutral system, the small-`k` limit is related to the total dipole moment

```math
\mathbf{M}^*
= \sum_{j=1}^{N} z_j \mathbf{r}_j^*.
```

Expanding `q_k` at small `k` gives

```math
q_{\mathbf{k}^*}
\approx
-i \mathbf{k}^*\cdot\mathbf{M}^*.
```

After angular averaging in two dimensions,

```math
\left\langle
q_{\mathbf{k}^*}q_{-\mathbf{k}^*}
\right\rangle
\approx
\frac{(k^*)^2}{2}
\left\langle |\mathbf{M}^*|^2 \right\rangle.
```

This gives the equivalent thermodynamic-limit dipole-fluctuation form

```math
\epsilon_0^{-1}
=
1
- \frac{\pi}{T^*(L^*)^2}
\left\langle |\mathbf{M}^*|^2 \right\rangle.
```

### Implementation Notes

1. The finite-size estimator should use the wave-vector formula, not a naive wrapped-coordinate dipole formula, because polarization and particle wrapping are subtle under periodic boundary conditions.

2. Orkoulas & Panagiotopoulos evaluate the `k -> 0` limit by averaging over the lowest allowed nonzero wave vectors in the periodic cell. The later implementation step should choose the same low-`k` shell convention and document it explicitly.

3. The estimator itself does not directly contain density except through `N`, `L`, and the sampled configurations. For fixed reduced density,

```math
\rho^*
= \rho \sigma^2
= \frac{N\sigma^2}{L^2}
= \frac{N}{(L^*)^2}.
```

4. The KT crossing criterion used later must be expressed in the same variable convention as the scan. In the paper's reduced-temperature convention, the universal line is

```math
\epsilon_0^{-1} = 4T^*.
```

Translating this into this project's `kappa = K/T` convention is the next task.

### Status

Step 1 is documented. The next task is to translate `T*` into this project's variables and write the crossing criterion in terms of `kappa = K/T`.

## 2026-04-17: Step 2 - Translate `T*` To `kappa = K/T`

Task from `# Revised Next Steps`:

> Translate the paper's reduced variables into this project's variables, especially the relation between `T*` and `kappa = K/T`, and write down the KT crossing criterion in the same convention the code will use.

### Variable Map

The reference Coulomb-gas interaction can be written as

```math
V_{ij}^{\mathrm{paper}}(r)
=
-\frac{q^2}{D} z_i z_j \log\left(\frac{r}{\sigma}\right),
```

where `z_i = +/- 1`, `q^2/D` is the Coulomb energy scale, and `sigma` is the hard-core length scale. The paper's reduced temperature is

```math
T^*
=
\frac{k_B T D}{q^2}.
```

This project's first-term interaction is

```math
V_{ij}^{\mathrm{project}}(r)
=
-K \zeta_i \zeta_j \log\left(\frac{r}{\xi}\right).
```

For the `E0 = 0` benchmark, identify

```math
\zeta_i = z_i,
\qquad
K = \frac{q^2}{D}.
```

Then the dimensionless Coulomb coupling used by this project is

```math
\kappa
=
\frac{K}{k_B T}
=
\frac{q^2}{D k_B T}
=
\frac{1}{T^*}.
```

The code currently uses units with `k_B = 1`, so this is simply

```math
\kappa = \frac{K}{T},
\qquad
T^* = \frac{T}{K},
\qquad
T^* = \frac{1}{\kappa}.
```

### Length And Density Map

The paper's reduced length and density are

```math
r^* = \frac{r}{\sigma},
\qquad
L^* = \frac{L}{\sigma},
\qquad
\rho^* = \rho\sigma^2 = \frac{N}{(L^*)^2}.
```

For reproducing the paper, use the hard-core length as the paper's `sigma`:

```math
\sigma = a.
```

The logarithm length in this project is `xi`. For a direct benchmark, set

```math
\xi = a = \sigma.
```

If `xi` and `a` differ in a neutral fixed-`N` Coulomb-only run, the difference in the log reference length produces only a constant energy shift because

```math
\sum_{i<j} z_i z_j
=
\frac{1}{2}
\left[
\left(\sum_i z_i\right)^2
- \sum_i z_i^2
\right]
=
-\frac{N}{2}
```

for a neutral system. This constant shift does not change Metropolis acceptance, but using `xi = a` avoids ambiguity when comparing with the paper.

### Dielectric Estimator In `kappa` Convention

Starting from the step 1 reduced estimator,

```math
\epsilon_0^{-1}
=
\lim_{\mathbf{k}^*\to 0}
\left[
1
- \frac{2\pi}{(k^*)^2 T^* (L^*)^2}
\left\langle q_{\mathbf{k}^*}q_{-\mathbf{k}^*}\right\rangle
\right],
```

substitute `T* = 1/kappa`:

```math
\epsilon_0^{-1}
=
\lim_{\mathbf{k}^*\to 0}
\left[
1
- \frac{2\pi\kappa}{(k^*)^2 (L^*)^2}
\left\langle q_{\mathbf{k}^*}q_{-\mathbf{k}^*}\right\rangle
\right].
```

The corresponding thermodynamic-limit dipole-fluctuation form becomes

```math
\epsilon_0^{-1}
=
1
- \frac{\pi\kappa}{(L^*)^2}
\left\langle |\mathbf{M}^*|^2 \right\rangle.
```

For implementation, the finite-size low-`k` estimator should be treated as the primary estimator:

```math
\epsilon_0^{-1}(\kappa, L^*)
\approx
1
- \frac{1}{N_k}
\sum_{\mathbf{k}^* \in \mathcal{K}_{\mathrm{low}}}
\frac{2\pi\kappa}{(k^*)^2 (L^*)^2}
\left\langle
q_{\mathbf{k}^*}q_{-\mathbf{k}^*}
\right\rangle.
```

Here `K_low` is the selected set of the lowest allowed nonzero wave vectors, and `N_k` is the number of vectors averaged.

### KT Crossing Criterion

In the paper's convention, the KT transition estimate is the crossing

```math
\epsilon_0^{-1} = 4T^*.
```

Using

```math
T^* = \frac{1}{\kappa},
```

the same crossing criterion in this project's `kappa = K/T` convention is

```math
\epsilon_0^{-1} = \frac{4}{\kappa}.
```

Equivalently, at fixed density and finite size, scan `kappa` and solve

```math
F(\kappa; \rho^*, L^*)
=
\epsilon_0^{-1}(\kappa; \rho^*, L^*)
- \frac{4}{\kappa}
=
0.
```

At zero density, the reference KT point is

```math
T_c^* = \frac{1}{4}
\qquad \Longleftrightarrow \qquad
\kappa_c = 4.
```

So the first sanity check should recover a crossing near `kappa = 4` at very low `rho*` when `E0 = 0`.

### Practical Convention For The Next Code Step

For the Coulomb-only benchmark:

```math
E0 = 0,
\qquad
e0_{\mathrm{red}} = \frac{E0}{T} = 0,
\qquad
\kappa = \frac{K}{T}.
```

If using the current physical-energy engine, set

```math
K = \kappa T,
\qquad
E0 = 0.
```

If rewriting to a reduced-energy engine, use

```math
\beta V_{ij}
=
-\kappa z_i z_j \log\left(\frac{r}{\xi}\right),
```

and accept moves with

```math
\min\left(1, \exp[-\Delta(\beta V)]\right).
```

### Status

Step 2 is documented. The next task is to choose one parameter convention for all new simulation code.

## 2026-04-17: Code Representation Of `epsilon_0^{-1}`

Task:

> Write out the code representation of the equation for `1/epsilon_0` into `1/epsilon_0.py`, using the sign convention from `Monte_Carlo_sim.py`, `plotting.py`, and `full_approx.ipynb`.

### Progress

Created `1/epsilon_0.py` as a standalone estimator module. Since a slash cannot be part of a single filename on disk, this is implemented as the path `1/epsilon_0.py`: a directory named `1` containing `epsilon_0.py`.

The code follows the existing project sign convention:

- `charges = +1` for positive particles.
- `charges = -1` for negative particles.
- The Coulomb energy convention is `-K * qi * qj * log(r / xi)`.
- The reduced Coulomb coupling is `kappa = K / T`.
- Therefore `T* = 1 / kappa` for the `E0 = 0` Coulomb-gas benchmark.

### Implemented Equation

The finite-size estimator implemented in code is

```math
\epsilon_0^{-1}
\approx
1
-
\frac{1}{N_k}
\sum_{\mathbf{k}\in\mathcal{K}_{\mathrm{low}}}
\frac{2\pi\kappa}{k^2L^2}
\left\langle q_{\mathbf{k}}q_{-\mathbf{k}}\right\rangle,
```

where

```math
q_{\mathbf{k}}
=
\sum_{j=1}^{N}
z_j
\exp(-i\mathbf{k}\cdot\mathbf{r}_j).
```

Because `q_{-k}` is the complex conjugate of `q_k` for real charge signs, the code computes

```math
q_{\mathbf{k}}q_{-\mathbf{k}}
=
|q_{\mathbf{k}}|^2.
```

The helper functions are:

- `low_k_vectors(L, n_shells=3)`: builds the low nonzero wave vectors for a square periodic box.
- `charge_density_mode(pos, charges, kvec)`: computes `q_k`.
- `epsilon_inverse_from_samples(samples, charges, L, kappa, ...)`: computes `epsilon_0^{-1}` from one or many sampled configurations.
- `kt_line_kappa(kappa)`: returns the KT line `4 / kappa`.
- `kt_crossing_residual(epsilon_inv, kappa)`: returns `epsilon_inv - 4 / kappa`.

### Notes

The estimator accepts either one configuration with shape `(N, 2)` or many configurations with shape `(n_samples, N, 2)`. It expects snapshots after burn-in filtering; burn-in selection is intentionally kept outside this formula module.

The default wave-vector average uses the first three nonzero squared-magnitude shells. This is the code-level version of the low-`k` averaging described in step 1.

### Status

The code representation has been written. Syntax validation passed with `python3 -m py_compile 1/epsilon_0.py`. A small neutral two-particle smoke test also loaded the module by path, constructed the lowest wave-vector shell, evaluated `epsilon_inverse_from_samples`, and checked that `kt_line_kappa(4.0)` returns `1.0`.
