# Current Plan

Our first task is to set `E_0 = 0` and `K = 0` respectively to generate the `K` or `K/T` vs. `rho` and `E_0` or `E_0/T` vs. `rho` diagrams. The former case was already discussed in Orkoulas & Panagiotopoulos (1996), so our goal is to reproduce Fig. 6 and generate a similar `E_0` vs. `rho` diagram based on the cosh interaction. After that, we can try to figure out `K` vs. `E_0`.

It seems that a consistent plan is to follow Orkoulas & Panagiotopoulos (1996). They compute the inverse dielectric function `1/epsilon_0` at fixed density as a function of temperature, using its intersection with the line `4T*` to estimate the KT transition. So the reasonable steps are:

1. Choose a `rho`.
2. Derive a set of curves `1/epsilon_0(T, L)`.
3. Find and record their intersections with `1/epsilon_0 = 4T*` as `T_c(L)`.
4. Draw `T_c(L)` vs. `1/L`, and estimate the `L -> infinity` limit as `T_c`.
5. Repeat steps 1-3 to get a set of `(rho, T_c)`, and generate the final graph.

For step 4, Orkoulas & Panagiotopoulos (1996) point out that "the KT transition temperature ... is not too sensitive to the system size. We have performed simulations for N=50 and N=200 for all densities studied but do not show the data in Fig. 4 for clarity (p. 7208)." So the estimation part may not be strictly necessary, but we can still try to do it.

# Revised Next Steps

1. Figure out the exact formula for the inverse dielectric function `1/epsilon_0` used by Orkoulas & Panagiotopoulos, including its normalization in terms of `T*`, density, box size, charges, and dipole fluctuations.

2. Translate the paper's reduced variables into this project's variables, especially the relation between `T*` and `kappa = K/T`, and write down the KT crossing criterion in the same convention the code will use.

3. Decide one parameter convention for all new simulation code: preferably reduced variables `kappa = K/T` and `e0_red = E0/T`, with Metropolis acceptance using `exp(-delta_beta_V)`.

4. Implement a dielectric observable for the `E0 = 0` Coulomb-gas case and verify that it is computed only from equilibrated samples after burn-in.

5. Run a small sanity scan at very low density with `E0 = 0` and check that the estimated KT transition is near `kappa ~= 4`, equivalent to `T* ~= 1/4`.

6. Define a fixed-density simulation protocol: choose `rho*`, choose neutral particle counts with `N_plus = N_minus`, and set `L` from `rho* = N a^2 / L^2`.

7. Reproduce the Coulomb-gas KT line by scanning `kappa` or `T*` at fixed `rho*`, computing `1/epsilon_0`, and locating the intersection with the KT criterion.

8. Repeat the Coulomb-gas calculation for at least two system sizes, such as `N = 50` and `N = 200`, while keeping `rho*` fixed.

9. Compare the reproduced `E0 = 0` phase boundary against Fig. 6 of Orkoulas & Panagiotopoulos and record the expected discrepancy from using minimum-image logarithmic interactions instead of the paper's long-range treatment.

10. Before studying `K = 0`, decide what physical feature the cosh-only interaction should diagnose, because the charge-independent attraction does not directly define a KT unbinding transition.

11. Add observables for the `K = 0` cosh-only case: cluster-size distribution, all-pair radial distribution, energy fluctuations, and density/aggregation diagnostics.

12. Run exploratory `K = 0` scans over `e0_red = E0/T` and `rho*` to identify clustering or liquid-gas-like behavior, without labeling the result as a KT line unless a valid criterion is derived.

13. After validating both limiting cases, run full-interaction scans over `(kappa, e0_red)` at selected densities to study how the Coulomb-gas KT behavior changes when the Majorana/cosh attraction is included.

14. Add nonlocal trial moves, such as pair displacement or cluster moves, if local Metropolis sampling becomes inefficient in strongly bound or clustered regimes.

15. Only attempt a quantitative phase diagram after the dielectric estimator, finite-size checks, sampling diagnostics, and limiting-case benchmarks are working.
