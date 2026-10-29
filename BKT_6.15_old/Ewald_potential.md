# Logarithmic Ewald energy convention

We simulate a neutral two-component 2D Coulomb gas with charges

$$
q_i = \pm 1, \qquad \sum_i q_i = 0,
$$

in a unit square with periodic boundary conditions. Coordinates are stored as

$$
R_i = \frac{r_i}{L}, \qquad R_i \in [0,1)^2.
$$

The dimensionless logarithmic Ewald energy used in the code is

$$
U_1^{\rm code}
=
U_{\rm self}
+
U_{\rm real}
+
U_{\rm recip}.
$$

The self term is

$$
U_{\rm self}
=
-\frac{1}{4}
\left(
\gamma + \ln \alpha^2
\right)
\sum_i q_i^2 .
$$

The real-space term is

$$
U_{\rm real}
=
\frac{1}{4}
\sum_{n\in\mathbb{Z}^2}
\sum_{i,j}^{\prime}
q_i q_j
E_1\left(
\alpha^2 |R_i - R_j + n|^2
\right),
$$

where the prime means that the \(i=j, n=0\) term is omitted.

The reciprocal-space term is

$$
U_{\rm recip}
=
\frac{1}{4\pi}
\sum_{m\in\mathbb{Z}^2,\ m\ne 0}
\frac{
\exp\left(-\pi^2 |m|^2/\alpha^2\right)
}{
|m|^2
}
|\rho_m|^2,
$$

with

$$
\rho_m
=
\sum_j q_j e^{2\pi i m\cdot R_j}.
$$

The sums are truncated by `n_cut` and `m_cut`.

The Metropolis exponent for the first logarithmic term is

$$
\Delta S = \kappa \Delta U_1,
\qquad
\kappa = K/T.
$$

Therefore the acceptance probability is

$$
P_{\rm acc}
=
\min\left(1, e^{-\kappa \Delta U_1}\right).
$$

## Boundary convention

The default convention is strict torus/conducting periodic boundary condition:

$$
U_1^{\rm code}
=
U_{\rm self}
+
U_{\rm real}
+
U_{\rm recip}.
$$

The Ewald parameter `alpha` and the cutoffs `n_cut`, `m_cut` are numerical parameters, not physical parameters. Results should be checked for convergence by varying them at fixed configuration.