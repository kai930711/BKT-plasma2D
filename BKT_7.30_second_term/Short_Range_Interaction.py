"""
Short-range "cosh" pair interaction.

Physical interaction (Fermi-liquid-screened logarithmic potential):

    V_{zeta,zeta'}(r) = -zeta*zeta' * K * ln(r/xi)
                         - T * ln[2 * cosh((E0/T) * exp(-r/xi_F))]

The first (logarithmic) term is long-ranged and is handled exclusively by
Ewald_Method_pbc.py / ewald_energy1 (U1). This module implements ONLY the
second term, which is short-ranged, charge-independent, and is therefore
evaluated with a plain minimum-image sum instead of Ewald summation.

Dimensionless form
-------------------
Define

    x_ij = (E0/T) * exp(-r_ij / xi_F) = E0_over_T * exp(-r_ij / xi_F)

so that beta * V2(r) = -ln[2 * cosh(x)].

For Monte Carlo sampling at fixed particle number N, the constant
-ln(2) per pair does not depend on the configuration, so it can be dropped
from the energy used for Metropolis acceptance ("shifted" energy):

    V2_shifted(r) = -ln cosh(x)              (per pair, no -ln(2))
    beta*V2(r)    = V2_shifted(r) - ln(2)     (exact, physical)

Summed over all i < j pairs:

    U2_shifted = sum_{i<j} V2_shifted(r_ij)
    U2_exact   = U2_shifted - [N(N-1)/2] * ln(2)

Numerical stability
--------------------
-ln(cosh(x)) is evaluated as

    -ln(cosh(x)) = ln(2) - logaddexp(x, -x)

using numpy.logaddexp, which is stable for large |x| (it never forms
exp(x) directly), instead of the naive -log(cosh(x)) that overflows for
x greater than about 350-700.

Units / conventions
--------------------
Positions R_all are stored as fractional coordinates in [0, 1) x [0, 1)
(box length L = 1), matching the convention already used by
Ewald_Method_pbc.py and Setup.py. All length-like parameters passed to
the functions below (r, xi_F, L) must be expressed in these same
"box units" (i.e. physical length / L). If your physical xi_F is given
in units of the hard-core radius a (xiF_over_xi = xi_F_physical / a),
convert with

    xi_F_box = xiF_over_xi * a

before calling these functions -- do NOT assume xi = a unless that is the
convention you have independently chosen; this module makes no such
assumption and only ever consumes xi_F in box units directly.
"""

import numpy as np


def minimum_image_displacement(dR, L=1.0):
    """Minimum-image displacement for dR = Ri - Rj, box length L."""
    dR = np.asarray(dR, dtype=float)
    return dR - L * np.rint(dR / L)


def minimum_image_distance(Ri, Rj, L=1.0):
    """Minimum-image scalar distance(s) between Ri and Rj (box length L)."""
    dR = minimum_image_displacement(np.asarray(Ri, dtype=float) - np.asarray(Rj, dtype=float), L=L)
    return np.sqrt(np.sum(dR**2, axis=-1))


def energy2_pair(r, E0_over_T, xi_F):
    """
    Shifted second-interaction pair energy V2_shifted(r) = -ln cosh(x),
    x = E0_over_T * exp(-r / xi_F).

    Numerically stable for arbitrarily large x via
        -ln cosh(x) = ln(2) - logaddexp(x, -x).

    r may be a scalar or an array; the return has the same shape.
    """
    if xi_F <= 0:
        raise ValueError("xi_F must be positive.")

    r = np.asarray(r, dtype=float)
    x = E0_over_T * np.exp(-r / xi_F)

    return np.log(2.0) - np.logaddexp(x, -x)


def energy2(R_all, E0_over_T, xi_F, L=1.0):
    """
    Total shifted second-interaction energy U2_shifted, summed once over
    all pairs i < j, using the minimum-image distance (no Ewald sum, no
    periodic images beyond the nearest one).
    """
    R_all = np.asarray(R_all, dtype=float)
    N = R_all.shape[0]

    if N < 2:
        return 0.0

    idx_i, idx_j = np.triu_indices(N, k=1)

    dR = minimum_image_displacement(R_all[idx_i] - R_all[idx_j], L=L)
    r = np.sqrt(np.sum(dR**2, axis=1))

    return float(np.sum(energy2_pair(r, E0_over_T, xi_F)))


def energy2_diff(R_all, i, R_i_old, R_i_new, E0_over_T, xi_F, L=1.0):
    """
    O(N) change in the shifted second-interaction energy from moving
    particle i from R_i_old to R_i_new, holding all other particles fixed:

        dU2 = sum_{j != i} [V2(R_i_new, Rj) - V2(R_i_old, Rj)]

    R_all is used only to supply the (fixed) positions of the other
    particles; R_all[i] itself is ignored (R_i_old/R_i_new are used
    instead), so this is safe to call whether or not R_all[i] has already
    been overwritten with the trial position.
    """
    R_all = np.asarray(R_all, dtype=float)
    N = R_all.shape[0]

    mask = np.ones(N, dtype=bool)
    mask[i] = False
    R_others = R_all[mask]

    if R_others.shape[0] == 0:
        return 0.0

    dR_new = minimum_image_displacement(np.asarray(R_i_new, dtype=float) - R_others, L=L)
    r_new = np.sqrt(np.sum(dR_new**2, axis=1))

    dR_old = minimum_image_displacement(np.asarray(R_i_old, dtype=float) - R_others, L=L)
    r_old = np.sqrt(np.sum(dR_old**2, axis=1))

    e_new = energy2_pair(r_new, E0_over_T, xi_F)
    e_old = energy2_pair(r_old, E0_over_T, xi_F)

    return float(np.sum(e_new - e_old))


def energy2_exact_from_shifted(U2_shifted, N):
    """
    Recover the exact absolute second energy (matching the original
    -T ln[2 cosh(...)] equation) from the shifted energy used for MC
    acceptance:

        U2_exact = U2_shifted - [N(N-1)/2] * ln(2)
    """
    n_pairs = N * (N - 1) / 2.0
    return U2_shifted - n_pairs * np.log(2.0)
