"""
Inverse dielectric constant 1/eps0 of the 2D logarithmic Coulomb gas, via
the charge-structure-factor / linear-response formula of Kosterlitz-Thouless
theory (Nelson & Kosterlitz 1977), as used e.g. in Orkoulas & Panagiotopoulos,
J. Chem. Phys. 104, 7205 (1996), Eq. (9):

    1/eps0 = lim_{k->0} [ 1 - (2*pi / T*) * (1/k^2) * <n(k) n(-k)> ]

where n(k) is the Fourier transform of the charge density on the reciprocal
lattice of the (unit) simulation box, and T* = k_B*T*D/q^2 is the reduced
temperature. In this codebase's units (V = -zeta*zeta'*K*ln(r/xi), kappa =
K/T), T* = 1/kappa.

The universal Nelson-Kosterlitz jump condition for the KT transition is

    1/(eps0 * T*) = 4   =>   1/eps0 = 4*T*.

n(k) at the smallest nonzero reciprocal-lattice vector k = 2*pi*m (integer
2-vector m, |m|^2 = 1 for the box's unit square) is exactly the same
quantity already used in Ewald_Method_pbc.py's reciprocal-space sum:

    rho_m = sum_j zeta_j * exp(2*pi*i * m . R_j)

but WITHOUT the Ewald Gaussian screening factor -- this is a raw structure
factor measurement over sampled configurations, not an energy term, so it
does not touch or reuse ewald_recip_energy1 itself.

This module only post-processes the position snapshots that run_mc1 /
run_mc already return; it does not change the Monte Carlo core.
"""

import numpy as np


def charge_structure_factor(R_all, q, m_vec):
    """|rho_m|^2 for one configuration and one reciprocal-lattice vector m_vec."""
    R_all = np.asarray(R_all, dtype=float)
    q = np.asarray(q, dtype=float)
    m_vec = np.asarray(m_vec, dtype=float)

    phase = R_all @ m_vec
    rho_m = np.sum(q * np.exp(2j * np.pi * phase))

    return np.abs(rho_m) ** 2


def smallest_m_shell():
    """The four smallest nonzero reciprocal-lattice vectors of a unit box, |m|^2=1."""
    return np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)


def inverse_dielectric_constant(snapshots, q, kappa, m_shell=None):
    """
    Estimate 1/eps0 from a set of position snapshots at fixed kappa, using
    Eq. (9) evaluated at the smallest available reciprocal-lattice shell
    (finite-k proxy for the k->0 limit, as in Orkoulas & Panagiotopoulos).

    Returns (inv_eps0, S_avg, sem) where S_avg = <|rho_m|^2> averaged over
    the shell vectors and over snapshots, and sem is its standard error.
    """
    if m_shell is None:
        m_shell = smallest_m_shell()
    else:
        m_shell = np.asarray(m_shell, dtype=float)

    m2 = np.sum(m_shell ** 2, axis=1)
    if not np.allclose(m2, m2[0]):
        raise ValueError("m_shell vectors must all have the same |m|^2.")
    m2_val = m2[0]

    S_per_snapshot = np.array([
        np.mean([charge_structure_factor(R_all, q, m) for m in m_shell])
        for R_all in snapshots
    ])

    S_avg = np.mean(S_per_snapshot)
    S_sem = np.std(S_per_snapshot, ddof=1) / np.sqrt(len(S_per_snapshot))

    k2 = (2.0 * np.pi) ** 2 * m2_val

    inv_eps0 = 1.0 - (2.0 * np.pi * kappa) * S_avg / k2
    inv_eps0_sem = (2.0 * np.pi * kappa) * S_sem / k2

    return inv_eps0, S_avg, inv_eps0_sem
