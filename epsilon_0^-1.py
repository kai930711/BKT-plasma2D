"""Inverse dielectric estimator for the 2D Coulomb gas.

Sign convention matches this project:
- charges are integer signs, +1 for positive charges and -1 for negative charges;
- the Coulomb energy is -K * qi * qj * log(r / xi);
- kappa is the reduced coupling K / T.

With this convention, T* = 1 / kappa and the KT crossing line is
epsilon_0^{-1} = 4 / kappa.
"""

import numpy as np


def low_k_vectors(L, n_shells=3):
    """Return the lowest nonzero wave-vector shells for a square box.

    The returned vectors are dimensional wave vectors,

        k = (2*pi/L) * (nx, ny).

    The zero vector is excluded. All lattice vectors in the first `n_shells`
    squared-magnitude shells are included, including +/- directions. Including
    both signs is harmless because q_k q_-k = |q_k|^2, and it gives a direct
    average over the selected low-k vector set.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if n_shells < 1:
        raise ValueError("n_shells must be at least 1")

    max_n = int(np.ceil(np.sqrt(n_shells))) + n_shells
    indices = []
    for nx in range(-max_n, max_n + 1):
        for ny in range(-max_n, max_n + 1):
            n2 = nx * nx + ny * ny
            if n2 > 0:
                indices.append((n2, nx, ny))

    shell_values = sorted({n2 for n2, _, _ in indices})[:n_shells]
    selected = [(nx, ny) for n2, nx, ny in indices if n2 in shell_values]
    selected.sort(key=lambda item: (item[0] * item[0] + item[1] * item[1], item[0], item[1]))

    return (2.0 * np.pi / L) * np.asarray(selected, dtype=float)


def charge_density_mode(pos, charges, kvec):
    """Compute q_k = sum_j z_j exp(-i k dot r_j)."""
    pos = np.asarray(pos, dtype=float)
    charges = np.asarray(charges, dtype=float)
    kvec = np.asarray(kvec, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape (N, 2)")
    if charges.shape != (pos.shape[0],):
        raise ValueError("charges must have shape (N,)")
    if kvec.shape != (2,):
        raise ValueError("kvec must have shape (2,)")

    phase = pos @ kvec
    return np.sum(charges * np.exp(-1j * phase))


def epsilon_inverse_from_samples(samples, charges, L, kappa, k_vectors=None, n_shells=3):
    """Estimate epsilon_0^{-1} from sampled configurations.

    Parameters
    ----------
    samples : array_like
        Either one configuration with shape (N, 2), or many configurations
        with shape (n_samples, N, 2). These can be the `snapshots` returned by
        `Monte_Carlo_sim.run_mc` after burn-in filtering.
    charges : array_like
        Charge signs using the project convention: +1 and -1.
    L : float
        Simulation-box side length in the same units as `samples`.
    kappa : float
        Reduced Coulomb coupling K / T.
    k_vectors : array_like, optional
        Explicit dimensional wave vectors with shape (N_k, 2). If omitted,
        the first `n_shells` nonzero wave-vector shells are used.
    n_shells : int
        Number of nonzero wave-vector shells to average when `k_vectors` is
        omitted. The default, 3, follows the low-k averaging convention noted
        for Orkoulas & Panagiotopoulos.

    Returns
    -------
    float
        The finite-size low-k estimate

            epsilon_0^{-1}
            = 1 - average_k[2*pi*kappa*<q_k q_-k> / (k^2 L^2)].
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")

    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 2:
        samples = samples[np.newaxis, :, :]
    if samples.ndim != 3 or samples.shape[2] != 2:
        raise ValueError("samples must have shape (N, 2) or (n_samples, N, 2)")

    charges = np.asarray(charges, dtype=float)
    if charges.shape != (samples.shape[1],):
        raise ValueError("charges must have shape (N,)")
    if not np.isclose(np.sum(charges), 0.0):
        raise ValueError("dielectric estimator expects a neutral charge configuration")

    if k_vectors is None:
        k_vectors = low_k_vectors(L, n_shells=n_shells)
    else:
        k_vectors = np.asarray(k_vectors, dtype=float)

    if k_vectors.ndim != 2 or k_vectors.shape[1] != 2:
        raise ValueError("k_vectors must have shape (N_k, 2)")
    if len(k_vectors) == 0:
        raise ValueError("at least one nonzero k vector is required")

    terms = []
    for kvec in k_vectors:
        k2 = float(np.dot(kvec, kvec))
        if k2 == 0.0:
            raise ValueError("k_vectors must not contain the zero vector")

        q2_samples = []
        for pos in samples:
            qk = charge_density_mode(pos, charges, kvec)
            q2_samples.append(float(np.real(qk * np.conjugate(qk))))

        mean_q2 = float(np.mean(q2_samples))
        terms.append((2.0 * np.pi * kappa * mean_q2) / (k2 * L * L))

    return 1.0 - float(np.mean(terms))


def kt_line_kappa(kappa):
    """Universal KT crossing line in this project's kappa convention."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return 4.0 / kappa


def kt_crossing_residual(epsilon_inv, kappa):
    """Residual whose zero estimates the KT crossing at fixed density and size."""
    return float(epsilon_inv) - kt_line_kappa(kappa)
