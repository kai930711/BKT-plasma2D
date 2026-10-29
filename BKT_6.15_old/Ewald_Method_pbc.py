import numpy as np
from scipy.special import exp1


def ewald_self_energy1(q, alpha):
    q = np.asarray(q, dtype=float)

    gamma = np.euler_gamma
    return -0.25 * (gamma + np.log(alpha**2)) * np.sum(q**2)


def ewald_real_energy1(R_all, q, alpha, n_cut):
    R_all = np.asarray(R_all, dtype=float)
    q = np.asarray(q, dtype=float)

    U_real = 0.0

    qq = q[:, None] * q[None, :]

    for nx in range(-n_cut, n_cut + 1):
        for ny in range(-n_cut, n_cut + 1):
            n_vec = np.array([nx, ny], dtype=float)

            dR = R_all[:, None, :] - R_all[None, :, :] + n_vec
            r2 = np.sum(dR**2, axis=2)

            if nx == 0 and ny == 0:
                mask = ~np.eye(len(q), dtype=bool)
                U_real += np.sum(qq[mask] * exp1(alpha**2 * r2[mask]))
            else:
                U_real += np.sum(qq * exp1(alpha**2 * r2))

    return 0.25 * U_real


def ewald_recip_energy1(R_all, q, alpha, m_cut):
    R_all = np.asarray(R_all, dtype=float)
    q = np.asarray(q, dtype=float)

    m_list = []

    for mx in range(-m_cut, m_cut + 1):
        for my in range(-m_cut, m_cut + 1):
            if mx == 0 and my == 0:
                continue
            m_list.append([mx, my])

    m_vecs = np.asarray(m_list, dtype=float)

    m2 = np.sum(m_vecs**2, axis=1)

    phase = R_all @ m_vecs.T
    rho = q @ np.exp(2j * np.pi * phase)

    weight = np.exp(-np.pi**2 * m2 / alpha**2) / m2

    U_recip = np.sum(weight * np.abs(rho)**2)

    return U_recip / (4.0 * np.pi)


def ewald_energy1(R_all, q, alpha, n_cut, m_cut):
    U1_self = ewald_self_energy1(q, alpha)
    U1_real = ewald_real_energy1(R_all, q, alpha, n_cut)
    U1_recip = ewald_recip_energy1(R_all, q, alpha, m_cut)
    return U1_self + U1_real + U1_recip