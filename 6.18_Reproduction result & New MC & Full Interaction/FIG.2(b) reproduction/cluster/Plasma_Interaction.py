import numpy as np
from scipy.special import exp1
from Setup import minimum_image_displacement


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


def energy2_pair(Ri, Rj, E0_over_T, xiF_over_xi, a):
    xiF = xiF_over_xi * a
    dR = minimum_image_displacement(Ri - Rj)
    r = np.linalg.norm(dR)

    x = E0_over_T * np.exp(-r / xiF)

    return -(np.logaddexp(x, -x) - np.log(2.0))


def energy2(R_all, E0_over_T, xiF_over_xi, a):
    U2 = 0.0
    N = len(R_all)

    for i in range(N):
        for j in range(i + 1, N):
            U2 += energy2_pair(R_all[i], R_all[j], E0_over_T, xiF_over_xi, a)

    return U2


def energy_total(
    R_all,
    q,
    kappa,
    alpha,
    n_cut,
    m_cut,
    E0_over_T,
    xiF_over_xi,
    a,
    include_log=True,
    include_cosh=False,
):
    beta_U = 0.0

    if include_log:
        U1 = ewald_energy1(R_all, q, alpha, n_cut, m_cut)
        beta_U += kappa * U1

    if include_cosh:
        U2 = energy2(R_all, E0_over_T, xiF_over_xi, a)
        beta_U += U2

    return beta_U


# Since Monte Carlo only relies on energy differences, we can compute the change in energy to reduce computation workload. 


def make_m_vectors(m_cut):
    m_list = []

    for mx in range(-m_cut, m_cut + 1):
        for my in range(-m_cut, m_cut + 1):
            if mx == 0 and my == 0:
                continue
            m_list.append([mx, my])

    return np.asarray(m_list, dtype=float)


def ewald_recip_weight1(alpha, m_vecs):
    m_vecs = np.asarray(m_vecs, dtype=float)
    m2 = np.sum(m_vecs**2, axis=1)

    return np.exp(-np.pi**2 * m2 / alpha**2) / m2


def ewald_rho1(R_all, q, m_vecs):
    R_all = np.asarray(R_all, dtype=float)
    q = np.asarray(q, dtype=float)
    m_vecs = np.asarray(m_vecs, dtype=float)

    phase = R_all @ m_vecs.T

    return q @ np.exp(2j * np.pi * phase)


def ewald_recip_state1(R_all, q, alpha, m_cut):
    m_vecs = make_m_vectors(m_cut)
    weight = ewald_recip_weight1(alpha, m_vecs)
    rho = ewald_rho1(R_all, q, m_vecs)

    return m_vecs, weight, rho


def ewald_real_energy1_diff(R_all, q, i, R_new_i, alpha, n_cut):

    R_all = np.asarray(R_all, dtype=float)
    q = np.asarray(q, dtype=float)
    R_new_i = np.asarray(R_new_i, dtype=float)

    R_old_i = R_all[i].copy()
    qi = q[i]

    mask = np.ones(len(q), dtype=bool)
    mask[i] = False

    Rj = R_all[mask]
    qj = q[mask]

    delta_sum = 0.0

    for nx in range(-n_cut, n_cut + 1):
        for ny in range(-n_cut, n_cut + 1):
            n_vec = np.array([nx, ny], dtype=float)

            dR_new = R_new_i[None, :] - Rj + n_vec
            dR_old = R_old_i[None, :] - Rj + n_vec

            r2_new = np.sum(dR_new**2, axis=1)
            r2_old = np.sum(dR_old**2, axis=1)

            delta_sum += np.sum(
                qj * (
                    exp1(alpha**2 * r2_new)
                    - exp1(alpha**2 * r2_old)
                )
            )

    return 0.5 * qi * delta_sum


def ewald_recip_energy1_diff(R_all, q, i, R_new_i, rho, m_vecs, weight):
    
    R_all = np.asarray(R_all, dtype=float)
    q = np.asarray(q, dtype=float)
    R_new_i = np.asarray(R_new_i, dtype=float)
    rho = np.asarray(rho, dtype=complex)
    m_vecs = np.asarray(m_vecs, dtype=float)
    weight = np.asarray(weight, dtype=float)

    R_old_i = R_all[i].copy()
    qi = q[i]

    phase_old = R_old_i @ m_vecs.T
    phase_new = R_new_i @ m_vecs.T

    delta_rho = qi * (
        np.exp(2j * np.pi * phase_new)
        - np.exp(2j * np.pi * phase_old)
    )

    dU_recip = np.sum(
        weight * (
            np.abs(rho + delta_rho)**2
            - np.abs(rho)**2
        )
    ) / (4.0 * np.pi)

    return dU_recip, delta_rho


def energy2_diff(R_all, i, R_new_i, E0_over_T, xiF_over_xi, a):
    
    R_all = np.asarray(R_all, dtype=float)
    R_new_i = np.asarray(R_new_i, dtype=float)

    R_old_i = R_all[i].copy()

    dU2 = 0.0

    for j in range(len(R_all)):
        if j == i:
            continue

        U_old = energy2_pair(
            R_old_i,
            R_all[j],
            E0_over_T,
            xiF_over_xi,
            a,
        )

        U_new = energy2_pair(
            R_new_i,
            R_all[j],
            E0_over_T,
            xiF_over_xi,
            a,
        )

        dU2 += U_new - U_old

    return dU2


def energy_total_diff(
    R_all,
    q,
    i,
    R_new_i,
    kappa,
    alpha,
    n_cut,
    m_cut,
    E0_over_T,
    xiF_over_xi,
    a,
    rho,
    m_vecs,
    weight,
    include_log=True,
    include_cosh=False,
):

    d_beta_U = 0.0
    dU1 = 0.0
    dU2 = 0.0
    delta_rho = None

    if include_log:
        dU_real = ewald_real_energy1_diff(
            R_all=R_all,
            q=q,
            i=i,
            R_new_i=R_new_i,
            alpha=alpha,
            n_cut=n_cut,
        )

        dU_recip, delta_rho = ewald_recip_energy1_diff(
            R_all=R_all,
            q=q,
            i=i,
            R_new_i=R_new_i,
            rho=rho,
            m_vecs=m_vecs,
            weight=weight,
        )

        dU1 = dU_real + dU_recip
        d_beta_U += kappa * dU1

    if include_cosh:
        dU2 = energy2_diff(
            R_all=R_all,
            i=i,
            R_new_i=R_new_i,
            E0_over_T=E0_over_T,
            xiF_over_xi=xiF_over_xi,
            a=a,
        )

        d_beta_U += dU2

    return d_beta_U, dU1, dU2, delta_rho