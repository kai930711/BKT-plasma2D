def ewald_polarization_energy1(R_all, q):
    R_all = np.asarray(R_all, dtype=float)
    q = np.asarray(q, dtype=float)

    P = np.sum(q[:, None] * R_all, axis=0)

    return 0.5 * np.pi * np.dot(P, P)


def ewald_energy1(R_all, q, alpha, n_cut, m_cut, include_polarization=True):
    U1_self = ewald_self_energy1(q, alpha)
    U1_real = ewald_real_energy1(R_all, q, alpha, n_cut)
    U1_recip = ewald_recip_energy1(R_all, q, alpha, m_cut)

    if include_polarization:
        U1_pol = ewald_polarization_energy1(R_all, q)
    else:
        U1_pol = 0.0

    return U1_self + U1_real + U1_recip + U1_pol