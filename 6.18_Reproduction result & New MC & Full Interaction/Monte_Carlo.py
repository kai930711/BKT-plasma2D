import numpy as np

from Setup import hardcore_check
from Plasma_Interaction import (
    ewald_energy1,
    energy2,
    ewald_recip_state1,
    energy_total_diff,
)


def initialize_positions(N, a, rng, max_attempts=10000):
    R_all = np.empty((N, 2), dtype=float)

    for i in range(N):
        for attempt in range(max_attempts):
            R_trial = rng.random(2)

            if i == 0:
                R_all[i] = R_trial
                break

            if not hardcore_check(R_trial, R_all[:i], a):
                R_all[i] = R_trial
                break

        else:
            raise RuntimeError(
                f"Could not place particle {i} after {max_attempts} attempts."
            )

    return R_all


def run_mc_by_diff(
    N,
    q,
    kappa,
    a,
    alpha,
    n_cut,
    m_cut,
    step_size,
    n_sweeps,
    sample_every,
    E0_over_T=0.0,
    xiF_over_xi=1.0,
    include_log=True,
    include_cosh=False,
    n_burnin=0,
    seed=0,
):

    rng = np.random.default_rng(seed)

    q = np.asarray(q, dtype=float)

    if len(q) != N:
        raise ValueError("len(q) must equal N.")

    if include_log and not np.isclose(np.sum(q), 0.0):
        raise ValueError("The logarithmic Ewald sum requires charge neutrality: sum(q) = 0.")

    R_all = initialize_positions(N, a, rng)

    U1_current = 0.0
    U2_current = 0.0
    rho = None
    m_vecs = None
    weight = None

    if include_log:
        U1_current = ewald_energy1(
            R_all,
            q,
            alpha,
            n_cut,
            m_cut,
        )
        m_vecs, weight, rho = ewald_recip_state1(
            R_all,
            q,
            alpha,
            m_cut,
        )

    if include_cosh:
        U2_current = energy2(
            R_all,
            E0_over_T,
            xiF_over_xi,
            a,
        )

    beta_U_current = 0.0
    if include_log:
        beta_U_current += kappa * U1_current
    if include_cosh:
        beta_U_current += U2_current

    beta_U_trace = []
    U1_trace = []
    U2_trace = []
    snapshots = []

    accept = 0
    trials = 0

    total_sweeps = n_burnin + n_sweeps

    for sweep in range(total_sweeps):
        for _ in range(N):
            i = rng.integers(N)

            R_old_i = R_all[i].copy()
            R_trial_i = (R_old_i + rng.uniform(-step_size, step_size, size=2)) % 1.0

            trials += 1

            if hardcore_check(R_trial_i, R_all, a, exclude_index=i):
                continue

            d_beta_U, dU1, dU2, delta_rho = energy_total_diff(
                R_all=R_all,
                q=q,
                i=i,
                R_new_i=R_trial_i,
                kappa=kappa,
                alpha=alpha,
                n_cut=n_cut,
                m_cut=m_cut,
                E0_over_T=E0_over_T,
                xiF_over_xi=xiF_over_xi,
                a=a,
                rho=rho,
                m_vecs=m_vecs,
                weight=weight,
                include_log=include_log,
                include_cosh=include_cosh,
            )

            if np.isfinite(d_beta_U) and (
                d_beta_U <= 0.0 or rng.random() < np.exp(-d_beta_U)
            ):
                R_all[i] = R_trial_i

                if include_log:
                    U1_current += dU1
                    rho = rho + delta_rho

                if include_cosh:
                    U2_current += dU2

                beta_U_current += d_beta_U
                accept += 1

        if sweep >= n_burnin:
            production_sweep = sweep - n_burnin

            if production_sweep % sample_every == 0:
                beta_U_trace.append(beta_U_current)
                U1_trace.append(U1_current)
                U2_trace.append(U2_current)
                snapshots.append(R_all.copy())

    acceptance_rate = accept / trials

    return {
        "snapshots": snapshots,
        "beta_U_trace": np.asarray(beta_U_trace, dtype=float),
        "U1_trace": np.asarray(U1_trace, dtype=float),
        "U2_trace": np.asarray(U2_trace, dtype=float),
        "acceptance_rate": acceptance_rate,
    }