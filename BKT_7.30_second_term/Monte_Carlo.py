import numpy as np
from Setup import hardcore_check
from Ewald_Method_pbc import ewald_energy1
from Short_Range_Interaction import (
    energy2,
    energy2_diff,
    energy2_exact_from_shifted,
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


def run_mc1(
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
    n_burnin=0,
    seed=0,
):
    
    rng = np.random.default_rng(seed)

    q = np.asarray(q, dtype=float)

    if len(q) != N:
        raise ValueError("len(q) must equal N.")

    if not np.isclose(np.sum(q), 0.0):
        raise ValueError("The logarithmic Ewald sum requires charge neutrality: sum(q) = 0.")

    R_all = initialize_positions(N, a, rng)

    U1_current = ewald_energy1(
        R_all,
        q,
        alpha,
        n_cut,
        m_cut,
    )

    U1_trace = []
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

            R_all[i] = R_trial_i

            U1_trial = ewald_energy1(
                R_all,
                q,
                alpha,
                n_cut,
                m_cut,
            )

            dU1 = U1_trial - U1_current
    
            if np.isfinite(dU1) and (dU1 <= 0.0 or rng.random() < np.exp(-kappa * dU1)):
                U1_current = U1_trial
                accept += 1
            else:
                R_all[i] = R_old_i

        if sweep >= n_burnin:
            production_sweep = sweep - n_burnin
    
            if production_sweep % sample_every == 0:
                U1_trace.append(U1_current)
                snapshots.append(R_all.copy())


    acceptance_rate = accept / trials

    return snapshots, np.array(U1_trace), acceptance_rate


def run_mc(
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
    n_burnin=0,
    seed=0,
    include_log=True,
    include_cosh=False,
    E0_over_T=0.0,
    xi_F=1.0,
    L=1.0,
):
    """
    Extended Monte Carlo driver supporting the full pair interaction

        beta*U = kappa*U1 + U2

    where U1 is the existing logarithmic Ewald energy (unchanged, see
    run_mc1 / Ewald_Method_pbc.py) and U2 is the short-range "cosh"
    interaction from Short_Range_Interaction.py, evaluated with the
    minimum-image convention (no Ewald sum, no extra periodic images).

    include_log / include_cosh select which term(s) are active:
      - include_log=True,  include_cosh=False -> reproduces run_mc1 exactly
      - include_log=False, include_cosh=True   -> second term only
      - include_log=True,  include_cosh=True   -> full interaction

    E0_over_T and xi_F are the dimensionless coupling E0/T and the decay
    length xi_F of the second term, both expressed in box units (see
    Short_Range_Interaction.py for unit conventions). L is the box length
    used for the minimum-image convention of the second term; positions
    R_all are fractional coordinates so L=1.0 matches the box convention
    already used throughout this codebase.

    Returns snapshots plus four separate traces sampled every
    sample_every production sweeps:
      - U1_trace:            logarithmic Ewald energy (0 if include_log=False)
      - U2_shifted_trace:    short-range energy used for MC acceptance
                              (0 if include_cosh=False)
      - U2_exact_trace:      absolute short-range energy, i.e.
                              U2_shifted - [N(N-1)/2]*ln(2)
                              (0 if include_cosh=False)
      - beta_U_total_trace:  kappa*U1_current + U2_shifted_current, the
                              configuration-dependent quantity actually
                              driving the Metropolis sampler
    """

    rng = np.random.default_rng(seed)

    q = np.asarray(q, dtype=float)

    if len(q) != N:
        raise ValueError("len(q) must equal N.")

    if not np.isclose(np.sum(q), 0.0):
        raise ValueError("The logarithmic Ewald sum requires charge neutrality: sum(q) = 0.")

    R_all = initialize_positions(N, a, rng)

    n_pairs = N * (N - 1) / 2.0

    if include_log:
        U1_current = ewald_energy1(R_all, q, alpha, n_cut, m_cut)
    else:
        U1_current = 0.0

    if include_cosh:
        U2_current = energy2(R_all, E0_over_T, xi_F, L=L)
    else:
        U2_current = 0.0

    U1_trace = []
    U2_shifted_trace = []
    U2_exact_trace = []
    beta_U_total_trace = []
    snapshots = []

    accept = 0
    trials = 0

    total_sweeps = n_burnin + n_sweeps

    for sweep in range(total_sweeps):
        for _ in range(N):
            i = rng.integers(N)

            R_old_i = R_all[i].copy()
            R_trial_i = (R_old_i + rng.uniform(-step_size, step_size, size=2)) % L

            trials += 1

            if hardcore_check(R_trial_i, R_all, a, exclude_index=i):
                continue

            if include_cosh:
                dU2 = energy2_diff(R_all, i, R_old_i, R_trial_i, E0_over_T, xi_F, L=L)
            else:
                dU2 = 0.0

            if include_log:
                R_all[i] = R_trial_i
                U1_trial = ewald_energy1(R_all, q, alpha, n_cut, m_cut)
                dU1 = U1_trial - U1_current
            else:
                R_all[i] = R_trial_i
                U1_trial = U1_current
                dU1 = 0.0

            dbeta_U = kappa * dU1 + dU2

            if np.isfinite(dbeta_U) and (dbeta_U <= 0.0 or rng.random() < np.exp(-dbeta_U)):
                U1_current = U1_trial
                U2_current = U2_current + dU2
                accept += 1
            else:
                R_all[i] = R_old_i

        if sweep >= n_burnin:
            production_sweep = sweep - n_burnin

            if production_sweep % sample_every == 0:
                U2_exact_current = (
                    energy2_exact_from_shifted(U2_current, N) if include_cosh else 0.0
                )
                beta_U_total_current = kappa * U1_current + U2_current

                U1_trace.append(U1_current)
                U2_shifted_trace.append(U2_current)
                U2_exact_trace.append(U2_exact_current)
                beta_U_total_trace.append(beta_U_total_current)
                snapshots.append(R_all.copy())

    acceptance_rate = accept / trials

    return (
        snapshots,
        np.array(U1_trace),
        np.array(U2_shifted_trace),
        np.array(U2_exact_trace),
        np.array(beta_U_total_trace),
        acceptance_rate,
    )