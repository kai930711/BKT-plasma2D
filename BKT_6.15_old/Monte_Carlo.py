import numpy as np
from Setup import hardcore_check
from Ewald_Method_pbc import ewald_energy1

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