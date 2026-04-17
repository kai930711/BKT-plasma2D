"""Metropolis Monte Carlo for a 2D Coulomb plasma with periodic boundaries."""

import numpy as np


def min_image(vec, L):
    return vec - L * np.rint(vec / L)


def pair_distance(r1, r2, L):
    d = min_image(r1 - r2, L)
    return np.hypot(d[0], d[1])


def initialize_positions(N, L, a, rng):
    pos = np.empty((N, 2), dtype=float)
    for i in range(N):
        while True:
            trial = rng.uniform(0, L, size=2)
            if i == 0:
                pos[i] = trial
                break
            disp = min_image(trial - pos[:i], L)
            dist = np.sqrt(np.sum(disp**2, axis=1))
            if np.all(dist >= a):
                pos[i] = trial
                break
    return pos


def local_energy_1(i, pos, charges, N, L, K, xi, a):
    ri = pos[i]
    qi = charges[i]
    e = 0.0
    for j in range(N):
        if j == i:
            continue
        r = pair_distance(ri, pos[j], L)
        if r < a:
            return np.inf
        e += -K * qi * charges[j] * np.log(r / xi)
    return e


def total_energy_1(pos, charges, N, L, K, xi, a):
    e = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = pair_distance(pos[i], pos[j], L)
            if r < a:
                return np.inf
            e += -K * charges[i] * charges[j] * np.log(r / xi)
    return e


def local_energy_2(i, pos, charges, N, L, T, E0, xiF, a):
    ri = pos[i]
    e = 0.0
    for j in range(N):
        if j == i:
            continue
        r = pair_distance(ri, pos[j], L)
        if r < a:
            return np.inf

        x = (E0 / T) * np.exp(-r / xiF)
        e += -T * np.logaddexp(x, -x)   # = -T * ln(2 cosh(x))

    return e


def total_energy_2(pos, charges, N, L, T, E0, xiF, a):
    e = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = pair_distance(pos[i], pos[j], L)
            if r < a:
                return np.inf

            x = (E0 / T) * np.exp(-r / xiF)
            e += -T * np.logaddexp(x, -x)

    return e


def local_energy_full(i, pos, charges, N, L, K, T, E0, xi, xiF, a):
    return (
        local_energy_1(i, pos, charges, N, L, K, xi, a)
        + local_energy_2(i, pos, charges, N, L, T, E0, xiF, a)
    )


def total_energy_full(pos, charges, N, L, K, T, E0, xi, xiF, a):
    return (
        total_energy_1(pos, charges, N, L, K, xi, a)
        + total_energy_2(pos, charges, N, L, T, E0, xiF, a)
    )


def nearest_unlike_distance(pos, charges, N, L):
    """Mean nearest opposite-charge distance."""
    dlist = []
    for i in range(N):
        qi = charges[i]
        best = np.inf
        for j in range(N):
            if charges[j] != -qi:
                continue
            r = pair_distance(pos[i], pos[j], L)
            if r < best:
                best = r
        dlist.append(best)
    return np.mean(dlist)


def run_mc(L, N, charges, K, T, E0, a, xi, xiF,
           step_size, n_sweeps, sample_every,
           target_term="first", seed=0):
    rng = np.random.default_rng(seed)
    pos = initialize_positions(N, L, a, rng)

    E_trace = []
    pair_trace = []
    snapshots = []
    accept = 0
    trials = 0

    if target_term == "first":
        local_energy = lambda i, pos: local_energy_1(i, pos, charges, N, L, K, xi, a)
        total_energy = lambda pos: total_energy_1(pos, charges, N, L, K, xi, a)
    elif target_term == "second":
        local_energy = lambda i, pos: local_energy_2(i, pos, charges, N, L, T, E0, xiF, a)
        total_energy = lambda pos: total_energy_2(pos, charges, N, L, T, E0, xiF, a)
    elif target_term == "full":
        local_energy = lambda i, pos: local_energy_full(i, pos, charges, N, L, K, T, E0, xi, xiF, a)
        total_energy = lambda pos: total_energy_full(pos, charges, N, L, K, T, E0, xi, xiF, a)
    else:
        raise ValueError("target_term must be 'first', 'second', or 'full'")

    for sweep in range(n_sweeps):
        for _ in range(N):
            i = rng.integers(N)
            old_pos = pos[i].copy()

            e_old = local_energy(i, pos)

            trial_pos = (old_pos + rng.uniform(-step_size, step_size, size=2)) % L
            pos[i] = trial_pos

            e_new = local_energy(i, pos)
            dE = e_new - e_old
            trials += 1

            if np.isfinite(dE) and (dE <= 0 or rng.random() < np.exp(-dE / T)):
                accept += 1
            else:
                pos[i] = old_pos

        if sweep % sample_every == 0:
            E_trace.append(total_energy(pos))
            pair_trace.append(nearest_unlike_distance(pos, charges, N, L))
            snapshots.append(pos.copy())

    return snapshots, np.array(E_trace), np.array(pair_trace), accept / trials