"""Metropolis Monte Carlo for a 2D Coulomb plasma with periodic boundaries."""

import numpy as np


def min_image(vec, L):
    """Apply minimum image convention for periodic boundary conditions."""
    return vec - L * np.rint(vec / L)


def pair_distance(r1, r2, L):
    """Distance between two points under PBC."""
    d = min_image(r1 - r2, L)
    return np.hypot(d[0], d[1])


def initialize_positions(N, L, a, rng):
    """Place N particles via rejection sampling, respecting hard-core radius a."""
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


def local_energy(i, pos, charges, N, L, K, xi, a):
    """Energy contribution of particle i with all others."""
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


def total_energy(pos, charges, N, L, K, xi, a):
    """Total energy of the configuration."""
    e = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = pair_distance(pos[i], pos[j], L)
            if r < a:
                return np.inf
            e += -K * charges[i] * charges[j] * np.log(r / xi)
    return e


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


def run_mc(L, N, charges, K, a, xi, step_size, n_sweeps, sample_every, seed=0):
    """Run Metropolis MC simulation.

    Returns (snapshots, E_trace, pair_trace, acceptance_rate).
    """
    rng = np.random.default_rng(seed)
    pos = initialize_positions(N, L, a, rng)

    E_trace = []
    pair_trace = []
    snapshots = []
    accept = 0
    trials = 0

    for sweep in range(n_sweeps):
        for _ in range(N):
            i = rng.integers(N)
            old_pos = pos[i].copy()
            e_old = local_energy(i, pos, charges, N, L, K, xi, a)
            trial_pos = (old_pos + rng.uniform(-step_size, step_size, size=2)) % L
            pos[i] = trial_pos
            e_new = local_energy(i, pos, charges, N, L, K, xi, a)
            dE = e_new - e_old
            trials += 1
            if np.isfinite(dE) and (dE <= 0 or rng.random() < np.exp(-dE)):
                accept += 1
            else:
                pos[i] = old_pos

        if sweep % sample_every == 0:
            E_trace.append(total_energy(pos, charges, N, L, K, xi, a))
            pair_trace.append(nearest_unlike_distance(pos, charges, N, L))
            snapshots.append(pos.copy())

    return snapshots, np.array(E_trace), np.array(pair_trace), accept / trials
