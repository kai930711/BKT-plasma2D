"""Small-k structure factors for every state of the 9.21 scan.

Convention (no 1/N):
    S_NN(k) = < |sum_j exp(i k.r_j)|^2 >,   S_QQ(k) = < |sum_j q_j exp(i k.r_j)|^2 >
    1/eps(k) = 1 - kappa * S_QQ / (2 pi m^2)       (unit box, k = 2 pi m, kappa = 1/T*)
Large-k limit of both S is N. A uniform ideal gas has S_NN = N at every k != 0.

Usage (Python 3.11+, NumPy, pandas):
    python compute_structure_factors.py            # all extracted batches under Data/
    python compute_structure_factors.py --n-shells 10 --n-blocks 5

Outputs in Results/:
    structure_factors.csv   one row per (state, shell): means, block SEMs, half-run drift, 1/eps
    S_kmin_series.npz       per-frame S_NN(k_min), S_QQ(k_min) for every state (MC-time monitoring)
"""
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd

from read_scan import load_run

BASE_DIR = Path(__file__).resolve().parent


def shell_modes(n_shells):
    """Half-plane integer vectors m (one of each +/- pair) for the first n_shells |m|^2 values.
    |Q(-m)|^2 = |Q(m)|^2 for real charges, so the shell average is unchanged."""
    radius = int(np.ceil(np.sqrt(4 * n_shells))) + 1
    modes = [(mx, my) for mx in range(0, radius + 1) for my in range(-radius, radius + 1)
             if mx > 0 or (mx == 0 and my > 0)]
    modes = np.array(modes, dtype=int)
    m2 = (modes**2).sum(axis=1)
    shells = np.unique(m2)[:n_shells]
    keep = np.isin(m2, shells)
    return modes[keep], m2[keep], shells


def shell_series(xy, q, modes, m2, shells):
    """Per-frame shell-averaged S_NN and S_QQ, shape (frames, shells). No 1/N."""
    phase = np.exp(2j * np.pi * np.einsum('tnd,md->tnm', xy, modes))
    s_nn_modes = np.abs(phase.sum(axis=1))**2
    s_qq_modes = np.abs(np.einsum('tnm,n->tm', phase, q))**2
    s_nn = np.stack([s_nn_modes[:, m2 == s].mean(axis=1) for s in shells], axis=1)
    s_qq = np.stack([s_qq_modes[:, m2 == s].mean(axis=1) for s in shells], axis=1)
    return s_nn, s_qq


def block_mean_sem(series, n_blocks):
    """Equal contiguous time blocks; SEM = std(block means)/sqrt(B)."""
    n = len(series)
    if n % n_blocks:
        raise ValueError(f'{n} frames not divisible by {n_blocks} blocks')
    blocks = series.reshape(n_blocks, n // n_blocks, -1).mean(axis=1)
    return series.mean(axis=0), blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def find_batches(data_dir):
    batches = sorted(p.parent for p in data_dir.rglob('states.csv'))
    if not batches:
        raise FileNotFoundError(f'No extracted states.csv under {data_dir}')
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=BASE_DIR / 'Data')
    parser.add_argument('--out-dir', type=Path, default=BASE_DIR / 'Results')
    parser.add_argument('--n-shells', type=int, default=10)
    parser.add_argument('--n-blocks', type=int, default=5)
    args = parser.parse_args()

    modes, m2, shells = shell_modes(args.n_shells)
    degeneracy = np.array([2 * np.count_nonzero(m2 == s) for s in shells])
    rows, series_ids, series_nn, series_qq = [], [], [], []
    started = time.time()

    for batch in find_batches(args.data_dir):
        states = pd.read_csv(batch / 'states.csv')
        print(f'{batch.relative_to(args.data_dir)}: {len(states)} states', flush=True)
        for _, st in states.iterrows():
            data = load_run(batch, int(st['state_id']), int(st['replicate_id']))
            xy = np.asarray(data['coordinates'])
            q = np.asarray(data['charges'])
            n = len(q)
            kappa = float(st['kappa'])
            a = float(st['a'])

            s_nn, s_qq = shell_series(xy, q, modes, m2, shells)
            inv_eps = 1.0 - kappa * s_qq / (2 * np.pi * shells)

            nn_mean, nn_sem = block_mean_sem(s_nn, args.n_blocks)
            qq_mean, qq_sem = block_mean_sem(s_qq, args.n_blocks)
            ie_mean, ie_sem = block_mean_sem(inv_eps, args.n_blocks)
            # Drift: first vs second half, each with its own block SEM.
            half = len(s_nn) // 2
            h1, h1_sem = block_mean_sem(s_nn[:half], args.n_blocks)
            h2, h2_sem = block_mean_sem(s_nn[half:2 * half], args.n_blocks)
            drift_z = (h2 - h1) / np.sqrt(h1_sem**2 + h2_sem**2)

            prog = data['progress']
            acceptance = int(prog['n_accept']) / int(prog['n_trials'])
            for j, s in enumerate(shells):
                rows.append(dict(
                    state_id=int(st['state_id']), N=n, rho_star=float(st['rho_star']),
                    T_star=float(st['T_star']), kappa=kappa, E0_star=float(st['E0_star']),
                    a=a, m2=int(s), degeneracy=int(degeneracy[j]),
                    kL=2 * np.pi * np.sqrt(s), k_sigma=2 * np.pi * np.sqrt(s) * a,
                    S_NN=nn_mean[j], S_NN_sem=nn_sem[j],
                    S_QQ=qq_mean[j], S_QQ_sem=qq_sem[j],
                    inv_eps=ie_mean[j], inv_eps_sem=ie_sem[j],
                    S_NN_first_half=h1[j], S_NN_second_half=h2[j], S_NN_drift_z=drift_z[j],
                    n_frames=len(s_nn), acceptance_rate=acceptance))
            series_ids.append(int(st['state_id']))
            series_nn.append(s_nn[:, 0])
            series_qq.append(s_qq[:, 0])
        print(f'  done, {time.time() - started:.0f}s elapsed', flush=True)

    args.out_dir.mkdir(exist_ok=True)
    table = pd.DataFrame(rows).sort_values(['state_id', 'm2'])
    table.to_csv(args.out_dir / 'structure_factors.csv', index=False)
    order = np.argsort(series_ids)
    np.savez_compressed(args.out_dir / 'S_kmin_series.npz',
                        state_id=np.array(series_ids)[order],
                        S_NN=np.array(series_nn)[order], S_QQ=np.array(series_qq)[order])
    print(f'Wrote {len(table)} rows ({table.state_id.nunique()} states) to {args.out_dir}')


if __name__ == '__main__':
    main()
