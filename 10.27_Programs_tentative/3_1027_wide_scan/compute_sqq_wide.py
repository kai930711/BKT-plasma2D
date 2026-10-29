"""Wide-scan structure factors with coexistence diagnostics.

    /opt/anaconda3/bin/python compute_sqq_wide.py --data-dir Data_pilot --out-dir Results_pilot

Convention (same as 9.21): S_NN and S_QQ carry NO 1/N; L = 1; k = 2*pi*m;
1/eps(k) = 1 - kappa * S_QQ / (2*pi*m^2).  Shells go out to --m2-max (default 400, k*sigma up to ~10
for dilute states), so the droplet / screening scale is resolved.

Coexistence diagnostics (per parameter point, from hot-start and cold-start runs):
  hysteresis   hot and cold runs disagree on S_QQ(k_min) or S_NN(k_min) by > --z-hyst sigma
  bimodal      per-frame log S_NN(k_min) or S_QQ(k_min) has Sarle bimodality coefficient > 0.555
  drifting     first-half vs second-half S_NN(k_min) differ by > --z-drift sigma
  frozen       production acceptance < 0.02 (chain did not move)
A point with any flag is NOT a clean single-phase average: read the hot and cold columns separately.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from read_scan import load_run

HERE = Path(__file__).resolve().parent
CLUSTER_FACTOR = 5.0      # clustered when S_NN(k_min) > 5 N, as in the 9.21 notebooks
INIT_NAMES = {'random_nonoverlap': 'hot', 'cold_checkerboard_droplet': 'cold'}


def shell_setup(m2_max):
    r = int(np.floor(np.sqrt(m2_max)))
    modes = np.array([(mx, my) for mx in range(0, r + 1) for my in range(-r, r + 1)
                      if (mx > 0 or (mx == 0 and my > 0)) and mx * mx + my * my <= m2_max], dtype=float)
    m2 = (modes ** 2).sum(axis=1).astype(int)
    shells = np.unique(m2)
    shell_of = np.searchsorted(shells, m2)
    avg = np.zeros((len(modes), len(shells)))
    avg[np.arange(len(modes)), shell_of] = 1.0
    avg /= avg.sum(axis=0, keepdims=True)
    return modes, shells, avg


def shell_series(xy, q, modes, avg, chunk=50):
    """Per-frame shell-averaged S_NN, S_QQ; shapes (frames, shells)."""
    snn, sqq = [], []
    for i in range(0, len(xy), chunk):
        phase = np.exp(2j * np.pi * np.einsum('tnd,md->tnm', xy[i:i + chunk], modes))
        snn.append((np.abs(phase.sum(axis=1)) ** 2) @ avg)
        sqq.append((np.abs(np.einsum('tnm,n->tm', phase, q)) ** 2) @ avg)
    return np.concatenate(snn), np.concatenate(sqq)


def block_mean_sem(x, n_blocks):
    n = (len(x) // n_blocks) * n_blocks
    b = x[:n].reshape(n_blocks, n // n_blocks, *x.shape[1:]).mean(axis=1)
    return x[:n].mean(axis=0), b.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def bimodality_coefficient(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 10 or np.ptp(x) == 0:
        return np.nan
    g, k = stats.skew(x, bias=False), stats.kurtosis(x, bias=False)
    return (g * g + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def find_batches(data_dir):
    batches = sorted(p.parent for p in Path(data_dir).rglob('states.csv'))
    if not batches:
        raise FileNotFoundError(f'No states.csv under {data_dir}')
    return batches


def low_k_intercept(shells, ie, ie_sem, max_m2=5):
    """Weighted linear fit 1/eps = D0 + c*m2 on the lowest shells; returns D0, err."""
    sel = shells <= max_m2
    x, y, s = shells[sel].astype(float), ie[sel], np.maximum(ie_sem[sel], 1e-6)
    A = np.vstack([np.ones_like(x), x]).T / s[:, None]
    cov = np.linalg.inv(A.T @ A)
    beta = cov @ A.T @ (y / s)
    return beta[0], np.sqrt(cov[0, 0])


def process(args):
    modes, shells, avg = shell_setup(args.m2_max)
    runs, shell_rows = [], {}
    for batch in find_batches(args.data_dir):
        states = pd.read_csv(batch / 'states.csv')
        extra = pd.read_csv(batch / 'run_extra.csv').set_index('run_id') if (batch / 'run_extra.csv').exists() else None
        for _, st in states.iterrows():
            try:
                d = load_run(batch, int(st['state_id']), int(st['replicate_id']))
            except ValueError as err:                      # incomplete point: skip, never guess
                print(f'  skip {st["run_id"]} in {batch.name}: {err}')
                continue
            xy, q = np.asarray(d['coordinates']), np.asarray(d['charges'])
            n, kappa, a = len(q), float(st['kappa']), float(st['a'])
            snn, sqq = shell_series(xy, q, modes, avg)
            ie = 1.0 - kappa * sqq / (2 * np.pi * shells)
            nn_m, nn_s = block_mean_sem(snn, args.n_blocks)
            qq_m, qq_s = block_mean_sem(sqq, args.n_blocks)
            ie_m, ie_s = block_mean_sem(ie, args.n_blocks)
            half = len(snn) // 2
            h1, h1s = block_mean_sem(snn[:half, 0], args.n_blocks)
            h2, h2s = block_mean_sem(snn[half:2 * half, 0], args.n_blocks)
            drift_z = float((h2 - h1) / np.hypot(h1s, h2s))
            D0, D0_err = low_k_intercept(shells, ie_m, ie_s)
            ex = extra.loc[st['run_id']] if extra is not None else None
            acc = float(ex['acceptance_prod']) if ex is not None else d['progress']['n_accept'] / d['progress']['n_trials']
            run = dict(
                batch=batch.name, run_id=st['run_id'], replicate_id=int(st['replicate_id']),
                init=INIT_NAMES.get(st['initial_condition'], st['initial_condition']),
                state_id=int(st['state_id']), N=n, rho_star=round(float(st['rho_star']), 6),
                T_star=round(float(st['T_star']), 6), E0_star=round(float(st['E0_star']), 6),
                kappa=kappa, a=a, acceptance_prod=acc,
                step_final=float(ex['step_size_final']) if ex is not None else np.nan,
                S_NN_kmin=nn_m[0], S_NN_kmin_sem=nn_s[0], S_QQ_kmin=qq_m[0], S_QQ_kmin_sem=qq_s[0],
                inv_eps_kmin=ie_m[0], inv_eps_kmin_sem=ie_s[0], inv_eps_D0=D0, inv_eps_D0_err=D0_err,
                S_NN_drift_z=drift_z,
                bimodal_BC_SNN=bimodality_coefficient(np.log(snn[:, 0])),
                bimodal_BC_SQQ=bimodality_coefficient(np.log(np.maximum(sqq[:, 0], 1e-12))))
            runs.append(run)
            shell_rows[len(runs) - 1] = (nn_m, nn_s, qq_m, qq_s, ie_m, ie_s, snn[:, 0], sqq[:, 0])
        print(f'{batch.name}: {len(states)} states read', flush=True)
    return runs, shell_rows, shells


def combine_hot_cold(runs_df, args):
    """One row per (rho, T, E0): hot and cold values side by side + coexistence flags."""
    rows = []
    for (rho, T, E0), g in runs_df.groupby(['rho_star', 'T_star', 'E0_star']):
        row = dict(rho_star=rho, T_star=T, E0_star=E0, N=int(g.N.iloc[0]), kappa=g.kappa.iloc[0], a=g.a.iloc[0])
        for init in ('hot', 'cold'):
            r = g[g.init == init]
            row[f'n_rep_{init}'] = len(r)
            for col in ('S_NN_kmin', 'S_QQ_kmin', 'inv_eps_kmin', 'inv_eps_D0'):
                if len(r) == 0:
                    row[f'{col}_{init}'] = row[f'{col}_{init}_sem'] = np.nan
                    continue
                m = r[col].mean()
                sem_block = np.sqrt((r[f'{col}_sem'] ** 2).mean() / len(r)) if f'{col}_sem' in r else np.nan
                if col == 'inv_eps_D0':
                    sem_block = np.sqrt((r['inv_eps_D0_err'] ** 2).mean() / len(r))
                sem_rep = r[col].std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
                row[f'{col}_{init}'] = m
                row[f'{col}_{init}_sem'] = max(sem_block, sem_rep)
            row[f'acc_{init}'] = r.acceptance_prod.mean() if len(r) else np.nan
            row[f'drift_z_{init}'] = r.S_NN_drift_z.abs().max() if len(r) else np.nan
            row[f'BC_{init}'] = np.nanmax(np.r_[r.bimodal_BC_SNN.values, r.bimodal_BC_SQQ.values]) if len(r) else np.nan
        both = row['n_rep_hot'] > 0 and row['n_rep_cold'] > 0
        def z(col):
            return abs(row[f'{col}_hot'] - row[f'{col}_cold']) / np.hypot(row[f'{col}_hot_sem'], row[f'{col}_cold_sem']) if both else np.nan
        row['z_hot_cold_SQQ'], row['z_hot_cold_SNN'] = z('S_QQ_kmin'), z('S_NN_kmin')
        row['hysteresis'] = bool(both and (row['z_hot_cold_SQQ'] > args.z_hyst or row['z_hot_cold_SNN'] > args.z_hyst))
        row['bimodal'] = bool(np.nanmax([row['BC_hot'], row['BC_cold']]) > 0.555)
        row['drifting'] = bool(np.nanmax([row['drift_z_hot'], row['drift_z_cold']]) > args.z_drift)
        row['frozen'] = bool(np.nanmin([row['acc_hot'], row['acc_cold']]) < 0.02)
        row['coexistence_flag'] = row['hysteresis'] or row['bimodal'] or row['drifting']
        pool = g.S_NN_kmin.mean()
        row['clustered'] = bool(pool > CLUSTER_FACTOR * row['N'])
        row['conductor'] = bool(g.inv_eps_kmin.mean() < 4 * T)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['rho_star', 'T_star', 'E0_star'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=HERE / 'Data')
    ap.add_argument('--out-dir', type=Path, default=HERE / 'Results')
    ap.add_argument('--m2-max', type=int, default=400)
    ap.add_argument('--n-blocks', type=int, default=10)
    ap.add_argument('--z-hyst', type=float, default=4.0)
    ap.add_argument('--z-drift', type=float, default=3.0)
    args = ap.parse_args()
    runs, shell_rows, shells = process(args)
    args.out_dir.mkdir(exist_ok=True)
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(args.out_dir / 'wide_runs.csv', index=False)
    idx = sorted(shell_rows)
    np.savez_compressed(args.out_dir / 'wide_shells.npz', shells=shells, run_index=np.array(idx),
        S_NN=np.array([shell_rows[i][0] for i in idx]), S_NN_sem=np.array([shell_rows[i][1] for i in idx]),
        S_QQ=np.array([shell_rows[i][2] for i in idx]), S_QQ_sem=np.array([shell_rows[i][3] for i in idx]),
        inv_eps=np.array([shell_rows[i][4] for i in idx]), inv_eps_sem=np.array([shell_rows[i][5] for i in idx]),
        S_NN_kmin_frames=np.array([shell_rows[i][6] for i in idx]), S_QQ_kmin_frames=np.array([shell_rows[i][7] for i in idx]))
    summary = combine_hot_cold(runs_df, args)
    summary.to_csv(args.out_dir / 'wide_summary.csv', index=False)
    print(f'Wrote {len(runs_df)} runs, {len(summary)} parameter points, {len(shells)} shells (m2<= {args.m2_max}) to {args.out_dir}')
    print(summary[['coexistence_flag', 'hysteresis', 'bimodal', 'drifting', 'frozen', 'clustered', 'conductor']].sum().to_string())


if __name__ == '__main__':
    main()
