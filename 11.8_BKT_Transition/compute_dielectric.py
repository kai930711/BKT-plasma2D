"""Per-run 1/eps, S_QQ, S_NN and droplet diagnostics for the 11.8 scan.

Convention (same as 9.21): S_QQ, S_NN carry NO 1/N; L = 1; kappa = 1/T*.
    1/eps(k) = 1 - kappa * S_QQ(m^2) / (2*pi*m^2),   k = 2*pi*sqrt(m^2)
The BKT test uses the smallest shell, m^2 = 1.

Writes Results/runs_11p8.csv. Run with /opt/anaconda3/bin/python.
"""
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from read_scan_v2 import list_runs, load_run, SHELL_M2

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/kai/Downloads/11.8_data/runs')
OUT = HERE / 'Results' / 'runs_11p8.csv'
N_BLOCKS = 20
CLUSTER_CUT = 1.5     # in units of sigma (= a); same as the code's cluster_cutoff


def block_sem(x, nb):
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(axis=1)
    return b.std(ddof=1) / np.sqrt(nb)


def largest_cluster(X, a):
    """Largest-cluster size per frame, PBC minimum image, bond if r < CLUSTER_CUT*a."""
    n = X.shape[1]
    iu = np.triu_indices(n, 1)
    out = np.empty(len(X), dtype=int)
    for f, R in enumerate(X):
        d = R[:, None, :] - R[None, :, :]
        d -= np.rint(d)
        r2 = (d ** 2).sum(-1)[iu]
        bond = r2 < (CLUSTER_CUT * a) ** 2
        g = coo_matrix((np.ones(bond.sum()), (iu[0][bond], iu[1][bond])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        out[f] = np.bincount(lab).max()
    return out


def one(row):
    d = load_run(row)
    S, X = d['series'], d['coordinates']
    N, kappa, a = int(row['N']), float(row['kappa']), float(row['a'])
    Q2 = S[:, 13:23]          # |Q_m|^2, shells 1..10
    R2 = S[:, 3:13]           # |rho_m|^2
    ie_t = 1.0 - kappa * Q2[:, 0] / (2 * np.pi * SHELL_M2[0])   # 1/eps(k_min) per sample
    ns = len(ie_t)
    h = ns // 2
    sem = block_sem(ie_t, N_BLOCKS)
    naive = ie_t.std(ddof=1) / np.sqrt(ns)
    tau = 0.5 * (sem / naive) ** 2 if naive > 0 else np.nan     # integrated autocorr time, in samples
    lc = largest_cluster(X, a)
    fh = len(lc) // 2
    out = dict(
        state_id=int(row['state_id']), run_id=row['run_id'], N=N, rho_star=float(row['rho_star']),
        T_star=float(row['T_star']), E0_star=float(row['E0_star']), kappa=kappa,
        n_sweeps=int(row['n_sweeps']), n_samples=ns,
        acc=float(row['n_accept']) / float(row['n_trials']),
        cl_acc=float(row['n_cluster_accept']) / max(1.0, float(row['n_cluster_trials'])),
        inv_eps=ie_t.mean(), inv_eps_sem=sem, tau_samples=tau, n_eff=ns / (2 * tau) if tau > 0 else np.nan,
        inv_eps_h1=ie_t[:h].mean(), inv_eps_h1_sem=block_sem(ie_t[:h], N_BLOCKS // 2),
        inv_eps_h2=ie_t[h:].mean(), inv_eps_h2_sem=block_sem(ie_t[h:], N_BLOCKS // 2),
        lc_frac=lc.mean() / N, lc_frac_h2=lc[fh:].mean() / N, lc_frac_max=lc.max() / N,
        U1=S[:, 0].mean(), U2=S[:, 1].mean(),
    )
    for i, m2 in enumerate(SHELL_M2):
        out[f'SQQ_m{m2}'] = Q2[:, i].mean()
        out[f'SNN_m{m2}'] = R2[:, i].mean()
        out[f'inv_eps_m{m2}'] = 1.0 - kappa * Q2[:, i].mean() / (2 * np.pi * m2)
    return out


if __name__ == '__main__':
    runs = list_runs(DATA)
    with Pool() as p:
        res = p.map(one, [r for _, r in runs.iterrows()], chunksize=20)
    df = pd.DataFrame(res).sort_values(['rho_star', 'E0_star', 'T_star']).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT}  ({len(df)} runs)')
