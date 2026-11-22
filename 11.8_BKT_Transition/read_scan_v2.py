"""Reader for the 11.8 scan (metadata schema_version = 2). NumPy + pandas only.

    from read_scan_v2 import list_runs, load_run
    runs = list_runs('/Users/kai/Downloads/11.8_data/runs')   # one row per run
    d = load_run(runs.iloc[0])
    d['series']       # (n_samples, 23): U1, U2, beta_U, |rho_m|^2 shells 1..10, |Q_m|^2 shells 1..10
    d['coordinates']  # (n_frames, N, 2), wrapped unit-box coordinates
    d['charges']      # (N,), +/-1

|rho_m|^2 and |Q_m|^2 are raw (no 1/N), already averaged over the directions of each shell.
"""
from pathlib import Path
import tomllib
import numpy as np
import pandas as pd

N_SERIES = 23
SHELL_M2 = np.array([1, 2, 4, 5, 8, 9, 10, 13, 16, 17])


def list_runs(root):
    rows = []
    for d in sorted(Path(root).glob('job*/job*_N*')):
        meta = tomllib.loads((d / 'metadata.toml').read_text())
        if meta['schema_version'] != 2 or meta['byte_order'] != 'little':
            raise ValueError(f'unexpected schema in {d}')
        if list(meta['shell_m2']) != SHELL_M2.tolist():
            raise ValueError(f'unexpected shells in {d}')
        s = pd.read_csv(d / 'states.csv')
        p = pd.read_csv(d / 'progress.csv')
        p = p[p.status == 'complete']          # only complete rows are valid runs
        m = s.merge(p.drop(columns=['state_id']), on='run_id', how='inner')
        m['dir'] = str(d)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def load_run(row):
    d = Path(row['dir'])
    n, ns, nf = int(row['N']), int(row['n_samples']), int(row['n_frames'])
    so, co, qo = int(row['series_offset_bytes']), int(row['coordinates_offset_bytes']), int(row['charges_offset_bytes'])
    if so + 8 * N_SERIES * ns != int(row['series_end_bytes']) or co + 16 * n * nf != int(row['coordinates_end_bytes']):
        raise ValueError(f"offsets do not match counts for {row['run_id']}")
    series = np.fromfile(d / 'series.f64', dtype='<f8', count=N_SERIES * ns, offset=so).reshape(ns, N_SERIES)
    coords = np.fromfile(d / 'coordinates.f64', dtype='<f8', count=2 * n * nf, offset=co).reshape(nf, 2, n).transpose(0, 2, 1)
    q = np.fromfile(d / 'charges.f64', dtype='<f8', count=n, offset=qo)
    if not np.isin(q, [-1, 1]).all() or q.sum() != 0:
        raise ValueError('invalid charges')
    if not np.allclose(series[:, 2], float(row['kappa']) * series[:, 0] + series[:, 1], rtol=1e-9, atol=1e-8):
        raise ValueError('beta_U != kappa*U1 + U2')
    return dict(series=series, coordinates=coords, charges=q)
