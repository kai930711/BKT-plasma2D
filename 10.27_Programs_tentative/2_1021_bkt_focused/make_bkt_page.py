"""Write BKT_phase_diagram.html: interactive (E0*, T*) dot map per rho*, from the 9.21 scan.

Same selection and classification as BKT_Focused_N50.ipynb (run that first; it writes
Results/bkt_focused_N50.csv). Run with:  /opt/anaconda3/bin/python make_bkt_page.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCAN = HERE.parent / '9.21_Density Correlation'
sys.path.insert(0, str(SCAN))
from read_scan import load_run

SOURCE = SCAN / 'Results' / 'structure_factors.csv'
FRAMES = [249, 499, 749, 999]   # snapshots shown in the box view (0-based frame index, 1000 frames per run)
BKT = HERE / 'Results' / 'bkt_focused_N50.csv'
OUT = HERE / 'BKT_phase_diagram.html'

RHO_SEL = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
E0_SEL = [0.0, 0.1, 0.2, 0.25, 0.3]
T_MAX = 0.31
CLUSTER_FACTOR = 5.0   # clustered when S_NN(k_min) > CLUSTER_FACTOR * N
DRIFT_Z = 3.0          # drifting when first/second-half S_NN(k_min) differ by > DRIFT_Z sigma


def isin(col, values):
    return np.isclose(col.to_numpy()[:, None], values).any(axis=1)


table = pd.read_csv(SOURCE)
d = table[(table.m2 == 1) & (table.T_star <= T_MAX + 1e-9)
          & isin(table.rho_star, RHO_SEL) & isin(table.E0_star, E0_SEL)].copy()
N = int(d.N.iloc[0])
d['clustered'] = d.S_NN > CLUSTER_FACTOR * N
d['drifting'] = d.S_NN_drift_z.abs() > DRIFT_Z
d['diff'] = d.inv_eps - 4 * d.T_star
d['phase'] = np.where((d['diff']).abs() < 2 * d.inv_eps_sem, 'near',
                      np.where(d['diff'] > 0, 'dielectric', 'conductor'))

batches = {}
for states_csv in sorted((SCAN / 'Data').rglob('states.csv')):
    for _, st in pd.read_csv(states_csv).iterrows():
        batches[int(st.state_id)] = states_csv.parent


def snapshots(state_id):
    """Selected frames as integers x*10000 (x then y per particle), plus charges and sweep numbers."""
    run = load_run(batches[state_id], state_id)
    xy = np.asarray(run['coordinates'])[FRAMES]
    return dict(xy=[np.rint(f.ravel() * 10000).astype(int).tolist() for f in xy],
                q=np.asarray(run['charges']).astype(int).tolist(),
                sweep=run['production_sweep'][FRAMES].tolist())


states = [dict(rho=r.rho_star, T=round(r.T_star, 4), E0=round(r.E0_star, 4), a=round(r.a, 6),
               box=snapshots(int(r.state_id)),
               inv_eps=round(r.inv_eps, 5), sem=round(r.inv_eps_sem, 5), diff=round(r['diff'], 5),
               sqq=round(r.S_QQ, 5), sqq_sem=round(r.S_QQ_sem, 5), snn=round(r.S_NN / N, 4),
               drift=round(r.S_NN_drift_z, 3), clustered=bool(r.clustered), drifting=bool(r.drifting),
               acc=round(r.acceptance_rate, 4), phase=r.phase)
          for _, r in d.iterrows()]

columns = []
for _, b in pd.read_csv(BKT).iterrows():
    why = []
    if b.status == 'crossing':
        s = d[np.isclose(d.rho_star, b.rho_star) & np.isclose(d.E0_star, b.E0_star)].sort_values('T_star')
        x = s.T_star.to_numpy()
        i = int(np.searchsorted(x, b.T_BKT, side='right')) - 1
        pair = s.iloc[[i, min(i + 1, len(s) - 1)]]
        if pair.clustered.any(): why.append('clustered')
        if pair.drifting.any(): why.append('drifting')
        if b.n_crossings > 1: why.append('several crossings')
    f = lambda v: None if pd.isna(v) else round(float(v), 5)
    columns.append(dict(rho=b.rho_star, E0=b.E0_star, T_BKT=f(b.T_BKT), low=f(b.T_BKT_low),
                        high=f(b.T_BKT_high), status=b.status, reliable=bool(b.reliable), why=why))

data = dict(N=N, rho=RHO_SEL, E0=E0_SEL, T=sorted(d.T_star.round(4).unique().tolist()),
            cluster_factor=CLUSTER_FACTOR, drift_z=DRIFT_Z, states=states, columns=columns)
template = (HERE / 'bkt_page_template.html').read_text()
OUT.write_text(template.replace('__DATA__', json.dumps(data)))
print('wrote', OUT)
