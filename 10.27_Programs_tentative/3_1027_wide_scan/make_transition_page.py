"""Write Transition_Lines.html: BKT and liquid-gas lines from S_QQ, with a box animation.

Tab 1 uses the 9.21 scan (10 rho*, E0* 0-0.5, T* 0.15-0.35, N=50).
Tab 2 uses the 10.27 wide pilot (rho*=0.05, E0* 0-1, T* 0.05-0.55, hot and cold starts, S_QQ(k) to k sigma ~ 4).

Run with:  /opt/anaconda3/bin/python make_transition_page.py
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

N = 50
FRAMES = [124, 249, 374, 499, 624, 749, 874, 999]   # 0-based frame index, 1000 frames per run
CLUSTER_FACTOR = 5.0
DRIFT_Z = 3.0
T_MAX = 0.35
OUT = HERE / 'Transition_Lines.html'


def batch_map(root):
    m = {}
    for states_csv in sorted(root.rglob('states.csv')):
        for _, st in pd.read_csv(states_csv).iterrows():
            m[(int(st.state_id), int(st.replicate_id))] = states_csv.parent
    return m


def box(batch, state_id, replicate_id=1):
    run = load_run(batch, state_id, replicate_id)
    xy = np.asarray(run['coordinates'])[FRAMES]
    return dict(xy=[np.rint(f.ravel() * 1000).astype(int).tolist() for f in xy],
                q=np.asarray(run['charges']).astype(int).tolist(),
                sweep=run['production_sweep'][FRAMES].tolist())


# ---------------------------------------------------------------- tab 1: 9.21 density scan
table = pd.read_csv(SCAN / 'Results' / 'structure_factors.csv')
d = table[(table.m2 == 1) & (table.T_star <= T_MAX + 1e-9)].copy()
d['clustered'] = d.S_NN > CLUSTER_FACTOR * N
d['drifting'] = d.S_NN_drift_z.abs() > DRIFT_Z
rhos = sorted(d.rho_star.unique().tolist())
E0s = sorted(d.E0_star.unique().tolist())
Ts = sorted(d.T_star.round(4).unique().tolist())
batches = batch_map(SCAN / 'Data')

states = []
for _, r in d.iterrows():
    states.append(dict(
        rho=float(r.rho_star), T=round(float(r.T_star), 4), E0=round(float(r.E0_star), 4), a=round(float(r.a), 6),
        box=box(batches[(int(r.state_id), 1)], int(r.state_id)),
        inv_eps=round(float(r.inv_eps), 5), sem=round(float(r.inv_eps_sem), 5),
        sqq=round(float(r.S_QQ), 5), snn=round(float(r.S_NN) / N, 4),
        drift=round(float(r.S_NN_drift_z), 3), clustered=bool(r.clustered), drifting=bool(r.drifting),
        acc=round(float(r.acceptance_rate), 4)))

bkt, lg = [], []
for rho, g0 in d.groupby('rho_star'):
    for e, g in g0.groupby('E0_star'):                       # BKT: 1/eps = 4T* along T*, at fixed E0*
        g = g.sort_values('T_star'); x = g.T_star.values; f = g.inv_eps.values - 4 * x; cl = g.clustered.values
        if f[0] > 0:
            i = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
            if len(i):
                i = int(i[0]); tb = x[i] + (x[i + 1] - x[i]) * f[i] / (f[i] - f[i + 1])
                bkt.append(dict(rho=float(rho), E0=float(e), T=round(float(tb), 4), homog=bool(not (cl[i] or cl[i + 1]))))
    for T, g in g0.groupby('T_star'):                        # liquid-gas: steepest rise of 1/eps along E0*, at fixed T*
        g = g.sort_values('E0_star'); E = g.E0_star.values; q = g.inv_eps.values
        if q.min() < -0.1 or q.max() > 1.2 or q.max() - q.min() < 0.15:
            continue
        k = int(np.argmax(np.diff(q)))
        lg.append(dict(rho=float(rho), T=round(float(T), 4), lo=float(E[k]), hi=float(E[k + 1]), jump=round(float(np.diff(q)[k]), 4)))

# ---------------------------------------------------------------- tab 2: wide pilot
runs = pd.read_csv(HERE / 'Results_pilot' / 'wide_runs.csv')
z = np.load(HERE / 'Results_pilot' / 'wide_shells.npz')
rho_p = 0.05
ks = 2 * np.pi * np.sqrt(z['shells']) * np.sqrt(rho_p / N)
pbatches = batch_map(HERE / 'Data_pilot')
pilot = []
for i, r in runs.iterrows():
    sid, rep = int(r.state_id), int(r.replicate_id)
    pilot.append(dict(
        init=str(r.init), E0=float(r.E0_star), T=float(r.T_star), a=float(r.a),
        box=box(pbatches[(sid, rep)], sid, rep),
        inv_eps=round(float(z['inv_eps'][i][0]), 4), sem=round(float(z['inv_eps_sem'][i][0]), 4),
        snn=round(float(z['S_NN'][i][0]) / N, 3), acc=round(float(r.acceptance_prod), 4),
        sqq_k=[round(v, 3) for v in (z['S_QQ'][i] / N).tolist()],
        inv_k=[round(v, 3) for v in z['inv_eps'][i].tolist()],
        snn_k=[round(v, 3) for v in (z['S_NN'][i] / N).tolist()]))

data = dict(N=N, rho=rhos, E0=E0s, T=Ts, cluster_factor=CLUSTER_FACTOR, drift_z=DRIFT_Z,
            states=states, bkt=bkt, lg=lg,
            pilot=dict(rho=rho_p, k=[round(v, 4) for v in ks.tolist()],
                       E0=sorted(runs.E0_star.unique().tolist()), T=sorted(runs.T_star.unique().tolist()), runs=pilot))
template = (HERE / 'transition_page_template.html').read_text()
OUT.write_text(template.replace('__DATA__', json.dumps(data, separators=(',', ':'))))
print('wrote', OUT, f'{OUT.stat().st_size / 1e6:.2f} MB;', len(states), 'states,', len(bkt), 'BKT points,', len(lg), 'liquid-gas points,', len(pilot), 'pilot runs')
