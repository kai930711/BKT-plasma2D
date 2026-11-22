"""Apply one fixed BKT criterion to the 11.8 scan (N = 50).

Input : Results/runs_11p8.csv   (from compute_dielectric.py)
Output: Results/states_labelled.csv  one row per state, with its label
        Results/bkt_lines.csv        one row per (rho*, E0*) line, with its verdict and T_BKT

THE CRITERION
-------------
Order parameter, per state:  d = 1/eps(k_min) - 4 T*,  sigma = block error of 1/eps.
(1/eps = 1 - kappa S_QQ(m^2=1) / (2 pi), S_QQ with no 1/N, L = 1.)

Step 1. A state is VALID only if it is homogeneous: in none of its 100 snapshots does one
        cluster (bond r < 1.5 sigma) hold DROPLET_FRAC = half of the particles or more
        (no droplet, not even part-time).
        (A half-run drift check exists but is off by default, DRIFT_Z = None: it only ever
        affected 3 lines and never moved T_BKT.)
Step 2. Each valid state gets a label
        D (dielectric, pairs bound)   if d > +Z sigma
        C (conducting, free charges)  if d < -Z sigma
        M (on the line within noise)  otherwise.
Step 3. A line at fixed (rho*, E0*) HAS A BKT TRANSITION if, going up in T* over its valid
        states, the D/C labels form one block of D followed by one block of C
        (D...D C...C, M allowed anywhere), and no invalid state sits inside the bracket
        [T of last D, T of first C].
        T_BKT = zero of a weighted straight-line fit of d(T*) over the states in the bracket.
Verdicts when Step 3 fails:
        all_droplet    : no valid state at all on the line
        no_dielectric  : valid states exist but none is D (the gas condenses before it can bind pairs)
        no_conductor   : no valid C state up to T* = 0.30
        droplet_in_bracket / non_monotone : ambiguous, not counted as BKT
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Z = 2.0             # significance of D / C labels, in sigma
DROPLET_FRAC = 0.5  # largest cluster holding >= half the particles = droplet
DRIFT_Z = None      # off; set e.g. 3.0 to exclude runs whose two halves disagree by > 3 sigma


def label_states(df, z=Z, droplet_frac=DROPLET_FRAC, drift_z=DRIFT_Z):
    df = df.copy()
    df['d'] = df.inv_eps - 4 * df.T_star
    df['zscore'] = df.d / df.inv_eps_sem
    df['droplet'] = df.lc_frac_max >= droplet_frac
    s12 = np.hypot(df.inv_eps_h1_sem, df.inv_eps_h2_sem)
    df['drift'] = False if drift_z is None else (df.inv_eps_h2 - df.inv_eps_h1).abs() > drift_z * s12
    df['valid'] = ~df.droplet & ~df.drift
    lab = np.where(df.zscore > z, 'D', np.where(df.zscore < -z, 'C', 'M'))
    df['label'] = np.where(df.valid, lab, np.where(df.droplet, 'drop', 'drift'))
    return df


def classify_line(g):
    g = g.sort_values('T_star')
    v = g[g.valid]
    out = dict(n_valid=len(v), n_droplet=int(g.droplet.sum()), n_drift=int((~g.droplet & g.drift).sum()),
               T_BKT=np.nan, T_BKT_err=np.nan, T_lo=np.nan, T_hi=np.nan)
    sig = v[v.label.isin(['D', 'C'])]
    seq = ''.join(sig.label)
    if len(v) == 0:
        return dict(out, verdict='all_droplet')
    if 'D' not in seq:
        return dict(out, verdict='no_dielectric')
    if 'C' not in seq:
        return dict(out, verdict='no_conductor')
    if 'CD' in seq:                       # a C below a D: not one D block then one C block
        return dict(out, verdict='non_monotone')
    T_lo = sig[sig.label == 'D'].T_star.max()
    T_hi = sig[sig.label == 'C'].T_star.min()
    inside = g[(g.T_star >= T_lo) & (g.T_star <= T_hi)]
    out.update(T_lo=T_lo, T_hi=T_hi)
    if not inside.valid.all():
        return dict(out, verdict='droplet_in_bracket')
    # weighted line d = c0 + c1 T through the bracket; T_BKT = -c0/c1
    w = 1 / inside.inv_eps_sem.values ** 2
    T, d = inside.T_star.values, inside.d.values
    A = np.vstack([np.ones_like(T), T]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    c0, c1 = cov @ (A.T @ (w * d))
    Tc = -c0 / c1
    grad = np.array([-1 / c1, c0 / c1 ** 2])
    err = float(np.sqrt(grad @ cov @ grad))
    return dict(out, verdict='BKT', T_BKT=Tc, T_BKT_err=err)


def classify_all(states):
    rows = []
    for (rho, e0), g in states.groupby(['rho_star', 'E0_star']):
        rows.append(dict(rho_star=rho, E0_star=e0, **classify_line(g)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    runs = pd.read_csv(HERE / 'Results' / 'runs_11p8.csv')
    states = label_states(runs)
    lines = classify_all(states)
    states.to_csv(HERE / 'Results' / 'states_labelled.csv', index=False)
    lines.to_csv(HERE / 'Results' / 'bkt_lines.csv', index=False)
    print(states.label.value_counts().to_string())
    print(lines.verdict.value_counts().to_string())
