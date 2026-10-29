import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
# ---- pilot (coarse, wide) as faint background
p = pd.read_csv('wide_summary.csv')
E = np.sort(p.E0_star.unique()); T = np.sort(p.T_star.unique())
g = lambda c: p.pivot(index='T_star', columns='E0_star', values=c).loc[T, E].values
pm = np.nanmean([g('inv_eps_kmin_hot'), g('inv_eps_kmin_cold')], axis=0)
# ---- 9.21 scan, rho*=0.05, fine in T*
t = pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv')
d = t[(t.m2 == 1) & np.isclose(t.rho_star, 0.05) & (t.T_star <= 0.31 + 1e-9)]
rows = []
for e, s in d.groupby('E0_star'):
    s = s.sort_values('T_star'); x = s.T_star.values; y = s.inv_eps.values; se = s.inv_eps_sem.values
    # BKT: first sign change of 1/eps - 4T*  (T_BKT)
    f = y - 4 * x; tb = np.nan
    if f[0] > 0:
        i = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
        if len(i): i = i[0]; tb = x[i] + (x[i+1]-x[i]) * f[i] / (f[i]-f[i+1])
    # steepest S_QQ drop; keep if > 3 sigma
    dy = np.diff(y); k = np.argmin(dy); sig = np.hypot(se[k], se[k+1])
    ok = (-dy[k] > 3 * sig)
    rows.append((e, tb, x[k], x[k+1], -dy[k], sig, ok))
r = pd.DataFrame(rows, columns=['E0','T_BKT','lo','hi','drop','sig','sig3'])
print(r.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.2, 5.6))
dE, dT = np.diff(E).mean(), np.diff(T).mean()
im = ax.pcolormesh(np.append(E-dE/2, E[-1]+dE/2), np.append(T-dT/2, T[-1]+dT/2), pm, cmap='viridis', vmin=0, vmax=1, alpha=.55)
fig.colorbar(im, ax=ax).set_label('1/ε(k_min)  [pilot, coarse background]')
# S_QQ drop brackets from 9.21
q = r[r.sig3]
ax.vlines(q.E0, q.lo, q.hi, color='k', lw=6, zorder=4)
ax.plot(q.E0, (q.lo+q.hi)/2, 'k--', lw=1.2, zorder=4, label='S_QQ steepest drop (9.21, ΔT*=0.02, >3σ)')
# pilot bracket at E0*=0.5
ax.vlines([0.5], [0.45], [0.55], color='k', lw=6, alpha=.45, zorder=3)
ax.plot([0.5], [0.5], 'ks', mfc='none', ms=8, label='S_QQ drop, pilot (ΔT*=0.1)')
# BKT line
b = r.dropna(subset=['T_BKT'])
ax.plot(b.E0, b.T_BKT, '-D', color='red', lw=2, ms=6, zorder=5, label='BKT: 1/ε = 4T* (10.21 method)')
ax.annotate('E0*=0, 0.05: 1/ε already < 4T* at T*=0.15\n(T_BKT below the grid)', xy=(0.0, 0.15), xytext=(0.04, 0.07),
            fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlim(-0.03, 1.03); ax.set_ylim(0.05, 0.6)
ax.set_xlabel('E0*'); ax.set_ylabel('T*'); ax.set_title('BKT line stacked on the S_QQ discontinuity (N=50, ρ*=0.05)')
ax.legend(loc='upper right', fontsize=8, framealpha=.92)
fig.tight_layout(); fig.savefig('stack_BKT_on_SQQ.png', dpi=150)
