import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = pd.read_csv('wide_summary.csv')
E = np.sort(d.E0_star.unique()); T = np.sort(d.T_star.unique())
def grid(c): return d.pivot(index='T_star', columns='E0_star', values=c).loc[T, E].values
mean = np.nanmean([grid('inv_eps_kmin_hot'), grid('inv_eps_kmin_cold')], axis=0)
hyst = grid('hysteresis').astype(bool)

fig, ax = plt.subplots(figsize=(6.6, 5.2))
dE, dT = np.diff(E).mean(), np.diff(T).mean()
im = ax.pcolormesh(np.append(E-dE/2, E[-1]+dE/2), np.append(T-dT/2, T[-1]+dT/2), mean, cmap='viridis', vmin=0, vmax=1)
# steepest drop of 1/eps along T*, ignoring the unequilibrated T*=0.05 row; require a drop > 0.15
for j, e in enumerate(E):
    col = mean[1:, j]; Tc = T[1:]
    drops = col[:-1] - col[1:]
    k = np.argmax(-drops)            # T increasing: 1/eps falls => col[k+1]-col[k] most negative
    k = np.argmin(np.diff(col))
    if -np.diff(col)[k] > 0.15:
        lo, hi = Tc[k], Tc[k+1]
        ax.plot([e, e], [lo, hi], color='white', lw=5, solid_capstyle='butt')
        ax.plot([e, e], [lo, hi], color='k', lw=1.5, solid_capstyle='butt')
        print(f'E0*={e}: steepest drop between T*={lo} and {hi} (Δ={-np.diff(col)[k]:.2f})')
    else:
        print(f'E0*={e}: no drop >0.15 in T*=0.15-0.55')
ix, iy = np.where(hyst)
ax.scatter(E[iy], T[ix], marker='x', c='red', s=55, lw=2)
ax.plot([], [], color='k', lw=1.5, label='steepest drop of 1/ε along T* (bracket)')
ax.scatter([], [], marker='x', c='red', label='hot ≠ cold: unreliable')
ax.set_xlabel('E0*'); ax.set_ylabel('T*'); ax.set_title('S_QQ-only map, N=50, ρ*=0.05')
cb = fig.colorbar(im, ax=ax); cb.set_label('1/ε(k_min)')
ax.legend(loc='upper left', fontsize=8, framealpha=.9)
fig.tight_layout(); fig.savefig('pilot_phase_map.png', dpi=150)
