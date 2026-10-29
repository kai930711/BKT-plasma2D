import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = pd.read_csv('wide_summary.csv')
E = np.sort(d.E0_star.unique()); T = np.sort(d.T_star.unique())
def grid(c): return d.pivot(index='T_star', columns='E0_star', values=c).loc[T, E].values
hot, cold = grid('inv_eps_kmin_hot'), grid('inv_eps_kmin_cold'); hyst = grid('hysteresis').astype(bool)
mean = np.nanmean([hot, cold], axis=0)

fig, ax = plt.subplots(figsize=(6.6, 5.2))
dE, dT = np.diff(E).mean(), np.diff(T).mean()
im = ax.pcolormesh(np.append(E-dE/2, E[-1]+dE/2), np.append(T-dT/2, T[-1]+dT/2), mean,
                   cmap='viridis', vmin=0, vmax=1, shading='flat')
cs = ax.contour(E, T, mean, levels=[0.5], colors='white', linewidths=2.5)
ax.clabel(cs, fmt={0.5: '1/ε = 0.5'}, fontsize=9)
ix, iy = np.where(hyst)
ax.scatter(E[iy], T[ix], marker='x', c='red', s=60, lw=2, label='hot ≠ cold start (not equilibrated / coexistence)')
ax.set_xlabel('E0*'); ax.set_ylabel('T*')
ax.set_title('S_QQ-only map: 1/ε(k_min), N=50, ρ*=0.05')
cb = fig.colorbar(im, ax=ax); cb.set_label('1/ε(k_min)  (≈0: conducting plasma,  ≈1: no screening)')
ax.legend(loc='upper left', fontsize=8, framealpha=.9)
ax.text(0.0, 0.52, 'plasma\n(screening)', color='w', fontsize=10, ha='left')
ax.text(0.75, 0.30, 'droplet\n(no screening)', color='k', fontsize=10)
fig.tight_layout(); fig.savefig('pilot_phase_map.png', dpi=150)
print(np.round(mean,2))
