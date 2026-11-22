import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
N = 50
t = pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv')
t = t[(t.m2 == 1)].copy(); t['clustered'] = t.S_NN > 5 * N
rhos = sorted(t.rho_star.unique()); Es = np.sort(t.E0_star.unique()); Ts = np.sort(t.T_star.unique())
dE, dT = np.diff(Es).mean(), np.diff(Ts).mean()
fig, axs = plt.subplots(2, 5, figsize=(17, 7.6), sharex=True, sharey=True)
out = []
for ax, rho in zip(axs.ravel(), rhos):
    d = t[np.isclose(t.rho_star, rho)]
    M = d.pivot(index='T_star', columns='E0_star', values='inv_eps').loc[Ts, Es].values
    im = ax.pcolormesh(np.append(Es-dE/2, Es[-1]+dE/2), np.append(Ts-dT/2, Ts[-1]+dT/2), M, cmap='viridis', vmin=0, vmax=1, alpha=.55)
    bx, by, bf, qx, ql, qh, qs = [], [], [], [], [], [], []
    for e in Es:
        s = d[np.isclose(d.E0_star, e)].sort_values('T_star')
        x, y, se, cl = s.T_star.values, s.inv_eps.values, s.inv_eps_sem.values, s.clustered.values
        f = y - 4*x
        if f[0] > 0:
            i = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
            if len(i):
                i = i[0]; tb = x[i] + (x[i+1]-x[i])*f[i]/(f[i]-f[i+1])
                bx.append(e); by.append(tb); bf.append(not (cl[i] or cl[i+1]))
        dy = np.diff(y); k = np.argmin(dy)
        total = y.max() - y.min(); sharp = -dy[k] / total if total > 0 else 0
        if y.min() < -0.1:      # S_QQ(k_min) > 2*pi*T*: not equilibrated / frozen droplet, skip
            continue
        if -dy[k] > 3*np.hypot(se[k], se[k+1]) and -dy[k] > 0.1:
            qx.append(e); ql.append(x[k]); qh.append(x[k+1]); qs.append(sharp >= 0.4)
            out.append((rho, e, x[k], x[k+1], -dy[k], sharp))
    qx, ql, qh, qs = map(np.array, (qx, ql, qh, qs))
    if len(qx):
        ax.vlines(qx[qs], ql[qs], qh[qs], color='k', lw=4, zorder=4)
        ax.vlines(qx[~qs], ql[~qs], qh[~qs], color='0.35', lw=4, alpha=.45, zorder=4)
    bx, by, bf = map(np.array, (bx, by, bf))
    if len(bx):
        ax.plot(bx, by, '-', color='red', lw=1.8, zorder=5)
        ax.plot(bx[bf], by[bf], 'D', color='red', ms=5, zorder=6)
        ax.plot(bx[~bf], by[~bf], 'D', mfc='white', mec='red', ms=5, zorder=6)
    ax.set_title(f'ρ* = {rho:g}', fontsize=11)
for ax in axs[1]: ax.set_xlabel('E0*')
for ax in axs[:, 0]: ax.set_ylabel('T*')
fig.suptitle('BKT line (red) stacked on the S_QQ step estimate (black), N=50, 9.21 scan', fontsize=13)
h = [plt.Line2D([], [], color='k', lw=4), plt.Line2D([], [], color='0.35', lw=4, alpha=.45), plt.Line2D([], [], color='red', marker='D', lw=1.8),
     plt.Line2D([], [], color='red', marker='D', mfc='white', lw=0)]
fig.legend(h, ['step-like: steepest 0.02-step holds ≥40% of the total fall', 'gradual: steepest step, <40% of the fall', 'BKT: 1/ε = 4T*, homogeneous state', 'BKT crossing inside a clustered (droplet) state: not a BKT point'],
           loc='lower center', ncol=4, fontsize=9, frameon=False)
fig.subplots_adjust(bottom=0.11, top=0.9, left=0.05, right=0.92, wspace=0.08, hspace=0.18)
cax = fig.add_axes([0.94, 0.2, 0.012, 0.6]); fig.colorbar(im, cax=cax).set_label('1/ε(k_min)')
fig.savefig('stack_BKT_on_SQQ_all_rho.png', dpi=140)
o = pd.DataFrame(out, columns=['rho','E0','lo','hi','drop','sharp']); print(o.round(3).to_string(index=False))
