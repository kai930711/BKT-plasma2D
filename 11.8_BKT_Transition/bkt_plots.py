"""Figures for the 11.8 BKT analysis. Reads Results/states_labelled.csv and Results/bkt_lines.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES = HERE / 'Results'
REF = Path('/Users/kai/BKT-plasma2D/9.6_Summary/Transition Temperature data/op_dielectric_julia_intersections.csv')

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
STATE_COLORS = {'D': BLUE, 'M': '#c9c8c2', 'C': ORANGE, 'drop': '#52514e', 'drift': '#ffffff'}
STATE_NAMES = {'D': 'D: dielectric, 1/ε > 4T*', 'M': 'M: on the line (within 2σ)', 'C': 'C: conducting, 1/ε < 4T*',
               'drop': 'droplet (cluster ≥ N/2 in some snapshot)', 'drift': 'drifting run (halves disagree)'}

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def load():
    st = pd.read_csv(RES / 'states_labelled.csv')
    li = pd.read_csv(RES / 'bkt_lines.csv')
    return st, li


def fig_cuts(st, li, rhos=(0.001, 0.00631, 0.0251), e0s=(0.0, 0.1, 0.2), save=True):
    """1/eps(k_min) vs T* along fixed (rho*, E0*), with the 4T* line. Hollow = not valid (droplet/drift)."""
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.2 * len(rhos), 3.6), sharey=True)
    T = np.linspace(0.04, 0.31, 2)
    for ax, rho in zip(axes, rhos):
        ax.plot(T, 4 * T, ls='--', color=INK, lw=1.2, label='4T*  (BKT jump)')
        for e0, c in zip(e0s, [BLUE, ORANGE, AQUA]):
            g = st[np.isclose(st.rho_star, rho) & np.isclose(st.E0_star, e0)].sort_values('T_star')
            ax.plot(g.T_star, g.inv_eps, color=c, lw=1.5, zorder=2)
            v = g.valid.values
            ax.errorbar(g.T_star[v], g.inv_eps[v], yerr=g.inv_eps_sem[v], fmt='o', ms=4.5, color=c,
                        mec='white', mew=0.8, elinewidth=1, capsize=0, zorder=3, label=f'E0* = {e0:g}')
            ax.plot(g.T_star[~v], g.inv_eps[~v], 'o', ms=4.5, mfc='white', mec=c, mew=1.2, zorder=3)
            l = li[np.isclose(li.rho_star, rho) & np.isclose(li.E0_star, e0)].iloc[0]
            if l.verdict == 'BKT':
                ax.plot([l.T_BKT], [4 * l.T_BKT], marker='D', ms=7, mfc=c, mec=INK, mew=1, zorder=4)
        ax.set_title(f'ρ* = {rho:g}', color=INK)
        ax.set_xlabel('T*')
        ax.set_xlim(0.04, 0.31)
        ax.set_ylim(-0.1, 1.08)
    axes[0].set_ylabel('1/ε  at k_min = 2π/L')
    h, lab = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, label='droplet or drifting (excluded)'),
          Line2D([], [], marker='D', ls='', mfc='white', mec=INK, label='T_BKT (crossing)')]
    fig.legend(handles=h, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle('1/ε against the 4T* line, N = 50', color=INK, y=1.02)
    if save:
        fig.savefig(RES / 'fig1_cuts.png')
    return fig


def fig_maps(st, li, save=True):
    """Every state labelled D / M / C / droplet / drift, one panel per rho*, with T_BKT on top."""
    order = ['D', 'M', 'C', 'drop', 'drift']
    cmap = ListedColormap([STATE_COLORS[k] for k in order])
    norm = BoundaryNorm(np.arange(len(order) + 1) - 0.5, len(order))
    rhos = sorted(st.rho_star.unique())
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), sharex=True, sharey=True)
    for ax, rho in zip(axes.flat, rhos):
        g = st[np.isclose(st.rho_star, rho)]
        idx = {k: i for i, k in enumerate(order)}
        P = g.pivot(index='T_star', columns='E0_star', values='label').apply(lambda col: col.map(idx))
        E, T = P.columns.values, P.index.values
        dE, dT = E[1] - E[0], T[1] - T[0]
        ax.pcolormesh(np.r_[E - dE / 2, E[-1] + dE / 2], np.r_[T - dT / 2, T[-1] + dT / 2], P.values.astype(float),
                      cmap=cmap, norm=norm, edgecolors='white', linewidth=0.4)
        b = li[np.isclose(li.rho_star, rho) & (li.verdict == 'BKT')].sort_values('E0_star')
        ax.plot(b.E0_star, b.T_BKT, '-', color=INK, lw=1.4)
        ax.plot(b.E0_star, b.T_BKT, 'o', ms=3.5, mfc='white', mec=INK, mew=1)
        ax.set_title(f'ρ* = {rho:g}', color=INK, fontsize=10)
        ax.grid(False)
    for ax in axes[-1]:
        ax.set_xlabel('E0*')
    for ax in axes[:, 0]:
        ax.set_ylabel('T*')
    leg = axes.flat[-1]
    leg.axis('off')
    hs = [Patch(fc=STATE_COLORS[k], ec=INK2 if k == 'drift' else 'none', label=STATE_NAMES[k]) for k in order]
    hs.append(Line2D([], [], color=INK, marker='o', mfc='white', label='T_BKT (lines that pass the criterion)'))
    leg.legend(handles=hs, loc='center', fontsize=9.5)
    fig.suptitle('State labels and the BKT line, N = 50 (cells: ΔT* = 0.01, ΔE0* = 0.02)', color=INK, y=0.995)
    fig.tight_layout()
    if save:
        fig.savefig(RES / 'fig2_state_maps.png')
    return fig


def fig_lines(li, save=True):
    """T_BKT(E0*) per rho*, and T_BKT(rho*) at E0* = 0 against the earlier reference."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rhos = sorted(li.rho_star.unique())
    for rho, c in zip(rhos, SEQ):
        g = li[np.isclose(li.rho_star, rho) & (li.verdict == 'BKT')].sort_values('E0_star')
        if g.empty:
            continue
        a.errorbar(g.E0_star, g.T_BKT, yerr=g.T_BKT_err, fmt='-o', ms=4, color=c, mec='white', mew=0.6,
                   lw=1.6, elinewidth=1, capsize=0)
        a.annotate(f'ρ*={rho:g}', (g.E0_star.iloc[-1], g.T_BKT.iloc[-1]), xytext=(5, 0),
                   textcoords='offset points', va='center', fontsize=8.5, color=INK2)
    a.set_xlabel('E0*')
    a.set_ylabel('T_BKT  (1/ε = 4T* crossing)')
    a.set_xlim(-0.01, 0.43)
    a.set_title('BKT line vs E0*: each curve ends where the gas condenses first', color=INK, fontsize=10)

    g0 = li[np.isclose(li.E0_star, 0) & (li.verdict == 'BKT')].sort_values('rho_star')
    b.errorbar(g0.rho_star, g0.T_BKT, yerr=g0.T_BKT_err, fmt='-o', ms=5, color=BLUE, mec='white', lw=1.6,
               label='11.8 scan, E0* = 0')
    if REF.exists():
        r = pd.read_csv(REF)
        r = r[r.N == 50]
        b.plot(r.rho_star, r.T_cross, 's', ms=7, mfc='white', mec=ORANGE, mew=1.6,
               label='9.6 reference crossings, N = 50')
    b.set_xscale('log')
    b.set_xlabel('ρ*')
    b.set_ylabel('T_BKT')
    b.set_title('Pure Coulomb gas (E0* = 0)', color=INK, fontsize=10)
    b.legend(fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(RES / 'fig3_bkt_lines.png')
    return fig


if __name__ == '__main__':
    st, li = load()
    fig_cuts(st, li); fig_maps(st, li); fig_lines(li)
    print('figures written to', RES)
