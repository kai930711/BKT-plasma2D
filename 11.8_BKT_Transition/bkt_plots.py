"""Figures for the 11.8 BKT analysis. Reads Results/states_labelled.csv and Results/bkt_lines.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
RES = HERE / 'Results'
REF = Path('/Users/kai/BKT-plasma2D/9.6_Summary/Transition Temperature data/op_dielectric_julia_intersections.csv')

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

# Phase colours sampled from the reference phase diagram (superfluid / vortex gas / vortex liquid)
PHASE = {
    'diel':   dict(fill='#c7a8f7', text='#6935af', name='dielectric gas (insulator)'),
    'plasma': dict(fill='#d6d6d6', text='#4e4e4e', name='plasma (conductor)'),
    'liquid': dict(fill='#bcdffa', text='#1e4988', name='liquid (droplet)'),
}
RED = '#f60709'

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def load():
    st = pd.read_csv(RES / 'states_labelled.csv')
    li = pd.read_csv(RES / 'bkt_lines.csv')
    return st, li


def fig_cuts(st, li, rhos=(0.001, 0.00631, 0.0251), e0s=(0.0, 0.1, 0.2), save=True):
    """1/eps(k_min) vs T* along fixed (rho*, E0*), with the 4T* line. Hollow = droplet (excluded)."""
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
    h += [Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, label='droplet (excluded)'),
          Line2D([], [], marker='D', ls='', mfc='white', mec=INK, label='T_BKT (crossing)')]
    fig.legend(handles=h, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle('1/ε against the 4T* line, N = 50', color=INK, y=1.02)
    if save:
        fig.savefig(RES / 'fig1_cuts.png')
    return fig


def phase_boundaries(st, li, rho, x):
    """Droplet boundary T_d(x) and top of the dielectric region T_top(x) on a fine E0* grid x.

    T_d: per E0* column, half a cell above the highest droplet state that is connected to the
    bottom of the column (isolated droplet states higher up are ignored).
    T_top: T_BKT where the line passes the criterion; past the last BKT column it closes onto
    T_d halfway to the next E0* point (the dashed, unmeasured end of the red line).
    """
    g = st[np.isclose(st.rho_star, rho)]
    Es = np.sort(g.E0_star.unique())
    Ts = np.sort(g.T_star.unique())
    dT = Ts[1] - Ts[0]
    Td = []
    for e in Es:
        col = g[np.isclose(g.E0_star, e)].sort_values('T_star')
        drop = col.droplet.values
        k = 0
        while k < len(drop) and drop[k]:
            k += 1
        Td.append(Ts[0] - dT / 2 if k == 0 else Ts[k - 1] + dT / 2)
    Td = np.array(Td)
    td = np.interp(x, Es, Td)

    L = li[np.isclose(li.rho_star, rho)].sort_values('E0_star')
    ok = (L.verdict == 'BKT').values
    n = 0                                    # BKT columns contiguous from E0* = 0
    while n < len(ok) and ok[n]:
        n += 1
    top = td.copy()
    e_end = e_mid = None
    if n:
        Eb, Tb = Es[:n], L.T_BKT.values[:n]
        e_end = Eb[-1]
        e_mid = e_end + (Es[n] - e_end) / 2 if n < len(Es) else e_end
        inside = x <= e_end
        top[inside] = np.interp(x[inside], Eb, Tb)
        taper = (x > e_end) & (x <= e_mid)
        if taper.any():
            t_end, t_mid = np.interp(e_end, Eb, Tb), np.interp(e_mid, Es, Td)
            top[taper] = t_end + (x[taper] - e_end) / (e_mid - e_end) * (t_mid - t_end)
    top = np.maximum(top, td)
    return td, top, (n, e_end, e_mid), L.iloc[:n]


def fig_maps(st, li, save=True):
    """Phase regions per rho*: liquid (droplet) / dielectric gas / plasma, with T_BKT in red."""
    rhos = sorted(st.rho_star.unique())
    x = np.linspace(-0.01, 0.51, 1041)
    T0, T1 = 0.045, 0.305
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), sharex=True, sharey=True)
    for ax, rho in zip(axes.flat, rhos):
        td, top, (n, e_end, e_mid), B = phase_boundaries(st, li, rho, x)
        td_c, top_c = np.clip(td, T0, T1), np.clip(top, T0, T1)
        ax.fill_between(x, T0, td_c, color=PHASE['liquid']['fill'], lw=0)
        ax.fill_between(x, td_c, top_c, color=PHASE['diel']['fill'], lw=0)
        ax.fill_between(x, top_c, T1, color=PHASE['plasma']['fill'], lw=0)
        if n:
            ax.plot(B.E0_star, B.T_BKT, '-', color=RED, lw=2.2, solid_capstyle='round', zorder=3)
            if e_mid > e_end:
                xs = np.linspace(e_end, e_mid, 20)
                ax.plot(xs, np.interp(xs, x, top), '--', color=RED, lw=1.6, zorder=3)
        # region labels at the point deepest inside each region
        ys = np.linspace(T0, T1, 261)
        X, Y = np.meshgrid(x, ys)
        TD, TOP = np.interp(X, x, td), np.interp(X, x, top)
        masks = {'liquid': Y < TD, 'diel': (Y >= TD) & (Y < TOP), 'plasma': Y >= np.maximum(TD, TOP)}
        word = {'liquid': 'liquid', 'diel': 'dielectric\ngas', 'plasma': 'plasma'}
        sx, sy = (x[-1] - x[0]) / (len(x) - 1), (T1 - T0) / (len(ys) - 1)
        for k, m in masks.items():
            if m.sum() < 400:
                continue
            # distance (in axis-fraction units) to the region edge or the panel border,
            # so the label sits in the visual middle of the region
            vis = m & (X >= 0) & (X <= 0.5) & (Y >= 0.05) & (Y <= 0.30)
            pad = np.pad(vis, 1, constant_values=False)
            dist = distance_transform_edt(pad, sampling=(sy / 0.25, sx / 0.5))[1:-1, 1:-1]
            if dist.max() < 0.07:
                continue
            i, j = np.unravel_index(np.argmax(dist), dist.shape)
            ax.text(x[j], ys[i], word[k], color=PHASE[k]['text'], ha='center', va='center',
                    fontsize=10, fontweight='medium', zorder=4)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0.05, 0.30)
        ax.grid(False)
        ax.set_title(f'ρ* = {rho:g}', color=INK, fontsize=10)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(True)
    for ax in axes[-1]:
        ax.set_xlabel('E0*')
    for ax in axes[:, 0]:
        ax.set_ylabel('T*')
    leg = axes.flat[-1]
    leg.axis('off')
    hs = [Patch(fc=PHASE[k]['fill'], ec='none', label=PHASE[k]['name']) for k in ('diel', 'plasma', 'liquid')]
    hs += [Line2D([], [], color=RED, lw=2.2, label='T_BKT  (1/ε = 4T*)'),
           Line2D([], [], color=RED, lw=1.6, ls='--', label='end of the BKT line (not measured)')]
    leg.legend(handles=hs, loc='center', fontsize=10)
    fig.suptitle('Phase regions and the BKT line, N = 50', color=INK, y=0.995)
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
