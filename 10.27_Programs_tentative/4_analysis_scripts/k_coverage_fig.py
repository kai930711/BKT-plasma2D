import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import j1
d = pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv'); N=50
def curve(rho,E0,T):
    g = d[(np.isclose(d.rho_star,rho))&(np.isclose(d.E0_star,E0))&(np.isclose(d.T_star,T))].sort_values('m2')
    return g
fig, ax = plt.subplots(1,3, figsize=(17,5), gridspec_kw=dict(width_ratios=[1.05,1.2,1.2]))

# --- A: k-coverage per density ---
rhos = sorted(d.rho_star.unique())
for i,rho in enumerate(rhos):
    a = np.sqrt(rho/N); kmin=2*np.pi*a; kmax=2*np.pi*np.sqrt(17)*a
    ax[0].plot([kmin,kmax],[i,i],'-o',color='C0',lw=3,ms=5)
ax[0].axvspan(0.5,1.2,color='orange',alpha=.35,label='droplet structure\n(1st minimum k$\\sigma$≈0.8-0.85)')
ax[0].axvspan(0.02,0.5,color='gray',alpha=.12,label='Guinier plateau (flat: no shape info)')
ax[0].set_xscale('log'); ax[0].set_yticks(range(len(rhos))); ax[0].set_yticklabels([f'{r:g}' for r in rhos])
ax[0].set_xlabel(r'$k\sigma$'); ax[0].set_ylabel(r'$\rho^*$'); ax[0].set_xlim(0.03,3)
ax[0].set_title('A. k-range covered by the 10 stored shells ($m^2\\leq17$)'); ax[0].legend(loc='lower right',fontsize=8)

# --- B, C: real curves, droplet state vs homogeneous state ---
T=0.21; R=4.8
kk = np.linspace(0.02,2,400); F = (2*j1(kk*R)/(kk*R))**2
for rho,c in [(0.2,'C3'),(0.001,'C0')]:
    for E0,ls,lab in [(0.5,'-','E0*=0.5'),(0.0,'--','E0*=0')]:
        g = curve(rho,E0,T); 
        ax[1].plot(g.k_sigma,g.S_NN/N,ls+'o',color=c,ms=5,label=f'rho*={rho}, {lab}')
        ax[2].plot(g.k_sigma,g.inv_eps,ls+'o',color=c,ms=5,label=f'rho*={rho}, {lab}')
g = curve(0.2,0.5,T); plat = g.S_NN.iloc[0]/N
ax[1].plot(kk,plat*F+ 1,'k:',lw=1.5,label=f'disk form factor, R={R}$\\sigma$ (schematic)')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'$k\sigma$'); ax[1].set_ylabel(r'$S_{NN}(k)/N$')
ax[1].axhline(1,color='gray',lw=.8); ax[1].axvspan(0.5,1.2,color='orange',alpha=.2)
ax[1].set_title(f'B. S_NN at T*={T}: droplet minimum only visible at high rho*'); ax[1].legend(fontsize=7,loc='lower left')
ax[2].set_xscale('log'); ax[2].set_xlabel(r'$k\sigma$'); ax[2].set_ylabel(r'$1/\epsilon(k)=1-\kappa S_{QQ}/(2\pi m^2)$')
ax[2].axvspan(0.5,1.2,color='orange',alpha=.2); ax[2].axhline(0,color='gray',lw=.8)
ax[2].set_title(f'C. S_QQ-derived 1/eps at T*={T}: same k-window'); ax[2].legend(fontsize=7,loc='lower left')
plt.tight_layout(); plt.savefig('k_coverage.png',dpi=140)
print(curve(0.2,0.5,T)[['m2','k_sigma','S_NN','inv_eps']].assign(SNN_over_N=lambda x:x.S_NN/N).round(3).to_string(index=False))
print(curve(0.001,0.5,T)[['m2','k_sigma','S_NN','inv_eps']].assign(SNN_over_N=lambda x:x.S_NN/N).round(3).to_string(index=False))
