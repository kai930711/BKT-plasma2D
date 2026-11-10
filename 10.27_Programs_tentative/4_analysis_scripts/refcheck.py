import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
N=50; rho=0.05
r=pd.read_csv('wide_runs.csv'); z=np.load('wide_shells.npz')
m2=z['shells']; ks=2*np.pi*np.sqrt(m2)*np.sqrt(rho/N)      # k*sigma, a = sqrt(rho/N)
def idx(E,T,init='hot'):
    s=r[np.isclose(r.E0_star,E)&np.isclose(r.T_star,T)&(r.init==init)]
    return int(s.index[0]) if len(s) else None
cases=[('plasma-like  E0*=0, T*=0.35',0.0,0.35,'C0'),('near-BKT gas  E0*=0, T*=0.15',0.0,0.15,'C2'),
       ('condensed  E0*=0.5, T*=0.25',0.5,0.25,'C3'),('condensed  E0*=1.0, T*=0.25',1.0,0.25,'C1')]
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
for lab,E,T,c in cases:
    i=idx(E,T)
    if i is None: print('missing',lab); continue
    sq=z['S_QQ'][i]; ie=z['inv_eps'][i]; sn=z['S_NN'][i]
    ax[0].plot(ks,sq/N,'-',color=c,label=lab); ax[1].plot(ks,ie,'-',color=c); ax[2].plot(ks,sn/N,'-',color=c)
    j=np.argmin(np.abs(ks-3.4)); print(f'{lab}: 1/eps(kmin)={ie[0]:.2f}, 1/eps min over k={ie.min():.2f} at k={ks[np.argmin(ie)]:.2f}, S_QQ max={sq.max():.2f} at k={ks[np.argmax(sq)]:.2f}, S_NN/N(kmin)={sn[0]/N:.1f}')
ax[0].axhline(1,color='gray',ls=':',lw=.8); ax[1].axhline(0,color='gray',ls=':',lw=.8)
ax[1].plot(ks,np.where(ks>0,4*0.35*np.ones_like(ks),0),alpha=0)  # keep axes
ax[2].set_yscale('log'); ax[2].set_xscale('log'); ax[2].axhline(1,color='gray',ls=':',lw=.8)
ax[0].set_title('(a) S_QQ(k)/N  (per particle, as in the reference)'); ax[1].set_title('(b) 1/ε(k) = 1 − κ S_QQ/(2π m²)'); ax[2].set_title('(c) S_NN(k)/N')
for a in ax: a.set_xlabel('kσ  (≈ ka in the reference)')
ax[0].legend(fontsize=8); fig.suptitle('Pilot states (hot start, N=50, ρ*=0.05) vs the reference signatures',y=1.0)
fig.tight_layout(); fig.savefig('refcheck_pilot_k.png',dpi=140)
