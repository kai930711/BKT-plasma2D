import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
N=50
t=pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv'); t=t[t.m2==1].copy()
t['clustered']=t.S_NN>5*N
rhos=sorted(t.rho_star.unique())

def lg_points(d):
    """per T*: bracket in E0* of the steepest RISE of 1/eps(kmin) along E0*  (S_QQ only)"""
    out=[]
    for T,g in d.groupby('T_star'):
        g=g.sort_values('E0_star'); E=g.E0_star.values; q=g.inv_eps.values
        if q.min()<-0.1 or q.max()>1.2 or q.max()-q.min()<0.15: continue
        k=int(np.argmax(np.diff(q))); out.append((T,E[k],E[k+1],np.diff(q)[k]))
    return pd.DataFrame(out,columns=['T','lo','hi','jump'])
def bkt_points(d):
    out=[]
    for e,g in d.groupby('E0_star'):
        g=g.sort_values('T_star'); x=g.T_star.values; f=g.inv_eps.values-4*x; cl=g.clustered.values
        if f[0]>0:
            i=np.where((f[:-1]>0)&(f[1:]<=0))[0]
            if len(i): i=i[0]; out.append((e,x[i]+(x[i+1]-x[i])*f[i]/(f[i]-f[i+1]),not(cl[i] or cl[i+1])))
    return pd.DataFrame(out,columns=['E0','T','homog'])

# ---------- Fig 1: rho*=0.05 ----------
d=t[np.isclose(t.rho_star,0.05)]; lg=lg_points(d); bk=bkt_points(d)
p=pd.read_csv('wide_summary.csv'); Ep=np.sort(p.E0_star.unique()); Tp=np.sort(p.T_star.unique())
piv=lambda c:p.pivot(index='T_star',columns='E0_star',values=c).loc[Tp,Ep].values
pm=np.nanmean([piv('inv_eps_kmin_hot'),piv('inv_eps_kmin_cold')],axis=0)
ext=[]
for i,T in enumerate(Tp):
    if T<0.4: continue
    q=pm[i]; k=int(np.argmax(np.diff(q)))
    if np.diff(q)[k]>0.15: ext.append((T,Ep[k],Ep[k+1]))
ext=pd.DataFrame(ext,columns=['T','lo','hi'])
fig,ax=plt.subplots(figsize=(6.8,5.4))
mid=(lg.lo+lg.hi)/2
ax.errorbar(mid,lg['T'],xerr=(lg.hi-lg.lo)/2,fmt='o-',color='C1',lw=1.8,ms=5,capsize=3,label='liquid–gas: steepest rise of 1/ε along E0* (S_QQ)')
if len(ext): ax.errorbar((ext.lo+ext.hi)/2,ext['T'],xerr=(ext.hi-ext.lo)/2,fmt='o',mfc='white',color='C1',capsize=3,label='same, pilot grid (ΔE0*=0.25, coarse)')
h=bk[bk.homog]; c=bk[~bk.homog]
ax.plot(bk.E0,bk['T'],':',color='red',lw=1.2)
if len(h): ax.plot(h.E0,h['T'],'D',color='red',ms=7,label='BKT: 1/ε = 4T* (homogeneous state)')
ax.plot(c.E0,c['T'],'D',mfc='white',mec='red',ms=6,label='1/ε = 4T* crossing, but state is a droplet (not BKT)')
ax.annotate('E0*=0: T_BKT below the grid (< 0.15)',xy=(0,0.15),xytext=(0.03,0.08),fontsize=8,color='red',arrowprops=dict(arrowstyle='->',color='red'))
ax.set_xlim(-0.03,1.0); ax.set_ylim(0.05,0.6); ax.set_xlabel('E0*'); ax.set_ylabel('T*')
ax.set_title('Transition lines, N=50, ρ*=0.05'); ax.legend(fontsize=8,loc='lower right',framealpha=.92)
fig.tight_layout(); fig.savefig('transition_lines_rho0.05.png',dpi=150)
print(lg.round(3).to_string(index=False)); print(ext.to_string(index=False)); print(bk.round(3).to_string(index=False))

# ---------- Fig 2: all densities ----------
fig,axs=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
cm=plt.cm.viridis(np.linspace(0,0.95,len(rhos)))
for c_,rho in zip(cm,rhos):
    d=t[np.isclose(t.rho_star,rho)]; lg=lg_points(d); bk=bkt_points(d); bk=bk[bk.homog]
    if len(lg)>=3: axs[1].errorbar((lg.lo+lg.hi)/2,lg['T'],xerr=(lg.hi-lg.lo)/2,fmt='o-',color=c_,ms=4,lw=1.3,capsize=2,label=f'ρ*={rho:g}')
    if len(bk): axs[0].plot(bk.E0,bk['T'],'D-',color=c_,ms=5,lw=1.3,label=f'ρ*={rho:g}')
axs[0].set_title('BKT line: 1/ε = 4T*  (homogeneous states only)'); axs[1].set_title('liquid–gas: steepest rise of 1/ε along E0*')
for a in axs: a.set_xlabel('E0*'); a.legend(fontsize=7,ncol=2)
axs[0].set_ylabel('T*'); axs[0].set_xlim(-0.03,0.55); axs[0].set_ylim(0.14,0.36)
fig.tight_layout(); fig.savefig('transition_lines_all_rho.png',dpi=150)
