import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
r=pd.read_csv('classify_R.csv')
E=np.sort(r.E0_star.unique()); T=np.sort(r.T_star.unique())
def cls(R): return 0 if R>=0.75 else (2 if R<0.4 else 1)   # 0 plasma, 1 crossover, 2 condensed
G={}
for init in ('hot','cold'):
    s=r[r.init==init].pivot(index='T_star',columns='E0_star',values='R_DH').loc[T,E]
    G[init]=np.vectorize(cls)(s.values)
Rm=r.groupby(['T_star','E0_star']).R_DH.mean().unstack().loc[T,E].values
Cm=np.vectorize(cls)(Rm); dis=G['hot']!=G['cold']
fig,ax=plt.subplots(figsize=(7.2,5.6))
cmap=ListedColormap(['#4c78a8','#e8d28a','#e07b39'])
dE,dT=np.diff(E).mean(),np.diff(T).mean()
ax.pcolormesh(np.append(E-dE/2,E[-1]+dE/2),np.append(T-dT/2,T[-1]+dT/2),Cm,cmap=cmap,vmin=0,vmax=2)
ix,iy=np.where(dis); ax.scatter(E[iy],T[ix],marker='x',c='k',s=55,lw=2)
ax.axhspan(0.025,0.075,color='white',alpha=.55,zorder=3); ax.text(0.5,0.05,'T*=0.05: not equilibrated (hot ≠ cold)',ha='center',va='center',fontsize=8,zorder=4)
# BKT line from 9.21 (rho*=0.05)
t=pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv'); t=t[(t.m2==1)&np.isclose(t.rho_star,0.05)]
bx,by=[],[]
for e,g in t.groupby('E0_star'):
    g=g.sort_values('T_star'); x=g.T_star.values; f=g.inv_eps.values-4*x
    if f[0]>0:
        i=np.where((f[:-1]>0)&(f[1:]<=0))[0]
        if len(i): i=i[0]; bx.append(e); by.append(x[i]+(x[i+1]-x[i])*f[i]/(f[i]-f[i+1]))
ax.plot(bx,by,'-D',color='red',ms=4,lw=1.5,label='BKT: 1/ε = 4T* (9.21, ρ*=0.05)')
ax.set_xlim(-0.125,1.125); ax.set_ylim(0.025,0.575)
ax.set_xlabel('E0*'); ax.set_ylabel('T*'); ax.set_title('S_QQ(k)-shape phase map, N=50, ρ*=0.05')
h=[Patch(color='#4c78a8',label='plasma:  R ≥ 0.75  (screens like Debye–Hückel)'),Patch(color='#e8d28a',label='crossover:  0.4 ≤ R < 0.75'),
   Patch(color='#e07b39',label='condensed:  R < 0.4  (charge suppressed at low k)'),
   plt.Line2D([],[],color='red',marker='D',lw=1.5,label='BKT: 1/ε = 4T* (9.21)'),plt.Line2D([],[],color='k',marker='x',lw=0,label='hot ≠ cold class')]
ax.legend(handles=h,loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig('sqq_shape_phase_map.png',dpi=150)
for c,n in zip(range(3),['plasma','crossover','condensed']): print(n,int((Cm[1:]==c).sum()),'of',Cm[1:].size,'(T*>=0.15 cells)')
print('class by cell (rows T* high->low, cols E0*):'); print(np.flipud(Cm))
