import numpy as np, pandas as pd
N=50
t = pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv')
t = t[t.m2==1].copy(); t['snn']=t.S_NN/N
rows=[]
for (rho,T),g in t.groupby(['rho_star','T_star']):
    g=g.sort_values('E0_star'); E=g.E0_star.values
    q=g.inv_eps.values; n=g.snn.values; sq=g.inv_eps_sem.values
    if q.max()-q.min()<0.15: continue            # no transition along E0 at this (rho,T)
    kq=np.argmax(np.diff(q)); kn=np.argmax(np.diff(n))   # 1/eps RISES with E0*; S_NN/N rises with E0*
    rows.append((rho,T,(E[kq]+E[kq+1])/2,np.diff(q)[kq],(E[kn]+E[kn+1])/2,np.diff(n)[kn]))
r=pd.DataFrame(rows,columns=['rho','T','E0_QQ','dQQ','E0_NN','dNN'])
r['gap']=(r.E0_QQ-r.E0_NN)
print(r[np.isclose(r.rho,0.05)].round(3).to_string(index=False))
print(r.groupby('rho').agg(n=('gap','size'),mean_gap=('gap','mean'),mean_abs_gap=('gap',lambda x:x.abs().mean())).round(3))
r.to_csv('alongE0_steepest.csv',index=False)
