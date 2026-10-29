import numpy as np, pandas as pd
d = pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv')
d = d[d.m2==1].copy(); N=50
rows=[]
for (rho,E0), g in d.groupby(['rho_star','E0_star']):
    g = g.sort_values('T_star')
    T=g.T_star.values; snn=g.S_NN.values/N; ie=g.inv_eps.values; ie_s=g.inv_eps_sem.values
    acc=g.acceptance_rate.values
    clus = snn>5
    # require a clustered->unclustered change inside the window (a transition on heating)
    if not (clus[:3].any() and (~clus[-3:]).any()): continue
    i_snn = int(np.argmax(clus[::-1]==False) ) # unused
    # T where S_NN/N falls through 5 (linear interp in log S_NN)
    j = np.where(clus[:-1] & ~clus[1:])[0]
    if len(j)==0: continue
    j=j[0]; l1,l2=np.log(snn[j]),np.log(snn[j+1])
    Tc = T[j]+(np.log(5)-l1)/(l2-l1)*(T[j+1]-T[j])
    # biggest single-step jump in S_QQ-derived 1/eps(k_min), in units of its sem
    dz = np.abs(np.diff(ie))/np.sqrt(ie_s[:-1]**2+ie_s[1:]**2)
    k = int(np.argmax(dz)); Tjump = 0.5*(T[k]+T[k+1])
    # biggest single-step drop of S_NN in log
    ks = int(np.argmax(np.abs(np.diff(np.log(snn))))); Tsnn_jump = 0.5*(T[ks]+T[ks+1])
    rows.append(dict(rho=rho,E0=E0,Tc_SNN_cross5N=round(Tc,3),T_maxjump_SNN=round(Tsnn_jump,3),
                     T_maxjump_invEps=round(Tjump,3), max_dz_invEps=round(dz[k],1),
                     jump_ratio_SNN=round(np.exp(np.abs(np.diff(np.log(snn))).max()),1),
                     frozen_pts=int((acc<0.02).sum())))
r = pd.DataFrame(rows); print(r.to_string(index=False))
r['dT'] = (r.T_maxjump_invEps - r.T_maxjump_SNN).abs()
print(f"\n{len(r)} series with a clustered->unclustered change on heating")
print("max S_NN step and max 1/eps step land in the same or adjacent T* interval (<=0.02 apart): "
      f"{int((r.dT<=0.021).sum())}/{len(r)}")
