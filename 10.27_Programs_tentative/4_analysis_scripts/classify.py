import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
N=50; rho=0.05
r=pd.read_csv('wide_runs.csv'); z=np.load('wide_shells.npz')
ks=2*np.pi*np.sqrt(z['shells'])*np.sqrt(rho/N)
sq=z['S_QQ']/N; sn=z['S_NN'][:,0]/N
T=r.T_star.values
kD2=2*np.pi*rho/T                      # Debye k_D^2 sigma^2 = 2*pi*kappa*n
band=(ks>=0.5)&(ks<=1.5)
dh=ks[None,band]**2/(ks[None,band]**2+kD2[:,None])       # Debye-Hueckel S_QQ/N
R=(sq[:,band]/dh).mean(axis=1)                            # ~1 for a screening plasma, <<1 if charge is suppressed
r['R_DH']=R; r['snn']=sn; r['clustered']=sn>5
print('AUC (R_DH predicts NOT clustered):',round(roc_auc_score(~r.clustered,R),3))
for cut in (0.3,0.4,0.5,0.6,0.7):
    print(cut,'accuracy plasma-vs-clustered:',round(((R>=cut)==(~r.clustered)).mean(),3))
print(r.sort_values('R_DH')[['init','E0_star','T_star','R_DH','snn']].round(2).to_string(index=False))
r.to_csv('classify_R.csv',index=False)
