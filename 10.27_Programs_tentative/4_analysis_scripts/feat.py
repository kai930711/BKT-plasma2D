import numpy as np, pandas as pd
N=50; rho=0.05
r=pd.read_csv('wide_runs.csv'); z=np.load('wide_shells.npz')
ks=2*np.pi*np.sqrt(z['shells'])*np.sqrt(rho/N)
sq=z['S_QQ']/N; ie=z['inv_eps']; sn=z['S_NN']/N
def band(a,lo,hi): m=(ks>=lo)&(ks<=hi); return a[:,m].mean(axis=1)
F=r[['init','E0_star','T_star']].copy()
F['ie_kmin']=ie[:,0]; F['lowk']=band(sq,0.9,1.5); F['peak']=band(sq,2.5,4.0); F['snn']=sn[:,0]
F['peak_over_lowk']=F.peak/F.lowk
F['4T']=4*F.T_star
pd.set_option('display.width',200)
print(F.sort_values('snn').round(2).to_string(index=False))
F.to_csv('sqq_features.csv',index=False)
