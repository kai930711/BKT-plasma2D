import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
df = pd.read_csv('sqq_only_features.csv', index_col=0)
shells=[1,2,4,5,8,9,10,13,16,17]; ie=[f'ie_{m}' for m in shells]
a = np.sqrt(df.rho_star/50); df['k_sigma_max'] = 2*np.pi*np.sqrt(17)*a
def auc_cv(sub, cols, label='clustered'):
    if sub[label].nunique()<2 or len(sub)<60: return np.nan
    g = sub.rho_star.round(4).astype(str).values
    p = cross_val_predict(GradientBoostingClassifier(random_state=0), sub[cols].values, sub[label].values,
                          cv=GroupKFold(n_splits=min(5, sub.rho_star.nunique())), groups=g, method='predict_proba')[:,1]
    return roc_auc_score(sub[label], p)
ok = ~df.frozen
print('--- S_QQ-only AUC for S_NN-clustered, by regime (GBM, grouped CV, frozen states removed) ---')
for name, sub in [('all', df[ok]),
                  ('conductor only (1/eps<4T*)', df[ok & df.conductor]),
                  ('dielectric only', df[ok & ~df.conductor]),
                  ('rho*>=0.1  (k*sigma_max ~1-1.6, reaches droplet scale)', df[ok & (df.rho_star>=0.1)]),
                  ('rho*<=0.01 (k*sigma_max <=0.2, long-wavelength only)', df[ok & (df.rho_star<=0.01)])]:
    print(f'  {name:58s} n={len(sub):4d}  AUC={auc_cv(sub, ie):.3f}   (frac clustered={sub.clustered.mean():.2f})')
print('\nk*sigma reached by the stored shells (m2<=17):')
print(df.groupby('rho_star').k_sigma_max.first().round(2).to_string())
