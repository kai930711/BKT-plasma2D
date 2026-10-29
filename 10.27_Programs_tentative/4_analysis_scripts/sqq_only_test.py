import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = pd.read_csv('/Users/kai/BKT-plasma2D/9.21_Density Correlation/Results/structure_factors.csv')
N = 50
shells = sorted(d.m2.unique())
meta = d[d.m2==1].set_index('state_id')[['rho_star','T_star','kappa','E0_star','acceptance_rate','S_NN','S_NN_drift_z','inv_eps']]
Q = d.pivot(index='state_id', columns='m2', values='inv_eps')       # 1/eps(k) per shell, S_QQ only
SQ = d.pivot(index='state_id', columns='m2', values='S_QQ')
df = meta.join(Q.add_prefix('ie_')).join(SQ.add_prefix('sq_'))

# ground truth from the existing S_NN / 4T* criteria
df['clustered'] = df.S_NN > 5*N
df['conductor'] = df.inv_eps < 4*df.T_star
df['frozen'] = df.acceptance_rate < 0.02
print('states', len(df), '| clustered', int(df.clustered.sum()), '| conductor', int(df.conductor.sum()), '| frozen', int(df.frozen.sum()))
print(pd.crosstab(df.conductor, df.clustered, rownames=['conductor(S_QQ,4T*)'], colnames=['clustered(S_NN)']))
print('\nclustered fraction by E0*:'); print(df.groupby('E0_star').clustered.mean().round(2).to_string())

# --- Can S_QQ alone predict the S_NN "clustered" label? ---
ie_cols = [f'ie_{m}' for m in shells]
feat_sets = {
 'inv_eps(k_min) only':            ['ie_1'],
 'full S_QQ shell curve (10 shells)': ie_cols,
 'trivial control: T*,rho*,E0* (no S_QQ)': ['T_star','rho_star','E0_star'],
}
y = df.clustered.values
groups = df.rho_star.round(4).astype(str).values           # hold out whole densities -> no neighbour leakage
cv = GroupKFold(n_splits=5)
print('\nPredict S_NN-clustered (grouped CV by rho*):')
for name, cols in feat_sets.items():
    X = df[cols].values
    for mname, model in [('logreg', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))),
                         ('GBM', GradientBoostingClassifier(random_state=0))]:
        p = cross_val_predict(model, X, y, cv=cv, groups=groups, method='predict_proba')[:,1]
        print(f'  {name:42s} {mname:6s} AUC={roc_auc_score(y,p):.3f}  bal.acc={balanced_accuracy_score(y,p>0.5):.3f}')

# --- simple S_QQ-only features: is there a visible signature? ---
df['dip'] = df[ie_cols].min(axis=1)                 # most negative 1/eps(k): overscreening dip
df['tilt'] = df['ie_17'] - df['ie_1']                # rise of 1/eps from k_min to highest shell
df['sqq_hi_over_N'] = df['sq_17']/N
for c in ['ie_1','dip','tilt','sqq_hi_over_N']:
    print(f'  single feature {c:15s} AUC(clustered)={roc_auc_score(y, df[c]):.3f}')
df.to_csv('sqq_only_features.csv')
