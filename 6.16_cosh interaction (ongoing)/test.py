import numpy as np
from Plasma_Interaction import energy2_pair, energy2, energy_total

R_all = np.array([
    [0.10, 0.10],
    [0.20, 0.10],
    [0.80, 0.80],
    [0.90, 0.80],
])

q = np.array([1, -1, 1, -1])

kappa = 3.0
alpha = np.sqrt(np.pi)
n_cut = 1
m_cut = 3

E0_over_T = 10.0
xiF_over_xi = 5.0
a = 0.02

U2 = energy2(R_all, E0_over_T, xiF_over_xi, a)

U_only_log = energy_total(
    R_all, q, kappa, alpha, n_cut, m_cut,
    E0_over_T, xiF_over_xi, a,
    include_log=True,
    include_cosh=False,
)

U_only_cosh = energy_total(
    R_all, q, kappa, alpha, n_cut, m_cut,
    E0_over_T, xiF_over_xi, a,
    include_log=False,
    include_cosh=True,
)

U_both = energy_total(
    R_all, q, kappa, alpha, n_cut, m_cut,
    E0_over_T, xiF_over_xi, a,
    include_log=True,
    include_cosh=True,
)

print("U2 =", U2)
print("only log =", U_only_log)
print("only cosh =", U_only_cosh)
print("both =", U_both)
print("check both - sum =", U_both - (U_only_log + U_only_cosh))