import numpy as np

eta2 = 5e-3

def Energy_corrected1(U1_trace, N):
    U1_trace = np.asarray(U1_trace, dtype=float) 
    U1_mean = np.mean(U1_trace)
    a = np.sqrt(4.0 * eta2 / (np.pi * N))
    return U1_mean / N - 0.5 * np.log(a)