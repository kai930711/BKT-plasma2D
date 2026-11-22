import numpy as np
import pandas as pd

from Monte_Carlo import run_mc1
from Regeneration_of_Size_Effect import eta2


def make_neutral_charges(n, seed=0):
    if n % 2 != 0:
        raise ValueError("n must be even.")

    q = np.ones(n, dtype=int)
    q[n // 2:] = -1

    rng = np.random.default_rng(seed)
    rng.shuffle(q)

    return q


def main():
    seed = 0

    N_values = [2, 8, 32, 104]
    kappa_values = np.linspace(1, 10, 49)

    alpha = np.sqrt(np.pi)
    n_cut = 1
    m_cut = 6

    step_size = 0.05

    n_burnin = 1000
    n_sweeps = 5000
    sample_every = 20

    summary_records = []

    for n in N_values:
        q = make_neutral_charges(n, seed=seed)
        a = np.sqrt(4.0 * eta2 / (np.pi * n))

        for k in kappa_values:
            print(f"Running N={n}, a={a}, kappa={k}", flush=True)

            snapshots, U1_trace, acceptance_rate = run_mc1(
                N=n,
                q=q,
                kappa=k,
                a=a,
                alpha=alpha,
                n_cut=n_cut,
                m_cut=m_cut,
                step_size=step_size,
                n_sweeps=n_sweeps,
                sample_every=sample_every,
                n_burnin=n_burnin,
                seed=seed,
            )

            U1_trace = np.asarray(U1_trace, dtype=float)

            y_samples = U1_trace / n - 0.5 * np.log(a)

            summary_records.append({
                "N": n,
                "a": a,
                "kappa": k,
                "one_over_kappa": 1.0 / k,
                "U1_mean": np.mean(U1_trace),
                "U1_std": np.std(U1_trace, ddof=1),
                "U_over_Ne2": np.mean(y_samples),
                "U_over_Ne2_std": np.std(y_samples, ddof=1),
                "U_over_Ne2_sem": np.std(y_samples, ddof=1) / np.sqrt(len(y_samples)),
                "acceptance_rate": acceptance_rate,
                "n_samples": len(U1_trace),
                "seed": seed,
            })

            
    summary_df = pd.DataFrame(summary_records)

    summary_df.to_csv("fig2b_summary.csv", index=False)

    print("Saved fig2b_summary.csv", flush=True)


if __name__ == "__main__":
    main()