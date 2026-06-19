import sys
import time
import numpy as np
import pandas as pd

from Monte_Carlo import run_mc_by_diff
from Size_Effect_Functions import eta2


def make_neutral_charges(n, seed=0):
    if n % 2 != 0:
        raise ValueError("n must be even.")

    q = np.ones(n, dtype=int)
    q[n // 2:] = -1

    rng = np.random.default_rng(seed)
    rng.shuffle(q)

    return q


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python Size_Effect_by_Diff_OneN.py N")

    n = int(sys.argv[1])

    if n not in [2, 8, 32, 104]:
        raise ValueError("N must be one of: 2, 8, 32, 104")

    total_start = time.time()

    seed = 0

    # Same 10 kappa points used for N=32/N=104 sparse reproduction.
    kappa_values = np.array([
        10.0, 7.0, 6.0, 5.0, 4.0,
        4.75, 3.5, 3.0, 2.0, 1.4
    ], dtype=float)

    alpha = np.sqrt(np.pi)
    n_cut = 1
    m_cut = 6

    step_size = 0.05

    n_burnin = 1000
    n_sweeps = 5000
    sample_every = 20

    include_log = True
    include_cosh = False

    E0_over_T = 0.0
    xiF_over_xi = 1.0

    a = np.sqrt(4.0 * eta2 / (np.pi * n))
    q = make_neutral_charges(n, seed=seed)

    output_name = f"fig2b_summary_N{n}_10pts_by_diff.csv"

    print("========================================", flush=True)
    print("Starting Size_Effect_by_Diff_OneN.py", flush=True)
    print("N =", n, flush=True)
    print("output =", output_name, flush=True)
    print("eta2 =", eta2, flush=True)
    print("a =", a, flush=True)
    print("alpha =", alpha, flush=True)
    print("n_cut =", n_cut, flush=True)
    print("m_cut =", m_cut, flush=True)
    print("step_size =", step_size, flush=True)
    print("n_burnin =", n_burnin, flush=True)
    print("n_sweeps =", n_sweeps, flush=True)
    print("sample_every =", sample_every, flush=True)
    print("seed =", seed, flush=True)
    print("include_log =", include_log, flush=True)
    print("include_cosh =", include_cosh, flush=True)
    print("sum(q) =", np.sum(q), flush=True)
    print("N_plus =", np.sum(q == 1), flush=True)
    print("N_minus =", np.sum(q == -1), flush=True)
    print("kappa_values =", kappa_values, flush=True)
    print("========================================", flush=True)

    summary_records = []

    for idx, k in enumerate(kappa_values, start=1):
        point_start = time.time()

        print("----------------------------------------", flush=True)
        print(f"Starting point {idx}/{len(kappa_values)}", flush=True)
        print("N =", n, flush=True)
        print("kappa =", k, flush=True)
        print("one_over_kappa =", 1.0 / k, flush=True)
        print("----------------------------------------", flush=True)

        result = run_mc_by_diff(
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
            E0_over_T=E0_over_T,
            xiF_over_xi=xiF_over_xi,
            include_log=include_log,
            include_cosh=include_cosh,
            n_burnin=n_burnin,
            seed=seed,
        )

        U1_trace = np.asarray(result["U1_trace"], dtype=float)
        U2_trace = np.asarray(result["U2_trace"], dtype=float)
        beta_U_trace = np.asarray(result["beta_U_trace"], dtype=float)

        acceptance_rate = result["acceptance_rate"]

        y_samples = U1_trace / n - 0.5 * np.log(a)

        elapsed = time.time() - point_start

        record = {
            "N": n,
            "a": a,
            "kappa": k,
            "one_over_kappa": 1.0 / k,
            "U1_mean": np.mean(U1_trace),
            "U1_std": np.std(U1_trace, ddof=1),
            "U_over_Ne2": np.mean(y_samples),
            "U_over_Ne2_std": np.std(y_samples, ddof=1),
            "U_over_Ne2_sem": np.std(y_samples, ddof=1) / np.sqrt(len(y_samples)),
            "U2_mean": np.mean(U2_trace),
            "beta_U_mean": np.mean(beta_U_trace),
            "acceptance_rate": acceptance_rate,
            "n_samples": len(U1_trace),
            "seed": seed,
            "include_log": include_log,
            "include_cosh": include_cosh,
            "elapsed_seconds": elapsed,
        }

        summary_records.append(record)

        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(output_name, index=False)

        print("Finished this point", flush=True)
        print("elapsed_seconds =", elapsed, flush=True)
        print("n_samples =", len(U1_trace), flush=True)
        print("U1_mean =", record["U1_mean"], flush=True)
        print("U_over_Ne2 =", record["U_over_Ne2"], flush=True)
        print("acceptance_rate =", acceptance_rate, flush=True)
        print("Saved partial CSV:", output_name, flush=True)

    total_elapsed = time.time() - total_start

    print("========================================", flush=True)
    print("Finished all points for N =", n, flush=True)
    print("Total elapsed seconds =", total_elapsed, flush=True)
    print("Final output:", output_name, flush=True)
    print("========================================", flush=True)


if __name__ == "__main__":
    main()
