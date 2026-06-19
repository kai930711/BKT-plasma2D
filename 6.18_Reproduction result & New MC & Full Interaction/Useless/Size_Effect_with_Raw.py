for sample_index, U1 in enumerate(U1_trace):
                raw_records.append({
                    "N": n,
                    "a": a,
                    "kappa": k,
                    "one_over_kappa": 1.0 / k,
                    "sample_index": sample_index,
                    "U1": U1,
                    "U1_per_particle": U1 / n,
                    "Energy_sample": U1 / n - 0.5 * np.log(a),
                    "seed": seed,
                })

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
