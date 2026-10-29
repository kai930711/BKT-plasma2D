import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def find_intersections_one_curve(T, y):
    """
    Find intersections between y(T) and 4T by linear interpolation.
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(T)
    T = T[order]
    y = y[order]

    diff = y - 4.0 * T

    intersections = []

    for i in range(len(T) - 1):
        T1, T2 = T[i], T[i + 1]
        y1, y2 = y[i], y[i + 1]
        d1, d2 = diff[i], diff[i + 1]

        if np.isnan(d1) or np.isnan(d2):
            continue

        if d1 == 0.0:
            intersections.append({
                "T_cross": T1,
                "kappa_cross": 1.0 / T1,
                "eps_inv_cross": y1,
                "method": "exact_point",
                "left_T": T1,
                "right_T": T1,
            })

        elif d1 * d2 < 0.0:
            T_cross = T1 - d1 * (T2 - T1) / (d2 - d1)
            y_cross = 4.0 * T_cross

            intersections.append({
                "T_cross": T_cross,
                "kappa_cross": 1.0 / T_cross,
                "eps_inv_cross": y_cross,
                "method": "linear_interpolation",
                "left_T": T1,
                "right_T": T2,
            })

    if len(T) > 0 and diff[-1] == 0.0:
        intersections.append({
            "T_cross": T[-1],
            "kappa_cross": 1.0 / T[-1],
            "eps_inv_cross": y[-1],
            "method": "exact_point",
            "left_T": T[-1],
            "right_T": T[-1],
        })

    return intersections


def main():
    input_csv = "op_dielectric_data_julia.csv"
    background_image = "paper_OP_fig4.jpg"

    output_overlay_plot = "op_dielectric_julia_overlay.png"
    output_clean_plot = "op_dielectric_julia_clean.png"
    output_intersections = "op_dielectric_julia_intersections.csv"

    df = pd.read_csv(input_csv)

    required_cols = [
        "N",
        "rho_star",
        "T_star",
        "kappa",
        "ϵ0_inv",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.sort_values(["N", "rho_star", "T_star"]).reset_index(drop=True)

    if "universal_line_4T" not in df.columns:
        df["universal_line_4T"] = 4.0 * df["T_star"]

    if "diff_to_4T" not in df.columns:
        df["diff_to_4T"] = df["ϵ0_inv"] - df["universal_line_4T"]

    # --------------------------------------------------
    # Find intersections
    # --------------------------------------------------

    intersection_rows = []

    for (N, rho_star), group in df.groupby(["N", "rho_star"]):
        group = group.sort_values("T_star")

        T = group["T_star"].to_numpy()
        y = group["ϵ0_inv"].to_numpy()

        intersections = find_intersections_one_curve(T, y)

        if len(intersections) == 0:
            intersection_rows.append({
                "N": N,
                "rho_star": rho_star,
                "T_cross": np.nan,
                "kappa_cross": np.nan,
                "eps_inv_cross": np.nan,
                "method": "no_sign_change",
                "left_T": np.nan,
                "right_T": np.nan,
            })
        else:
            for item in intersections:
                row = {
                    "N": N,
                    "rho_star": rho_star,
                }
                row.update(item)
                intersection_rows.append(row)

    intersections_df = pd.DataFrame(intersection_rows)
    intersections_df.to_csv(output_intersections, index=False)

    # --------------------------------------------------
    # Plot function
    # --------------------------------------------------

    def make_plot(with_background):
        fig, ax = plt.subplots(figsize=(7, 6))

        if with_background:
            if not Path(background_image).exists():
                raise FileNotFoundError(
                    f"Cannot find {background_image}. "
                    f"Put the paper image in this folder and name it {background_image}."
                )

            img = mpimg.imread(background_image)

            # Treat the cropped O&P Fig.4 image as occupying:
            # x = T* from 0 to 0.5
            # y = 1/epsilon0 from 0 to 1.2
            ax.imshow(
                img,
                extent=[0.0, 0.5, 0.0, 1.2],
                origin="upper",
                aspect="auto",
                alpha=0.45,
                zorder=0,
            )

        # Plot MC curves
        for (N, rho_star), group in df.groupby(["N", "rho_star"]):
            group = group.sort_values("T_star")

            T = group["T_star"].to_numpy()
            y = group["ϵ0_inv"].to_numpy()

            ax.plot(
                T,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=rf"Our MC: $\rho^*={rho_star:g}$, $N={int(N)}$",
                zorder=3,
            )

        # Universal jump line: 1/epsilon0 = 4T*
        T_line = np.linspace(0.0, 0.5, 400)

        ax.plot(
            T_line,
            4.0 * T_line,
            linestyle="--",
            linewidth=1.6,
            label=r"$4T^*$",
            zorder=2,
        )

        # Plot intersection points
        for _, row in intersections_df.iterrows():
            if np.isfinite(row["T_cross"]):
                ax.scatter(
                    row["T_cross"],
                    row["eps_inv_cross"],
                    s=70,
                    zorder=5,
                )

                ax.annotate(
                    rf"$T_c^*={row['T_cross']:.3f}$",
                    xy=(row["T_cross"], row["eps_inv_cross"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=6,
                )

        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(0.0, 1.2)

        ax.set_xlabel(r"$T^*$")
        ax.set_ylabel(r"$1/\epsilon_0$")
        ax.set_title(r"O&P Fig.4 reproduction with Julia MC")

        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        return fig, ax

    fig_overlay, _ = make_plot(with_background=True)
    fig_overlay.savefig(output_overlay_plot, dpi=300)
    plt.close(fig_overlay)

    fig_clean, _ = make_plot(with_background=False)
    fig_clean.savefig(output_clean_plot, dpi=300)
    plt.close(fig_clean)

    print(f"Saved overlay plot to {output_overlay_plot}")
    print(f"Saved clean plot to {output_clean_plot}")
    print(f"Saved intersections to {output_intersections}")
    print()
    print(intersections_df)


if __name__ == "__main__":
    main()