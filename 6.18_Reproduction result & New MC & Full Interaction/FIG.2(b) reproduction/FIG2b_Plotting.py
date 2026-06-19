import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


folder = Path(".").resolve()

files = sorted(folder.glob("fig2b_summary_N*.csv"))

if not files:
    files = sorted(folder.glob("fig2b_summary_N*.csv"))

if not files:
    raise FileNotFoundError("No fig2b_summary_N*.csv files found in this folder.")

print("Found CSV files:")
for f in files:
    print("  ", f.name)


def extract_N(path):
    m = re.search(r"N(\d+)", path.name)
    if m is None:
        return 10**9
    return int(m.group(1))


files = sorted(files, key=extract_N)

markers = {
    2: "o",
    8: "s",
    32: "^",
    104: "v",
}

plt.figure(figsize=(7.0, 5.0))

for f in files:
    df = pd.read_csv(f)

    if "N" not in df.columns:
        print(f"Skipping {f.name}: no N column")
        continue

    N = int(df["N"].iloc[0])

    if "one_over_kappa" not in df.columns:
        df["one_over_kappa"] = 1.0 / df["kappa"]

    df = df.sort_values("one_over_kappa")

    print()
    print(f"{f.name}:")
    print(df[["N", "kappa", "one_over_kappa", "U_over_Ne2", "acceptance_rate", "n_samples"]])

    plt.plot(
        df["one_over_kappa"],
        df["U_over_Ne2"],
        marker=markers.get(N, "o"),
        linestyle="-",
        label=f"N={N}",
    )

plt.xlabel(r"$1/\kappa$")
plt.ylabel(r"$U^{ex}/(N e^2)$")
plt.title(r"Size effect, $\eta_2=5\times 10^{-3}$")
plt.legend()
plt.tight_layout()

out_png = folder / "fig2b.png"
plt.savefig(out_png, dpi=300)
print()
print("Saved:", out_png)

plt.show()