import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Folder Path
folder = Path("###")

# Graph From Paper
img_path = folder / "###"
img = mpimg.imread(img_path)

# csv files
files = [
    ("fig2b_summary_N2.csv", "N=2", "o"),
    ("fig2b_summary_N8.csv", "N=8", "s"),
    ("fig2b_summary_N32.csv", "N=32", "^"),
    ("fig2b_summary_N104.csv", "N=104", "d"),
]


# Overlay Settings
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 0.877

plt.figure(figsize=(7, 6))

plt.imshow(
    img,
    extent=[x_min, x_max, y_min, y_max],
    aspect="auto",
    alpha=0.45,
    zorder=0
)


for filename, label, marker in files:
    path = folder / filename
    if not path.exists():
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(path)
    df = df.sort_values("one_over_kappa")

    plt.plot(
        df["one_over_kappa"],
        df["U_over_Ne2"],
        marker=marker,
        linestyle="-",
        linewidth=2,
        markersize=7,
        label=label,
        zorder=5
    )

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel(r"$T = 1/\Gamma \ (\sim 1/\kappa)$")
plt.ylabel(r"$U/(Ne^2)$")
plt.title("Fig. 2(b) Reproduction")
plt.legend()
plt.tight_layout()

out_path = folder / "fig2b_with_Comparison.png"
plt.savefig(out_path, dpi=300)
print("Saved:", out_path)

plt.show()