"""Visualization for 2D plasma Monte Carlo results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_run(snapshots, E_trace, pair_trace, charges, L, K, sample_every):
    """Build a 3-panel animation and return an IPython HTML object for display."""
    sweep_indices = np.arange(len(E_trace)) * sample_every
    n_frames = len(snapshots)
    mask_plus = charges == +1
    mask_minus = charges == -1

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Energy trace
    line_e, = axes[0].plot([], [], 'b-')
    axes[0].set_xlim(0, sweep_indices[-1])
    axes[0].set_ylim(
        np.min(E_trace) * 1.05,
        np.max(E_trace) * 1.05 if np.max(E_trace) > 0 else np.max(E_trace) * 0.95
    )
    axes[0].set_xlabel("Sweep")
    axes[0].set_ylabel("Total energy")
    axes[0].set_title("Energy trace")

    # Pairing trace
    line_p, = axes[1].plot([], [], 'r-')
    axes[1].set_xlim(0, sweep_indices[-1])
    axes[1].set_ylim(0, np.max(pair_trace) * 1.1)
    axes[1].set_xlabel("Sweep")
    axes[1].set_ylabel("Mean nearest opposite-charge distance")
    axes[1].set_title("Pairing indicator")

    # Configuration scatter
    scat_plus = axes[2].scatter([], [], marker="+", s=120, label="+")
    scat_minus = axes[2].scatter([], [], marker="_", s=120, label="-")
    axes[2].set_xlim(0, L)
    axes[2].set_ylim(0, L)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title(f"Configuration (K={K:.2f})")
    axes[2].legend()
    sweep_text = axes[2].text(
        0.02, 0.95, '', transform=axes[2].transAxes,
        fontsize=10, verticalalignment='top'
    )

    plt.tight_layout()

    def update(frame):
        f = frame + 1
        line_e.set_data(sweep_indices[:f], E_trace[:f])
        line_p.set_data(sweep_indices[:f], pair_trace[:f])

        pos = snapshots[frame]
        scat_plus.set_offsets(pos[mask_plus])
        scat_minus.set_offsets(pos[mask_minus])
        sweep_text.set_text(f"sweep {sweep_indices[frame]}")

        return line_e, line_p, scat_plus, scat_minus, sweep_text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())
