import os

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() not in ("agg",):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_snapshots1(snapshots, q, sample_every, kappa=None, interval=80, save_path=None):
    q = np.asarray(q)
    mask_plus = q == +1
    mask_minus = q == -1

    n_frames = len(snapshots)
    sweep_indices = np.arange(n_frames) * sample_every

    fig, ax = plt.subplots(figsize=(5, 5))

    scat_plus = ax.scatter([], [], marker="+", s=120, label="+")
    scat_minus = ax.scatter([], [], marker="_", s=120, label="-")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R_x$")
    ax.set_ylabel(r"$R_y$")

    if kappa is None:
        ax.set_title("Configuration")
    else:
        ax.set_title(rf"Configuration, $\kappa=K/T={kappa:.3g}$")

    ax.legend(loc="upper right")

    sweep_text = ax.text(
        0.03, 0.96, "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
    )

    def update(frame):
        R = snapshots[frame]

        scat_plus.set_offsets(R[mask_plus])
        scat_minus.set_offsets(R[mask_minus])
        sweep_text.set_text(f"sweep {sweep_indices[frame]}")

        return scat_plus, scat_minus, sweep_text

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=True,
    )

    if save_path is not None:
        from matplotlib.animation import PillowWriter
        fps = max(1, round(1000.0 / interval))
        anim.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return save_path

    plt.close(fig)

    from IPython.display import HTML
    return HTML(anim.to_jshtml())


def plot_snapshot1(snapshots, q, sample_every, sweep=None, frame=None, kappa=None, save_path=None):

    q = np.asarray(q)
    mask_plus = q == +1
    mask_minus = q == -1

    if frame is None:
        if sweep is None:
            frame = -1
        else:
            frame = sweep // sample_every

    if frame < 0:
        frame = len(snapshots) + frame

    if frame < 0 or frame >= len(snapshots):
        raise ValueError(f"frame must be between 0 and {len(snapshots)-1}.")

    R = snapshots[frame]
    shown_sweep = frame * sample_every

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(R[mask_plus, 0], R[mask_plus, 1], marker="+", s=120, label="+")
    ax.scatter(R[mask_minus, 0], R[mask_minus, 1], marker="_", s=120, label="-")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R_x$")
    ax.set_ylabel(r"$R_y$")

    if kappa is None:
        ax.set_title(f"Snapshot at sweep {shown_sweep}")
    else:
        ax.set_title(rf"Snapshot at sweep {shown_sweep}, $\kappa={kappa:.3g}$")

    ax.legend(loc="upper right")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()


def plot_final_snapshot1(snapshots, q, sample_every, kappa=None, save_path=None):
    return plot_snapshot1(
        snapshots=snapshots,
        q=q,
        sample_every=sample_every,
        frame=-1,
        kappa=kappa,
        save_path=save_path,
    )
