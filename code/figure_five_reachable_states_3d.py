from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_reachable_states(path: str | Path = "reachable_states.npy") -> np.ndarray:
    """
    Load reachable states from a NumPy .npy file.

    The expected array layout is:
        reachable[i, j, k]

    where:
        i = player 1 score
        j = player 2 score
        k = current turn total
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path!s}. "
            "Make sure reachable_states.npy is in the same folder as this script."
        )
    reachable = np.load(path)
    if reachable.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {reachable.shape}")
    return reachable.astype(bool)

def downsample_boolean_grid(grid: np.ndarray, step: int) -> np.ndarray:
    """
    Downsample a 3D Boolean grid by taking one point every `step` cells.
    Use step=1 for maximum detail, but it may be slow.
    """
    if step < 1:
        raise ValueError("step must be at least 1")
    return grid[::step, ::step, ::step]

def plot_3d_view(
    reachable: np.ndarray,
    *,
    elevation: float = 24,
    azimuth: float = -63,
    step: int = 2,
    goal: int = 100,
) -> None:
    """
    Plot one 3D voxel view of the reachable-state set.
    """
    # Restrict to the standard 0..100 Pig score/turn-total cube.
    reachable = reachable[: goal + 1, : goal + 1, : goal + 1]
    reachable_small = downsample_boolean_grid(reachable, step=step)
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        reachable_small,
        facecolors="0.55",
        edgecolors="0.20",
        linewidth=0.15,
        alpha=0.75,
    )

    max_axis = reachable_small.shape[0] - 1
    tick_positions = [0, max_axis]
    tick_labels = ["0", str(goal)]

    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_zlim(0, max_axis)

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_zticks(tick_positions)

    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_zticklabels(tick_labels)

    ax.set_xlabel("Player 1 Score (i)", labelpad=8)
    ax.set_ylabel("Player 2 Score (j)", labelpad=8)
    ax.set_zlabel("Turn Total (k)", labelpad=8)

    ax.set_title("Reachable States")
    ax.view_init(elev=elevation, azim=azimuth)

    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

    ax.grid(True)
    fig.tight_layout()
    plt.show()

def main() -> None:
    reachable_states = load_reachable_states("reachable_states.npy")
    # Increase detail by changing step to 1.
    # If the plot is slow or too visually dense, use step=2 or step=3.
    plot_3d_view(
        reachable_states,
        elevation=24,
        azimuth=-63,
        step=1,
    )

if __name__ == "__main__":
    main()
