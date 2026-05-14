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


def load_roll_optimal_states(
    path: str | Path = "reachable_states_roll_optimal.npy",
    *,
    goal: int = 100,
) -> np.ndarray:
    """
    Load reachable states where rolling is optimal and convert them to a 3D Boolean grid.

    The expected .npy file is an (n, 3) array whose rows are:
        [player_1_score, player_2_score, turn_total]

    This function converts those coordinate rows into a grid with the same layout as
    reachable_states.npy, so it can be plotted with the same 3D voxel function.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path!s}. "
            "Make sure reachable_states_roll_optimal.npy is in the same folder as this script."
        )
    coordinates = np.load(path)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "Expected reachable_states_roll_optimal.npy to have shape (n, 3), "
            f"got {coordinates.shape}"
        )

    coordinates = coordinates.astype(np.int64, copy=False)
    grid = np.zeros((goal + 1, goal + 1, goal + 1), dtype=bool)
    if coordinates.size == 0:
        return grid
    if coordinates.min() < 0 or coordinates.max() > goal:
        raise ValueError(
            f"All roll-optimal coordinates must be between 0 and {goal}; "
            f"found min={coordinates.min()} and max={coordinates.max()}"
        )
    grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = True
    return grid


def downsample_boolean_grid(grid: np.ndarray, step: int) -> np.ndarray:
    """
    Downsample a 3D Boolean grid by taking one point every `step` cells.
    Use step=1 for maximum detail, but it may be slow.
    """
    if step < 1:
        raise ValueError("step must be at least 1")
    return grid[::step, ::step, ::step]


def plot_3d_view(
    states: np.ndarray,
    *,
    title: str,
    elevation: float = 24,
    azimuth: float = -63,
    step: int = 2,
    goal: int = 100,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create one separate 3D voxel plot for a state set.

    The figure is returned so the caller can decide when to display it.
    """
    # Restrict to the standard 0..100 Pig score/turn-total cube.
    states = states[: goal + 1, : goal + 1, : goal + 1]
    states_small = downsample_boolean_grid(states, step=step)

    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        states_small,
        facecolors="0.55",
        edgecolors="0.20",
        linewidth=0.15,
        alpha=0.75,
    )

    max_axis = states_small.shape[0] - 1
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

    ax.set_title(title)
    ax.view_init(elev=elevation, azim=azimuth)

    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

    ax.grid(True)
    fig.tight_layout()

    return fig, ax


def main() -> None:
    goal = 100
    step = 2
    reachable_states = load_reachable_states("reachable_states.npy")
    roll_optimal_states = load_roll_optimal_states(
        "reachable_states_roll_optimal.npy",
        goal=goal,
    )

    # This creates figure 1: all reachable states.
    plot_3d_view(
        reachable_states,
        title="Reachable States",
        elevation=24,
        azimuth=-63,
        step=step,
        goal=goal,
    )

    # This creates figure 2: only reachable states where rolling is optimal.
    plot_3d_view(
        roll_optimal_states,
        title="Reachable States Where Rolling Is Optimal",
        elevation=24,
        azimuth=-63,
        step=step,
        goal=goal,
    )

    # Display both separate figures. They are not subplots and are not overlaid.
    plt.show()


if __name__ == "__main__":
    main()
