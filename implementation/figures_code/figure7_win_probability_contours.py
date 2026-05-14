"""
Recreate Figure 7: win-probability contours for optimal play in Pig.

The plot shows isosurfaces of the optimal value function V(i, j, k), where:
    i = Player 1/current-player score
    j = Player 2/opponent score
    k = current turn total

Default contour levels are 3%, 9%, 27%, and 81%.

Expected files in the same folder:
    pig_value_cache.pkl       # preferred: cached value function
    pig_value_iteration.py    # fallback: recompute value function if cache is absent
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

State = tuple[int, int, int]
ValueFunction = dict[State, float]


def load_or_compute_value_function(
    script_dir: Path,
    cache_path: Path,
    *,
    goal: int = 100,
    epsilon: float = 1e-10,
) -> ValueFunction:
    """Load a cached optimal value function, or compute and cache one."""
    if cache_path.exists():
        with cache_path.open("rb") as f:
            value_func = pickle.load(f)
        if not isinstance(value_func, dict):
            raise TypeError(f"Expected {cache_path} to contain a dict, got {type(value_func)!r}")
        return value_func

    sys.path.insert(0, str(script_dir))
    from pig_value_iteration import optimal_pig_value_iteration

    _policy, value_func = optimal_pig_value_iteration(goal=goal, epsilon=epsilon)
    with cache_path.open("wb") as f:
        pickle.dump(value_func, f, protocol=pickle.HIGHEST_PROTOCOL)
    return value_func


def build_value_volume(value_func: ValueFunction, *, goal: int = 100) -> np.ndarray:
    """
    Convert the sparse value-function dictionary into a dense 3D grid.

    The output is indexed as volume[i, j, k]. Terminal states with i + k >= goal
    are assigned value 1 because holding wins immediately.
    """
    volume = np.empty((goal + 1, goal + 1, goal + 1), dtype=np.float32)

    for i in range(goal + 1):
        for j in range(goal + 1):
            for k in range(goal + 1):
                if i >= goal:
                    volume[i, j, k] = 1.0
                elif j >= goal:
                    volume[i, j, k] = 0.0
                elif i + k >= goal:
                    volume[i, j, k] = 1.0
                else:
                    volume[i, j, k] = value_func[(i, j, k)]

    return volume


def add_isosurface(
    ax: plt.Axes,
    volume: np.ndarray,
    level: float,
    *,
    face_gray: str,
    alpha: float = 0.93,
) -> None:
    """Extract and draw one value-function isosurface."""
    verts, faces, _normals, _values = marching_cubes(volume, level=level, spacing=(1, 1, 1))

    # marching_cubes returns vertices as (i, j, k). The figure labels x as j,
    # depth as i, and vertical as k, so reorder to (j, i, k).
    plotted = np.column_stack((verts[:, 1], verts[:, 0], verts[:, 2]))
    mesh = Poly3DCollection(
        plotted[faces],
        facecolor=face_gray,
        edgecolor="none",
        linewidth=0.0,
        alpha=alpha,
    )
    ax.add_collection3d(mesh)


def label_contours(
    ax: plt.Axes,
    levels: Iterable[float],
    *,
    goal: int = 100,
) -> None:
    """Place simple percentage labels near the left/front side of the surface stack."""
    # Hand-tuned positions chosen to mimic the scanned figure's label placement.
    label_positions = {
        0.03: (96, 4, 18),
        0.09: (94, 7, 38),
        0.27: (92, 10, 61),
        0.81: (88, 16, 94),
    }
    for level in levels:
        x, y, z = label_positions.get(round(level, 2), (goal - 8, 8, level * goal))
        ax.text(
            x,
            y,
            z,
            f"{int(round(100 * level))}%",
            fontsize=8,
            color="white" if level < 0.5 else "black",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="0.15", edgecolor="none", alpha=0.75)
            if level < 0.5
            else None,
        )


def plot_win_probability_contours(
    value_func: ValueFunction,
    *,
    goal: int = 100,
    levels: tuple[float, ...] = (0.03, 0.09, 0.27, 0.81),
    output: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Create the Figure 7 style 3D win-probability contour plot."""
    volume = build_value_volume(value_func, goal=goal)

    fig = plt.figure(figsize=(5.2, 4.6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    # Darker surfaces represent lower win probabilities, lighter surfaces higher ones.
    grays = {
        0.03: "0.08",
        0.09: "0.18",
        0.27: "0.36",
        0.81: "0.62",
    }
    for level in sorted(levels, reverse=True):
        add_isosurface(ax, volume, level, face_gray=grays.get(round(level, 2), "0.45"))

    label_contours(ax, levels, goal=goal)

    ax.set_xlim(goal, 0)
    ax.set_ylim(0, goal)
    ax.set_zlim(0, goal)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("Player 2 Score (j)", labelpad=8)
    ax.set_ylabel("Player 1\nScore (i)", labelpad=10)
    ax.set_zlabel("Turn Total (k)", labelpad=8)

    ax.set_xticks([0, goal])
    ax.set_yticks([0, goal])
    ax.set_zticks([0, goal])

    # Viewpoint chosen to approximate the perspective in the supplied image.
    ax.view_init(elev=16, azim=-76)

    # Use pale panes/gridlines like the original printed figure.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor("0.75")
    ax.grid(True, color="0.82", linewidth=0.5)
    ax.tick_params(labelsize=7, pad=0)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.98)

    if output is not None:
        fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate Pig Figure 7 win-probability contours.")
    parser.add_argument("--goal", type=int, default=100)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).with_name("pig_value_cache.pkl"),
        help="Path to cached value-function pickle.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=None, help="Optional image output path.")
    parser.add_argument("--no-show", action="store_true", help="Save without opening an interactive window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    value_func = load_or_compute_value_function(
        script_dir,
        args.cache,
        goal=args.goal,
        epsilon=args.epsilon,
    )
    plot_win_probability_contours(
        value_func,
        goal=args.goal,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
