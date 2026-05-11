from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def load_or_compute_policy(script_dir: Path, cache_path: Path, epsilon: float):
    """Load a cached Pig policy, or compute it from pig_value_iteration.py."""
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    sys.path.insert(0, str(script_dir))
    from pig_value_iteration import optimal_pig_value_iteration
    policy, _value_func = optimal_pig_value_iteration(epsilon=epsilon)
    with cache_path.open("wb") as f:
        pickle.dump(policy, f, protocol=pickle.HIGHEST_PROTOCOL)
    return policy

def plot_cross_section(
    reachable_path: Path,
    policy: dict[tuple[int, int, int], str],
    opponent_score: int = 30,
    hold_at: int = 20,
    goal: int = 100,
    max_turn_total: int = 50,
):
    """Plot reachable states and the optimal roll/hold boundary."""
    reachable = np.load(reachable_path)
    if reachable.ndim != 3:
        raise ValueError(f"Expected a 3D reachable-state array, got shape {reachable.shape}")
    # Cross-section: x = current player score i, y = current turn total k.
    i_values = np.arange(goal + 1)
    k_values = np.arange(max_turn_total + 1)
    reachable_plane = reachable[: goal + 1, opponent_score, : max_turn_total + 1].T.astype(float)
    reachable_plane[reachable_plane == 0] = np.nan
    # 1 = hold, 0 = roll. Leave impossible terminal/out-of-domain states as NaN.
    action_grid = np.full((max_turn_total + 1, goal + 1), np.nan, dtype=float)
    for i in range(goal):
        for k in range(min(max_turn_total, goal - i - 1) + 1):
            action_grid[k, i] = 1.0 if policy.get((i, opponent_score, k)) == "hold" else 0.0
    fig, ax = plt.subplots(figsize=(6.2, 3.25), dpi=180)

    # Reachable region, drawn as square cells.
    ax.pcolormesh(
        np.arange(goal + 2) - 0.5,
        np.arange(max_turn_total + 2) - 0.5,
        reachable_plane,
        shading="flat",
        cmap="Greys",
        alpha=0.28,
        vmin=0,
        vmax=1,
        edgecolors="0.72",
        linewidth=0.18,
    )

    # Optimal boundary between roll and hold states.
    ax.contour(
        i_values,
        k_values,
        action_grid,
        levels=[0.5],
        colors="0.12",
        linewidths=1.6,
    )

    # Terminal boundary where holding/turn total reaches the goal.
    x_diag = np.linspace(goal - max_turn_total, goal, 200)
    ax.plot(x_diag, goal - x_diag, color="0.15", lw=1.35)

    # Reference policy: hold at 20.
    ax.axhline(hold_at, color="0.65", lw=0.9, ls=(0, (4, 6)), zorder=0)

    ax.set_xlim(0, goal)
    ax.set_ylim(0, max_turn_total)
    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Turn Total (k)")
    ax.set_xticks([0, goal])
    ax.set_yticks([max_turn_total])
    ax.set_xticks(np.arange(0, goal + 1, 10), minor=True)
    ax.set_yticks(np.arange(0, max_turn_total + 1, 10), minor=True)
    ax.tick_params(axis="both", which="major", labelsize=8, length=3, color="0.5")
    ax.tick_params(axis="both", which="minor", length=7, color="0.75")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("0.75")
    ax.spines["bottom"].set_color("0.75")
    boundary_handle = Line2D([0], [0], color="0.12", lw=1.6, marker="o", markersize=3)
    reachable_handle = Patch(facecolor="0.8", edgecolor="0.55", alpha=0.45)
    hold_handle = Line2D([0], [0], color="0.65", lw=0.9, ls=(0, (4, 6)))
    ax.legend(
        [boundary_handle, reachable_handle, hold_handle],
        ["Optimal Boundary", "Reachable", "Hold at 20"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.36)
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate the Pig roll/hold boundary figure.")
    parser.add_argument("--opponent-score", type=int, default=30)
    parser.add_argument(
        "--reachable",
        type=Path,
        default=Path(__file__).with_name("reachable_states.npy"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).with_name("pig_policy_cache.pkl"),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Convergence tolerance for value iteration",
    )
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    policy = load_or_compute_policy(script_dir, args.cache, args.epsilon)
    plot_cross_section(
        reachable_path=args.reachable,
        policy=policy,
        opponent_score=args.opponent_score,
    )

if __name__ == "__main__":
    main()
