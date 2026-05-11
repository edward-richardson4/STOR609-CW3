import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

from pig_value_iteration import optimal_pig_value_iteration

def build_policy_volume(policy, goal: int = 100):
    """
    Builds a 3D array where:
        1 = rolling is optimal
        0 = holding is optimal

    Axes are:
        i = player score
        j = opponent score
        k = turn total
    """
    volume = np.zeros((goal, goal, goal), dtype=float)
    for i in range(goal):
        for j in range(goal):
            for k in range(goal):
                if i + k >= goal:
                    volume[i, j, k] = 0  # holding wins immediately
                else:
                    volume[i, j, k] = 1 if policy[(i, j, k)] == "roll" else 0
    return volume


def plot_roll_hold_isosurface(goal: int = 100):
    """
    Plots the full 3D roll/hold boundary.
    This is closer to the paper's Figure 3 than a simple height-map surface.
    """

    policy, value_func = optimal_pig_value_iteration(goal=goal)
    volume = build_policy_volume(policy, goal=goal)

    # Extract the surface separating roll states from hold states.
    verts, faces, normals, values = marching_cubes(
        volume,
        level=0.5
    )

    fig = plt.figure(figsize=(8, 12))
    views = [
        (35, -128),
        (18, -72),
    ]

    for index, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 1, index, projection="3d")
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            cmap="Greys",
            linewidth=0.08,
            edgecolor="0.35",
            alpha=1.0,
            shade=True
        )
        
        ax.set_xlabel("Player 1 Score (i)")
        ax.set_ylabel("Player 2 Score (j)")
        ax.set_zlabel("Turn Total (k)")
        ax.set_xlim(0, goal)
        ax.set_ylim(0, goal)
        ax.set_zlim(0, goal)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_roll_hold_isosurface(goal=100)
