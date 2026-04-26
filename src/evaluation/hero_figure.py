"""
Generate the hero figure: discovered vs known equations for all environments.
"""

import os

import matplotlib
import matplotlib.pyplot as plt


def create_hero_figure(
    discovered_equations, known_equations, env_names, coefficient_errors, save_dir="figures"
):
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "text.usetex": False,
        }
    )

    n_envs = len(env_names)
    fig, ax = plt.subplots(figsize=(14, 2 + 1.5 * n_envs))
    ax.axis("off")

    headers = ["Environment", "Discovered Equation", "Known Equation", "Coeff. Error"]
    cell_data = []
    for i in range(n_envs):
        cell_data.append(
            [
                env_names[i].replace("_", " ").title(),
                discovered_equations[i],
                known_equations[i],
                f"{float(coefficient_errors[i]):.3f}%",
            ]
        )

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.35, 0.35, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#1a3a5c")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_envs + 1):
        for j in range(len(headers)):
            cell = table[i, j]
            if j in (1, 2):
                cell.set_text_props(fontfamily="monospace", fontsize=9)
            if i % 2 == 0:
                cell.set_facecolor("#f0f0f0")

    fig.suptitle(
        "Physics Equations Discovered from Trajectory Data",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "hero_equation_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero figure: {path}")

