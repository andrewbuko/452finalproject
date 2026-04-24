import os

import matplotlib.pyplot as plt
import numpy as np

from src.data_generation.pendulum import compute_energy_pendulum
from src.data_generation.projectile import compute_energy_projectile


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 150,
        }
    )
    os.makedirs("figures", exist_ok=True)

    # ── Projectile ──
    proj = np.load("data/projectile/trajectories.npy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(15):
        axes[0].plot(proj[i, :, 0], proj[i, :, 1], alpha=0.5, linewidth=0.8)
    axes[0].set_xlabel("x position")
    axes[0].set_ylabel("y position")
    axes[0].set_title("Sample Projectile Trajectories")

    E = compute_energy_projectile(proj[:5])
    for i in range(5):
        axes[1].plot(E[i], alpha=0.7)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Total Energy")
    axes[1].set_title("Energy Conservation Check")

    fig.tight_layout()
    fig.savefig("figures/data_sanity_projectile.png")
    plt.close(fig)
    print("Saved figures/data_sanity_projectile.png")

    # ── Pendulum ──
    pend = np.load("data/pendulum/trajectories.npy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(15):
        axes[0].plot(pend[i, :, 0], pend[i, :, 1], alpha=0.5, linewidth=0.8)
    axes[0].set_xlabel("theta (rad)")
    axes[0].set_ylabel("omega (rad/s)")
    axes[0].set_title("Pendulum Phase Space")

    E_p = compute_energy_pendulum(pend[:5])
    for i in range(5):
        axes[1].plot(E_p[i], alpha=0.7)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Total Energy")
    axes[1].set_title("Energy Conservation Check")

    fig.tight_layout()
    fig.savefig("figures/data_sanity_pendulum.png")
    plt.close(fig)
    print("Saved figures/data_sanity_pendulum.png")


if __name__ == "__main__":
    main()
