import json
import os

import numpy as np


def generate_projectile_data(
    n_trajectories: int = 10_000,
    n_timesteps: int = 100,
    dt: float = 0.01,
    g: float = 9.81,
    m: float = 1.0,
    vx0_fixed=None,
    vx0_range=(5.0, 20.0),
    vy0_range=(5.0, 20.0),
    seed: int = 42,
):
    """
    Generate 2D projectile trajectories analytically.

    Each trajectory is (n_timesteps, 4): [x, y, vx, vy].
    Physics:
      x(t)  = vx0 * t
      y(t)  = vy0 * t - 0.5 * g * t^2
      vx(t) = vx0
      vy(t) = vy0 - g * t
    """
    _ = m  # kept for symmetry with energy function signature
    rng = np.random.RandomState(seed)

    if vx0_fixed is not None:
        # We fix vx0 across trajectories so the easiest conserved signal (vx) can't
        # explain the between-trajectory variance; this biases the CDN toward energy.
        vx0 = np.full((n_trajectories,), float(vx0_fixed), dtype=np.float64)
    else:
        # For equation identification, vx must vary so vx^2 is identifiable.
        vx0 = rng.uniform(vx0_range[0], vx0_range[1], size=n_trajectories).astype(np.float64)
    vy0 = rng.uniform(vy0_range[0], vy0_range[1], size=n_trajectories)

    t = np.arange(n_timesteps) * dt

    trajectories = np.zeros((n_trajectories, n_timesteps, 4), dtype=np.float64)
    trajectories[:, :, 0] = vx0[:, None] * t[None, :]
    trajectories[:, :, 1] = vy0[:, None] * t[None, :] - 0.5 * g * t[None, :] ** 2
    trajectories[:, :, 2] = np.repeat(vx0[:, None], n_timesteps, axis=1)
    trajectories[:, :, 3] = vy0[:, None] - g * t[None, :]

    return trajectories


def compute_energy_projectile(trajectories: np.ndarray, g: float = 9.81, m: float = 1.0):
    """
    Analytical total energy:
      E = 0.5*m*(vx^2 + vy^2) + m*g*y

    Args:
      trajectories: (N, T, 4) with [x, y, vx, vy]
    Returns:
      energy: (N, T)
    """
    y = trajectories[:, :, 1]
    vx = trajectories[:, :, 2]
    vy = trajectories[:, :, 3]
    kinetic = 0.5 * m * (vx**2 + vy**2)
    potential = m * g * y
    return kinetic + potential


if __name__ == "__main__":
    os.makedirs("data/projectile", exist_ok=True)
    print("Generating projectile trajectories...")
    trajs = generate_projectile_data()
    np.save("data/projectile/trajectories.npy", trajs)

    params = {
        "g": 9.81,
        "m": 1.0,
        "dt": 0.01,
        "n_trajectories": 10_000,
        "n_timesteps": 100,
        "vx0_fixed": 10.0,
        "vy0_range": [5.0, 20.0],
        "state_dim": 4,
        "state_labels": ["x", "y", "vx", "vy"],
    }
    with open("data/projectile/params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    E = compute_energy_projectile(trajs)
    within_std = E.std(axis=1)
    print(f"Mean within-trajectory energy std: {within_std.mean():.2e}")
    print(f"Max within-trajectory energy std: {within_std.max():.2e}")

    across_std = E[:, 0].std()
    print(f"Across-trajectory energy std: {across_std:.4f}")
    print(f"Dataset shape: {trajs.shape}")
    print(f"Energy range: [{E.min():.2f}, {E.max():.2f}]")
    print("PASS" if within_std.max() < 1e-10 else "FAIL - check equations")
