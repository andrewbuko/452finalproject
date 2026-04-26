"""
Projectile trajectory generation.
Generates 2D projectile motion trajectories analytically (no simulator needed).
State vector: [x, y, vx, vy] where x,y are position and vx,vy are velocity.
Physics:
  x(t) = vx0 * t
  y(t) = vy0 * t - 0.5 * g * t^2
  vx(t) = vx0
  vy(t) = vy0 - g * t
Conserved quantity (total energy):
  E = 0.5 * (vx^2 + vy^2) + g * y   (with m=1)
"""

import json
import os

import numpy as np


def generate_projectile_data(
    n_trajectories=1_000_000,
    n_timesteps=200,
    dt=0.005,
    g=9.81,
    m=1.0,
    v0_range=(1.0, 30.0),
    angle_range_deg=(5.0, 85.0),
    seed=42,
    save_dir=None,
):
    """
    Generate projectile trajectories analytically. Fully vectorized.
    Args:
      n_trajectories: number of trajectories to generate
      n_timesteps: timesteps per trajectory
      dt: time step in seconds
      g: gravitational acceleration
      m: mass (kept at 1.0 for simplicity)
      v0_range: (min, max) initial speed
      angle_range_deg: (min, max) launch angle in degrees
      seed: random seed for reproducibility
      save_dir: if provided, save data and params here
    Returns:
      trajectories: (n_trajectories, n_timesteps, 4) array [x, y, vx, vy]
    """
    rng = np.random.RandomState(seed)
    v0 = rng.uniform(v0_range[0], v0_range[1], size=n_trajectories)
    angles = np.radians(rng.uniform(angle_range_deg[0], angle_range_deg[1], size=n_trajectories))
    vx0 = v0 * np.cos(angles)
    vy0 = v0 * np.sin(angles)

    t = np.arange(n_timesteps, dtype=np.float32) * np.float32(dt)

    trajectories = np.zeros((n_trajectories, n_timesteps, 4), dtype=np.float32)
    trajectories[:, :, 0] = vx0[:, None] * t[None, :]
    trajectories[:, :, 1] = vy0[:, None] * t[None, :] - 0.5 * g * t[None, :] ** 2
    trajectories[:, :, 2] = np.repeat(vx0[:, None], n_timesteps, axis=1)
    trajectories[:, :, 3] = vy0[:, None] - g * t[None, :]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "trajectories.npy"), trajectories)
        params = {
            "environment": "projectile",
            "g": g,
            "m": m,
            "dt": dt,
            "n_trajectories": int(n_trajectories),
            "n_timesteps": int(n_timesteps),
            "v0_range": list(v0_range),
            "angle_range_deg": list(angle_range_deg),
            "seed": int(seed),
            "state_dim": 4,
            "state_labels": ["x", "y", "vx", "vy"],
            "known_energy": "0.5*(vx^2 + vy^2) + 9.81*y",
        }
        with open(os.path.join(save_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"Saved {n_trajectories} projectile trajectories to {save_dir}")

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
    generate_projectile_data(save_dir="data/projectile")
