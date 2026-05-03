"""2d projectile trajectories. state=[x,y,vx,vy], energy = 0.5*(vx^2+vy^2) + g*y (m=1)."""

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
    """vectorized analytic generator. returns (n_trajectories, n_timesteps, 4)."""
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
        print(f"  saved {n_trajectories} projectile trajectories -> {save_dir}")

    return trajectories


def compute_energy_projectile(trajectories: np.ndarray, g: float = 9.81, m: float = 1.0):
    """analytic energy 0.5*m*(vx^2+vy^2) + m*g*y. trajectories: (N,T,4) -> (N,T)."""
    y = trajectories[:, :, 1]
    vx = trajectories[:, :, 2]
    vy = trajectories[:, :, 3]
    kinetic = 0.5 * m * (vx**2 + vy**2)
    potential = m * g * y
    return kinetic + potential


if __name__ == "__main__":
    generate_projectile_data(save_dir="data/projectile")
