"""
Spring-mass system (simple harmonic oscillator).
State: [x, v] - position and velocity
ODE: dx/dt = v, dv/dt = -(k/m)*x
Energy: E = 0.5*k*x^2 + 0.5*m*v^2
With k=10, m=1: E = 5.0*x^2 + 0.5*v^2
"""

import json
import os

import numpy as np


def generate_spring_mass_data(
    n_trajectories=500_000,
    n_timesteps=200,
    dt=0.005,
    k=10.0,
    m=1.0,
    x0_range=(-2.0, 2.0),
    v0_range=(-5.0, 5.0),
    seed=42,
    save_dir=None,
):
    rng = np.random.RandomState(seed)
    omega = np.sqrt(k / m)
    t = np.arange(n_timesteps, dtype=np.float32) * np.float32(dt)
    x0 = rng.uniform(x0_range[0], x0_range[1], size=n_trajectories)
    v0 = rng.uniform(v0_range[0], v0_range[1], size=n_trajectories)

    trajectories = np.zeros((n_trajectories, n_timesteps, 2), dtype=np.float32)
    trajectories[:, :, 0] = x0[:, None] * np.cos(omega * t[None, :]) + (v0[:, None] / omega) * np.sin(
        omega * t[None, :]
    )
    trajectories[:, :, 1] = -x0[:, None] * omega * np.sin(omega * t[None, :]) + v0[:, None] * np.cos(
        omega * t[None, :]
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "trajectories.npy"), trajectories)
        params = {
            "environment": "spring_mass",
            "k": k,
            "m": m,
            "dt": dt,
            "n_trajectories": int(n_trajectories),
            "n_timesteps": int(n_timesteps),
            "state_dim": 2,
            "state_labels": ["x", "v"],
            "known_energy": f"{0.5*k}*x^2 + {0.5*m}*v^2",
            "seed": int(seed),
        }
        with open(os.path.join(save_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"Saved {n_trajectories} spring-mass trajectories to {save_dir}")

    return trajectories


def compute_energy_spring(trajectories, k=10.0, m=1.0):
    x = trajectories[:, :, 0]
    v = trajectories[:, :, 1]
    return 0.5 * k * x**2 + 0.5 * m * v**2


if __name__ == "__main__":
    generate_spring_mass_data(save_dir="data/spring_mass")

