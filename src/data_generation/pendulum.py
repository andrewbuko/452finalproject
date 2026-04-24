import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


def pendulum_ode(t, state, g: float, l: float):
    theta, omega = state
    return [omega, -(g / l) * np.sin(theta)]


def generate_pendulum_data(
    n_traj: int = 10_000,
    T: int = 100,
    dt: float = 0.01,
    g: float = 9.81,
    l: float = 1.0,
    theta0_range=(-np.pi / 2, np.pi / 2),
    omega0_range=(-2.0, 2.0),
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    t_end = (T - 1) * dt
    t_eval = np.linspace(0, t_end, T)

    trajs = np.zeros((n_traj, T, 2), dtype=np.float64)
    for i in tqdm(range(n_traj), desc="Pendulum"):
        th0 = rng.uniform(*theta0_range)
        om0 = rng.uniform(*omega0_range)
        sol = solve_ivp(
            pendulum_ode,
            (0, t_end),
            [th0, om0],
            args=(g, l),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        trajs[i] = sol.y.T

    return trajs


def compute_energy_pendulum(trajs: np.ndarray, g: float = 9.81, l: float = 1.0, m: float = 1.0):
    theta = trajs[:, :, 0]
    omega = trajs[:, :, 1]
    return 0.5 * m * l**2 * omega**2 - m * g * l * np.cos(theta)


if __name__ == "__main__":
    os.makedirs("data/pendulum", exist_ok=True)
    print("Generating pendulum trajectories (~2-3 min)...")
    trajs = generate_pendulum_data()
    np.save("data/pendulum/trajectories.npy", trajs)

    params = {
        "g": 9.81,
        "l": 1.0,
        "m": 1.0,
        "dt": 0.01,
        "n_trajectories": 10_000,
        "n_timesteps": 100,
        "state_dim": 2,
        "state_labels": ["theta", "omega"],
    }
    with open("data/pendulum/params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    E = compute_energy_pendulum(trajs)
    within_std = E.std(axis=1)
    print(f"Mean energy std: {within_std.mean():.2e}")
    print(f"Max energy std: {within_std.max():.2e}")
    print(f"Dataset shape: {trajs.shape}")
    print("PASS" if within_std.max() < 1e-6 else "FAIL")
