"""
Pendulum trajectory generation.
State vector: [theta, omega] where theta is angle and omega is angular velocity.
ODE:
  dtheta/dt = omega
  domega/dt = -(g/l) * sin(theta)
Conserved quantity (Hamiltonian):
  H = 0.5*m*l^2*omega^2 - m*g*l*cos(theta)
With m=1, l=1: H = 0.5*omega^2 - 9.81*cos(theta)
"""

import json
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


def _pendulum_ode(t, state, g, l):
    theta, omega = state
    return [omega, -(g / l) * np.sin(theta)]


def _solve_single(args):
    """Solve one pendulum trajectory. For parallel execution."""
    theta0, omega0, g, l, t_eval, rtol, atol = args
    sol = solve_ivp(
        _pendulum_ode,
        (0, float(t_eval[-1])),
        [float(theta0), float(omega0)],
        args=(float(g), float(l)),
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success or np.any(np.isnan(sol.y)):
        raise RuntimeError("Pendulum solve failed")
    return sol.y.T.astype(np.float32)


def generate_pendulum_data(
    n_trajectories=500_000,
    n_timesteps=200,
    dt=0.005,
    g=9.81,
    l=1.0,
    m=1.0,
    theta0_range=(-np.pi / 2, np.pi / 2),
    omega0_range=(-3.0, 3.0),
    seed=42,
    save_dir=None,
    n_workers=None,
    rtol=1e-12,
    atol=1e-14,
):
    """
    Generate pendulum trajectories via RK45 integration.
    Uses tight tolerances to ensure energy conservation in the generated data.
    """
    rng = np.random.RandomState(seed)
    t_end = (n_timesteps - 1) * dt
    t_eval = np.linspace(0, t_end, n_timesteps)

    theta0s = rng.uniform(theta0_range[0], theta0_range[1], size=n_trajectories)
    omega0s = rng.uniform(omega0_range[0], omega0_range[1], size=n_trajectories)

    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    n_workers = max(1, int(n_workers))

    # IMPORTANT: don't materialize a giant Python list at H200 scale.
    args_iter = ((theta0s[i], omega0s[i], g, l, t_eval, rtol, atol) for i in range(n_trajectories))

    if n_workers == 1:
        results = []
        for a in tqdm(args_iter, total=n_trajectories, desc="Pendulum trajectories"):
            results.append(_solve_single(a))
    else:
        print(f"Generating {n_trajectories} pendulum trajectories with {n_workers} workers...")
        with Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_solve_single, args_iter, chunksize=1000),
                    total=n_trajectories,
                    desc="Pendulum trajectories",
                )
            )

    trajectories = np.array(results, dtype=np.float32)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "trajectories.npy"), trajectories)
        params = {
            "environment": "pendulum",
            "g": g,
            "l": l,
            "m": m,
            "dt": dt,
            "n_trajectories": int(n_trajectories),
            "n_timesteps": int(n_timesteps),
            "theta0_range": list(theta0_range),
            "omega0_range": list(omega0_range),
            "seed": int(seed),
            "state_dim": 2,
            "state_labels": ["theta", "omega"],
            "known_energy": "0.5*omega^2 - 9.81*cos(theta)",
        }
        with open(os.path.join(save_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"Saved {n_trajectories} pendulum trajectories to {save_dir}")

    return trajectories


def compute_energy_pendulum(trajectories, g=9.81, l=1.0, m=1.0):
    theta = trajectories[:, :, 0]
    omega = trajectories[:, :, 1]
    return 0.5 * m * l**2 * omega**2 - m * g * l * np.cos(theta)


if __name__ == "__main__":
    generate_pendulum_data(save_dir="data/pendulum", n_workers=1, n_trajectories=10_000)
