from __future__ import annotations

import torch


@torch.no_grad()
def simulate_pendulum_symplectic(
    theta0: torch.Tensor,
    omega0: torch.Tensor,
    steps: int,
    dt: float,
    g_over_l: float = 9.81,
) -> torch.Tensor:
    """
    GPU-friendly pendulum simulator using symplectic Euler:
      omega_{t+1} = omega_t - (g/l) * sin(theta_t) * dt
      theta_{t+1} = theta_t + omega_{t+1} * dt

    Args:
      theta0, omega0: (B,) tensors (same device/dtype)
      steps: number of time steps (T)
    Returns:
      traj: (B, T, 2) with [theta, omega]
    """
    if theta0.shape != omega0.shape:
        raise ValueError("theta0 and omega0 must have same shape")
    if steps < 2:
        raise ValueError("steps must be >= 2")

    B = theta0.shape[0]
    traj = torch.empty((B, steps, 2), device=theta0.device, dtype=theta0.dtype)
    theta = theta0.clone()
    omega = omega0.clone()

    for t in range(steps):
        traj[:, t, 0] = theta
        traj[:, t, 1] = omega
        if t == steps - 1:
            break
        omega = omega - g_over_l * torch.sin(theta) * dt
        theta = theta + omega * dt

    return traj

