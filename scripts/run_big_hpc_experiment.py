"""
Big GPU experiment for HPC (H200-class).

Key properties:
- Generates trajectories ON THE FLY (no huge .npy files).
- Uses large CDN (1024 hidden, 6 layers) + AMP.
- Keeps GPU saturated: big batch size and many steps/epoch.

Run locally only if you have a big GPU; intended for cluster.
"""

import os

import torch

from src.data_generation.pendulum_gpu import simulate_pendulum_symplectic
from src.training.train_cdn_streaming import StreamingCDNConfig, train_cdn_streaming


def make_pendulum_batch(
    batch_size: int,
    steps: int,
    dt: float,
    device: str,
    theta_range=(-3.14159, 3.14159),
    omega_range=(-8.0, 8.0),
):
    def _fn():
        theta0 = (theta_range[0] + (theta_range[1] - theta_range[0]) * torch.rand(batch_size, device=device))
        omega0 = (omega_range[0] + (omega_range[1] - omega_range[0]) * torch.rand(batch_size, device=device))
        traj = simulate_pendulum_symplectic(theta0, omega0, steps=steps, dt=dt, g_over_l=9.81)
        return traj, None

    return _fn


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("This script is intended for GPU (cuda).")

    os.makedirs("models", exist_ok=True)

    cfg = StreamingCDNConfig(
        env_name="pendulum_big",
        state_dim=2,
        hidden_dim=1024,
        n_layers=6,
        lr=3e-4,
        epochs=5,
        steps_per_epoch=3000,
        lambda_var=0.1,
        epsilon=1.0,
        var_reg="softplus",
        lambda_scale=0.1,
        target_mean=0.0,
        target_std=1.0,
        std_min=0.8,
        std_max=1.2,
        grad_clip=1.0,
        log_every=50,
        save_dir="models",
        device=device,
        amp=True,
    )

    make_batch = make_pendulum_batch(
        batch_size=8192,  # H200 can handle this; adjust if OOM
        steps=512,
        dt=0.01,
        device=device,
    )

    train_cdn_streaming(make_batch, cfg)

