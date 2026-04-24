import numpy as np
import torch

from src.data_generation.utils import scale_trajectories
from src.evaluation.probe_cdn import (
    plot_probes,
    probe_pendulum,
    probe_projectile,
)
from src.models.cdn import ConservationDiscoveryNetwork


def load_cdn(env_name: str, state_dim: int, ckpt_path: str, device: str):
    model = ConservationDiscoveryNetwork(state_dim=state_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def run_projectile(device: str):
    raw = np.load("data/projectile/trajectories.npy")
    scaled, _stats = scale_trajectories(raw, mode="minmax01")

    cdn = load_cdn("projectile", 4, "models/cdn_projectile_best.pt", device)
    probes, fits, summary = probe_projectile(cdn, scaled, device=device)
    plot_probes(probes, fits, env_name="projectile", save_dir="figures")

    print("\n[projectile] probe summary")
    print("  x constant ~", summary.x_const)
    print("  y slope ~", summary.y_slope)
    print("  vx quad coeff ~", summary.vx_quad)
    print("  vy quad coeff ~", summary.vy_quad)
    print("  ratio (y_slope / vx_quad) ~", summary.ratio_slope_over_quad)
    print("  expected ~ 2*g ~= 19.62")


def run_pendulum(device: str):
    raw = np.load("data/pendulum/trajectories.npy")
    scaled, _stats = scale_trajectories(raw, mode="standardize")

    cdn = load_cdn("pendulum", 2, "models/cdn_pendulum_best.pt", device)
    probes, fits, summary = probe_pendulum(cdn, scaled, device=device)
    plot_probes(probes, fits, env_name="pendulum", save_dir="figures")

    print("\n[pendulum] probe summary")
    print("  omega quad coeff ~", summary.omega_quad)
    print("  theta cos amplitude ~", summary.theta_cos_amp)
    print("  ratio (theta_amp / omega_quad) ~", summary.ratio_amp_over_quad)
    print("  expected ~ -(g/l) / (0.5*l^2) for energy up to scale; use ratios in report/README")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    run_projectile(device)
    run_pendulum(device)

