import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_generation.pendulum import compute_energy_pendulum
from src.data_generation.projectile import compute_energy_projectile
from src.data_generation.utils import scale_trajectories, unscale_trajectories
from src.models.cdn import ConservationDiscoveryNetwork


def _load_cdn(ckpt_path: str, state_dim: int, device: str):
    m = ConservationDiscoveryNetwork(state_dim=state_dim).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    m.eval()
    return m


def _cdn_eval(model, X: np.ndarray, device: str):
    with torch.no_grad():
        y = model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return y


def _fit_affine(x, y):
    # y ≈ a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def hero_equation_comparison(save_path="figures/equation_comparison.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Projectile
    raw_p = np.load("data/projectile/trajectories.npy")
    scaled_p, stats_p = scale_trajectories(raw_p, mode="minmax01")
    flat_p = scaled_p.reshape(-1, 4)
    cdn_p = _load_cdn("models/cdn_projectile_best.pt", 4, device)
    y_p = _cdn_eval(cdn_p, flat_p, device=device)
    raw_states_p = unscale_trajectories(flat_p.reshape(-1, 1, 4), stats_p, mode="minmax01")
    E_p = compute_energy_projectile(raw_states_p)[:, 0]
    a_p, b_p = _fit_affine(E_p, y_p)
    yhat_p = a_p * E_p + b_p
    r2_p = _r2(y_p, yhat_p)

    # Pendulum
    raw_d = np.load("data/pendulum/trajectories.npy")
    scaled_d, stats_d = scale_trajectories(raw_d, mode="standardize")
    flat_d = scaled_d.reshape(-1, 2)
    cdn_d = _load_cdn("models/cdn_pendulum_best.pt", 2, device)
    y_d = _cdn_eval(cdn_d, flat_d, device=device)
    raw_states_d = unscale_trajectories(flat_d.reshape(-1, 1, 2), stats_d, mode="standardize")
    E_d = compute_energy_pendulum(raw_states_d)[:, 0]
    a_d, b_d = _fit_affine(E_d, y_d)
    yhat_d = a_d * E_d + b_d
    r2_d = _r2(y_d, yhat_d)

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(E_p, y_p, s=3, alpha=0.2)
    xs = np.linspace(np.min(E_p), np.max(E_p), 200)
    ax.plot(xs, a_p * xs + b_p, "r", linewidth=2)
    ax.set_title(f"Projectile: CDN ≈ a·E + b (R²={r2_p:.3f})")
    ax.set_xlabel("Analytical energy E")
    ax.set_ylabel("CDN output f(s)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(E_d, y_d, s=3, alpha=0.2)
    xs = np.linspace(np.min(E_d), np.max(E_d), 200)
    ax.plot(xs, a_d * xs + b_d, "r", linewidth=2)
    ax.set_title(f"Pendulum: CDN ≈ a·E + b (R²={r2_d:.3f})")
    ax.set_xlabel("Analytical energy E")
    ax.set_ylabel("CDN output f(s)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print("Saved", save_path)


if __name__ == "__main__":
    hero_equation_comparison()

