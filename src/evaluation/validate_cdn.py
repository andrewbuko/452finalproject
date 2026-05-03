from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def _eval_model_on_states(model: torch.nn.Module, states: np.ndarray, device: torch.device, batch_size: int = 16384):
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            x = torch.tensor(states[i : i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(x).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def validate_cdn(
    model: torch.nn.Module,
    trajs_np: np.ndarray,
    energy_fn_np: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    n_samples: int = 5000,
    save_dir: str = "figures",
):
    """scatter learned f(s) vs analytical energy and report r2/spearman/conservation std."""
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})
    os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    N, T, D = trajs_np.shape
    all_states = trajs_np.reshape(-1, D)

    rng = np.random.RandomState(0)
    n_samples = min(n_samples, len(all_states))
    idx = rng.choice(len(all_states), size=n_samples, replace=False)
    states_sub = all_states[idx]

    f_learned = _eval_model_on_states(model, states_sub, device=device)

    all_energy = energy_fn_np(trajs_np).reshape(-1)
    e_analytical = all_energy[idx]

    r, _ = pearsonr(f_learned, e_analytical)
    r2 = float(r**2)
    sr, _ = spearmanr(f_learned, e_analytical)
    spearman = float(sr) if np.isfinite(sr) else 0.0

    # conservation: mean over trajectories of std_t(f(s_t)). lower is better
    f_traj = _eval_model_on_states(model, trajs_np.reshape(-1, D), device=device).reshape(N, T)
    invariant_std_mean = float(np.std(f_traj, axis=1).mean())
    energy_std_mean = float(np.std(energy_fn_np(trajs_np), axis=1).mean())

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(e_analytical, f_learned, alpha=0.06, s=2, color="tab:blue", rasterized=True)
    z = np.polyfit(e_analytical, f_learned, 1)
    x_line = np.linspace(e_analytical.min(), e_analytical.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5, label=f"R$^2$={r2:.4f}  Spearman={spearman:.3f}")
    ax.set_xlabel("Analytical Energy E(s)")
    ax.set_ylabel("Learned Conserved Quantity f(s)")
    ax.set_title(f"CDN Validation — {env_name.capitalize()}")
    ax.legend(fontsize=11, loc="upper left")
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"cdn_validation_{env_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  {env_name}: r2={r2:.4f}  spearman={spearman:.4f}  inv_std={invariant_std_mean:.3e}")
    return {
        "r_squared": float(r2),
        "pearson": float(r),
        "spearman": float(spearman),
        "invariant_std_mean": float(invariant_std_mean),
        "energy_std_mean_true": float(energy_std_mean),
    }


def validate_conservation_model(
    model,
    trajs_np,
    energy_fn_np,
    env_name,
    model_name="CDN",
    save_dir="figures",
    n_samples=10000,
    device=None,
):
    """thin wrapper around validate_cdn that tags the output with model_name."""
    _ = save_dir
    if device is not None:
        model = model.to(device)
    out = validate_cdn(model, trajs_np, energy_fn_np, env_name, n_samples=n_samples, save_dir=save_dir)
    out["model_name"] = str(model_name)
    return out

