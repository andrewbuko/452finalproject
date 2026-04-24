from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr


def validate_cdn(
    model: torch.nn.Module,
    trajs_np: np.ndarray,
    energy_fn_np: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    n_samples: int = 5000,
):
    """
    Scatter plot of learned f(s) vs analytical energy E(s) over random states.

    Args:
      model: CDN, trained on the same scaling as trajs_np
      trajs_np: (N, T, D)
      energy_fn_np: function(trajs_np) -> (N, T)
    Returns:
      r_squared: float
    """
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})
    os.makedirs("figures", exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    N, T, D = trajs_np.shape
    all_states = trajs_np.reshape(-1, D)

    rng = np.random.RandomState(0)
    n_samples = min(n_samples, len(all_states))
    idx = rng.choice(len(all_states), size=n_samples, replace=False)
    states_sub = all_states[idx]

    with torch.no_grad():
        f_learned = (
            model(torch.tensor(states_sub, dtype=torch.float32, device=device))
            .detach()
            .cpu()
            .numpy()
        )

    all_energy = energy_fn_np(trajs_np).reshape(-1)
    e_analytical = all_energy[idx]

    r, _ = pearsonr(f_learned, e_analytical)
    r2 = float(r**2)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(e_analytical, f_learned, alpha=0.06, s=2, color="tab:blue", rasterized=True)
    z = np.polyfit(e_analytical, f_learned, 1)
    x_line = np.linspace(e_analytical.min(), e_analytical.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5, label=f"R$^2$ = {r2:.4f}")
    ax.set_xlabel("Analytical Energy E(s)")
    ax.set_ylabel("Learned Conserved Quantity f(s)")
    ax.set_title(f"CDN Validation — {env_name.capitalize()}")
    ax.legend(fontsize=11, loc="upper left")
    fig.tight_layout()

    out_path = f"figures/cdn_validation_{env_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[{env_name}] Pearson r = {r:.4f}, R^2 = {r2:.4f}")
    print(f"Saved {out_path}")
    return r2

