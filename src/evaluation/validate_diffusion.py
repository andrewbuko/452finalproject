from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import numpy as np

from src.models.diffusion_transition import DiffusionTransitionModel
from src.training.train_diffusion import rollout_diffusion


def evaluate_diffusion_rollout(
    model: DiffusionTransitionModel,
    trajs_raw: np.ndarray,
    energy_fn_np: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    stats: Dict[str, np.ndarray],
    n_rollouts: int = 256,
    device: Optional[str] = None,
    save_dir: str = "figures",
) -> Dict[str, float]:
    """
    Compare diffusion rollouts vs ground truth on a subset of trajectories.

    Metrics:
      - rollout_mse: mean squared error over all timesteps/dims
      - energy_std_mean_true: mean over trajectories of std_t(E)
      - energy_std_mean_gen: same for generated rollouts
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.RandomState(0)

    N, T, D = trajs_raw.shape
    n = min(int(n_rollouts), int(N))
    idx = rng.choice(N, size=n, replace=False)
    gt = trajs_raw[idx].astype(np.float32, copy=False)

    s0 = gt[:, 0, :]
    gen = rollout_diffusion(model, s0_raw=s0, T=T, stats=stats, device=device)

    rollout_mse = float(np.mean((gen - gt) ** 2))

    E_true = energy_fn_np(gt)  # (n, T)
    E_gen = energy_fn_np(gen)
    energy_std_mean_true = float(np.std(E_true, axis=1).mean())
    energy_std_mean_gen = float(np.std(E_gen, axis=1).mean())

    out = {
        "rollout_mse": rollout_mse,
        "energy_std_mean_true": energy_std_mean_true,
        "energy_std_mean_gen": energy_std_mean_gen,
    }

    # Persist a small JSON for the experiment_results payload
    try:
        import json

        with open(os.path.join(save_dir, f"diffusion_metrics_{env_name}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass

    print(
        f"[diffusion {env_name}] rollout_mse={rollout_mse:.6e} "
        f"energy_std_mean(gen)={energy_std_mean_gen:.3e} true={energy_std_mean_true:.3e}"
    )
    return out

