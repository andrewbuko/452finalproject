from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.polynomial_cdn import PolynomialConservationModel
from src.training.train_cdn import cdn_loss


def _compute_feature_scaler(
    model: PolynomialConservationModel,
    trajectories_raw_np: np.ndarray,
    n_samples: int = 200_000,
    seed: int = 0,
):
    """
    Compute mean/std of the feature library on a subset of raw states.
    This ensures x, y, and cross-terms are on comparable scales during training.
    """
    flat = trajectories_raw_np.reshape(-1, trajectories_raw_np.shape[-1])
    n = min(n_samples, flat.shape[0])
    idx = np.random.RandomState(seed).choice(flat.shape[0], size=n, replace=False)
    X = flat[idx]
    with torch.no_grad():
        phi = model.features(torch.tensor(X, dtype=torch.float32)).cpu().numpy().astype(np.float64)
    mean = phi.mean(axis=0)
    std = phi.std(axis=0)
    return mean, std


@dataclass
class PolyTrainConfig:
    env_name: str  # 'projectile' | 'pendulum'
    lr: float = 1e-3
    epochs: int = 400
    batch_size: int = 512
    lambda_var: float = 0.1
    epsilon: float = 1.0
    var_reg: str = "softplus"
    # Raw-unit alignment: force f(s0) to match analytical energy in physical units.
    # This resolves scale/offset ambiguity and makes coefficients interpretable.
    lambda_energy: float = 1.0
    # Sparsity: encourages dropping cross-terms.
    l1_weight: float = 5e-3
    grad_clip: float = 1.0
    log_every: int = 50
    init_scale: float = 1e-2
    scaler_samples: int = 200_000
    save_dir: str = "models"
    device: str | None = None


def train_polynomial_cdn(trajectories_raw_np: np.ndarray, cfg: PolyTrainConfig, energy0_np: np.ndarray | None = None):
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = cfg.device

    trajs_tensor = torch.tensor(trajectories_raw_np, dtype=torch.float32)
    if energy0_np is None:
        dataset = TensorDataset(trajs_tensor)
        e0_scale = None
    else:
        e0_tensor = torch.tensor(energy0_np, dtype=torch.float32)
        dataset = TensorDataset(trajs_tensor, e0_tensor)
        # scale energy alignment so it doesn't dominate purely due to units
        e0_scale = float(np.std(energy0_np)) if np.isfinite(np.std(energy0_np)) else 1.0
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = PolynomialConservationModel(cfg.env_name).to(device)
    with torch.no_grad():
        model.weights.normal_(mean=0.0, std=cfg.init_scale)
    # Normalize feature library so x/y and cross-terms are comparable during optimization.
    phi_mean, phi_std = _compute_feature_scaler(model, trajectories_raw_np, n_samples=cfg.scaler_samples, seed=0)
    model.set_feature_scaler(phi_mean, phi_std, std_floor=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, threshold=1e-6)

    best = float("inf")
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt = os.path.join(cfg.save_dir, f"poly_cdn_{cfg.env_name}_best.pt")

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_time_std = 0.0
        n = 0
        for batch in loader:
            batch_traj = batch[0].to(device)
            batch_e0 = None if energy0_np is None else batch[1].to(device)
            opt.zero_grad(set_to_none=True)

            # Base conservation loss (consistency + variance regularizer)
            loss, cons, var, _scale = cdn_loss(
                model=model,  # works: model(states)->(B*T,)
                trajectories=batch_traj,
                energy0=None,
                lambda_var=cfg.lambda_var,
                epsilon=cfg.epsilon,
                var_reg=cfg.var_reg,
                lambda_scale=0.0,
                lambda_align=0.0,
            )

            # Raw-unit alignment to analytical energy at t=0 (physical units)
            if batch_e0 is not None and cfg.lambda_energy and cfg.lambda_energy > 0:
                f0 = model(batch_traj[:, 0, :])
                denom = (e0_scale or float(batch_e0.std(unbiased=False).detach().cpu().item()) or 1.0) ** 2
                loss = loss + cfg.lambda_energy * (torch.mean((f0 - batch_e0) ** 2) / (denom + 1e-12))

            # LASSO/L1 sparsity on *physical* coefficients to drop cross-terms.
            if cfg.l1_weight and cfg.l1_weight > 0:
                # compute physical coeffs cheaply: a = w/std, intercept adjusted; apply L1 to non-constant terms
                w = model.weights
                s = model.phi_std
                l1 = torch.sum(torch.abs(w[1:] / (s[1:] + 1e-12)))
                loss = loss + cfg.l1_weight * l1

            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.detach().cpu().item())

            # True conservation metric: per-trajectory std over time of f(s).
            # (Lower is better; this is independent of cross-trajectory variance at t=0.)
            with torch.no_grad():
                B, T, D = batch_traj.shape
                f_bt = model(batch_traj.reshape(B * T, D)).reshape(B, T)
                epoch_time_std += float(f_bt.std(dim=1, unbiased=False).mean().detach().cpu().item())
            n += 1

        avg = epoch_loss / max(1, n)
        avg_time_std = epoch_time_std / max(1, n)
        sched.step(avg)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), ckpt)

        if ((epoch + 1) % max(1, cfg.log_every) == 0) or epoch == 0:
            print(
                f"[poly {cfg.env_name}] epoch {epoch+1:4d}/{cfg.epochs} "
                f"loss={avg:.6f} (best={best:.6f}) cons={cons:.6f} "
                f"var_across_traj_t0={var:.4f} mean_std_over_time={avg_time_std:.6f}"
            )

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model

