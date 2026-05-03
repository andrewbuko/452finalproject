from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.cdn import ConservationDiscoveryNetwork


def conservation_loss(model, trajectories, lambda_var=0.1, epsilon=1.0):
    """temporal consistency loss with a hinge variance floor to avoid f(s)=const."""
    B, T, D = trajectories.shape
    states = trajectories.reshape(B * T, D)
    f_vals = model(states).reshape(B, T)
    diffs = f_vals[:, 1:] - f_vals[:, :-1]
    consistency = (diffs**2).mean()
    f_initial = f_vals[:, 0]
    variance = f_initial.var(unbiased=False)
    var_penalty = torch.relu(torch.as_tensor(epsilon, device=variance.device, dtype=variance.dtype) - variance)
    total = consistency + lambda_var * var_penalty
    return total, float(consistency.detach().cpu().item()), float(variance.detach().cpu().item())


def train_conservation_model(
    model,
    trajectories_np,
    save_path,
    lr=1e-3,
    epochs=400,
    batch_size=512,
    lambda_var=0.5,
    epsilon=10.0,
    device=None,
    model_name="model",
):
    """generic trainer for cdn/polynomial conservation models. expects normalized data for cdn, raw for poly."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    trajs_tensor = torch.tensor(trajectories_np, dtype=torch.float32, device=device)
    dataset = TensorDataset(trajs_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": [], "consistency": [], "variance": []}
    best_loss = float("inf")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_con = 0.0
        epoch_var = 0.0
        n_batches = 0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, con, var = conservation_loss(model, batch, lambda_var, epsilon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_con += con
            epoch_var += var
            n_batches += 1
        scheduler.step()

        avg_loss = epoch_loss / max(1, n_batches)
        avg_con = epoch_con / max(1, n_batches)
        avg_var = epoch_var / max(1, n_batches)
        history["loss"].append(avg_loss)
        history["consistency"].append(avg_con)
        history["variance"].append(avg_var)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 50 == 0:
            print(
                f"[{model_name}] epoch {epoch+1:4d}/{epochs} "
                f"loss={avg_loss:.6f} (best={best_loss:.6f}) cons={avg_con:.6f} var={avg_var:.6f}"
            )

    model.load_state_dict(torch.load(save_path, map_location=device))
    hist_path = save_path.replace(".pt", "_history.npy")
    np.save(hist_path, history)
    return model, history


@dataclass
class CDNTrainConfig:
    env_name: str
    state_dim: int
    hidden_dim: int = 256
    n_layers: int = 4
    lr: float = 1e-3
    epochs: int = 80
    batch_size: int = 256
    lambda_var: float = 0.1
    epsilon: float = 1.0
    var_reg: str = "hinge"  # 'hinge' | 'softplus'
    lambda_scale: float = 0.0
    target_mean: float = 0.0
    target_std: float = 1.0
    # safe band for std: no penalty inside [std_min, std_max]
    std_min: float = 0.8
    std_max: float = 1.2
    lambda_align: float = 0.0
    grad_clip: float = 1.0
    log_grad_norm: bool = False
    log_every: int = 10
    save_dir: str = "models"
    device: Optional[str] = None


def cdn_loss(
    model: ConservationDiscoveryNetwork,
    trajectories: torch.Tensor,
    energy0: Optional[torch.Tensor] = None,
    lambda_var: float = 0.1,
    epsilon: float = 1.0,
    var_reg: str = "hinge",
    lambda_scale: float = 0.0,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    std_min: float = 0.8,
    std_max: float = 1.2,
    lambda_align: float = 0.0,
):
    """conservation loss + optional scale/alignment terms. trajectories: (B,T,D)."""
    B, T, D = trajectories.shape
    states = trajectories.reshape(B * T, D)
    f_vals = model(states).reshape(B, T)

    diffs = f_vals[:, 1:] - f_vals[:, :-1]
    consistency = (diffs**2).mean()

    f_initial = f_vals[:, 0]
    variance = f_initial.var(unbiased=False)

    var_floor = torch.as_tensor(epsilon, device=variance.device, dtype=variance.dtype)
    if var_reg == "hinge":
        var_penalty = torch.relu(var_floor - variance) / (var_floor + 1e-8)
    elif var_reg == "softplus":
        # smooth hinge: softplus(k*(eps-var))/k ~= relu(eps-var) for large k
        k = 10.0
        var_penalty = F.softplus(k * (var_floor - variance)) / k / (var_floor + 1e-8)
    else:
        raise ValueError(f"unknown var_reg: {var_reg}")

    total = consistency + lambda_var * var_penalty

    # without a scale constraint the net can drift to huge magnitudes while still being "conserved"
    scale_loss = torch.zeros((), device=f_initial.device, dtype=f_initial.dtype)
    if lambda_scale and lambda_scale > 0:
        mean = f_initial.mean()
        std = torch.sqrt(variance + 1e-8)
        tm = torch.as_tensor(target_mean, device=f_initial.device, dtype=f_initial.dtype)
        mean_loss = (mean - tm) ** 2
        smin = torch.as_tensor(std_min, device=f_initial.device, dtype=f_initial.dtype)
        smax = torch.as_tensor(std_max, device=f_initial.device, dtype=f_initial.dtype)
        std_low = torch.relu(smin - std)
        std_high = torch.relu(std - smax)
        std_loss = std_low**2 + std_high**2
        scale_loss = mean_loss + std_loss
        total = total + lambda_scale * scale_loss

    if energy0 is not None and lambda_align > 0:
        # align the invariant scale to analytical energy at t=0; breaks the affine ambiguity
        e0 = energy0.to(f_initial.device).float()
        fz = (f_initial - f_initial.mean()) / (f_initial.std(unbiased=False) + 1e-8)
        ez = (e0 - e0.mean()) / (e0.std(unbiased=False) + 1e-8)
        total = total + lambda_align * torch.mean((fz - ez) ** 2)

    return (
        total,
        float(consistency.detach().cpu().item()),
        float(variance.detach().cpu().item()),
        float(scale_loss.detach().cpu().item()),
    )


def train_cdn(trajectories_np: np.ndarray, cfg: CDNTrainConfig, energy0_np: Optional[np.ndarray] = None):
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = cfg.device
    print(f"training cdn ({cfg.env_name}) on {device}")

    trajs_tensor = torch.tensor(trajectories_np, dtype=torch.float32)
    if energy0_np is None:
        dataset = TensorDataset(trajs_tensor)
    else:
        e0_tensor = torch.tensor(energy0_np, dtype=torch.float32)
        dataset = TensorDataset(trajs_tensor, e0_tensor)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = ConservationDiscoveryNetwork(cfg.state_dim, cfg.hidden_dim, cfg.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = {"loss": [], "consistency": [], "variance": []}
    best_loss = float("inf")
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f"cdn_{cfg.env_name}_best.pt")

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cons = 0.0
        epoch_var = 0.0
        epoch_scale = 0.0
        n_batches = 0

        for batch in loader:
            batch_cpu = batch[0]
            batch_traj = batch_cpu.to(device)
            batch_e0 = batch[1].to(device) if len(batch) > 1 else None
            opt.zero_grad(set_to_none=True)
            loss, cons, var, scale = cdn_loss(
                model=model,
                trajectories=batch_traj,
                energy0=batch_e0,
                lambda_var=cfg.lambda_var,
                epsilon=cfg.epsilon,
                var_reg=cfg.var_reg,
                lambda_scale=cfg.lambda_scale,
                target_mean=cfg.target_mean,
                target_std=cfg.target_std,
                std_min=cfg.std_min,
                std_max=cfg.std_max,
                lambda_align=cfg.lambda_align,
            )
            loss.backward()
            if cfg.grad_clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.grad_clip
                )
            opt.step()

            epoch_loss += float(loss.detach().cpu().item())
            epoch_cons += cons
            epoch_var += var
            epoch_scale += scale
            n_batches += 1

        sched.step()
        avg_loss = epoch_loss / max(1, n_batches)
        avg_cons = epoch_cons / max(1, n_batches)
        avg_var = epoch_var / max(1, n_batches)
        avg_scale = epoch_scale / max(1, n_batches)

        history["loss"].append(avg_loss)
        history["consistency"].append(avg_cons)
        history["variance"].append(avg_var)
        history.setdefault("scale_loss", []).append(avg_scale)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

        if ((epoch + 1) % max(1, cfg.log_every) == 0) or epoch == 0:
            print(
                f"  ep {epoch+1:3d}/{cfg.epochs} "
                f"loss={avg_loss:.6f} cons={avg_cons:.6f} var={avg_var:.4f} scale={avg_scale:.6f}"
            )
            if cfg.log_grad_norm and cfg.grad_clip is not None:
                try:
                    print(f"    grad_norm_before_clip={float(total_norm):.4f}")
                except Exception:
                    pass

    np.save(os.path.join(cfg.save_dir, f"cdn_{cfg.env_name}_history.npy"), history)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, history

