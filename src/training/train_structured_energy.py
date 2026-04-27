"""
Training loop for StructuredEnergyNetwork.

Uses the same conservation loss as CDN/polynomial, and adds energy alignment
to pin scale/offset so coefficients are in physical units.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.structured_energy import StructuredEnergyNetwork
from src.training.train_cdn import conservation_loss


@dataclass
class StructuredEnergyConfig:
    env_name: str
    state_dim: int
    pos_dims: list[int]
    vel_dims: list[int]
    hidden_dim: int = 128
    n_layers: int = 2
    activation: str = "silu"
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 1024
    lambda_var: float = 1.0
    epsilon: Optional[float] = None
    lambda_energy: float = 0.1
    grad_clip: float = 1.0
    save_dir: str = "models"
    device: Optional[str] = None


def train_structured_energy(
    raw_trajectories_np: np.ndarray,
    energy0_np: np.ndarray,
    cfg: StructuredEnergyConfig,
) -> Tuple[StructuredEnergyNetwork, dict]:
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)

    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f"structured_energy_{cfg.env_name}_best.pt")

    # epsilon heuristic from energy0 variance (so the variance hinge is not too strict)
    if cfg.epsilon is None:
        evar = float(np.var(energy0_np[:5000]))
        cfg.epsilon = max(evar * 0.1, 1.0)

    model = StructuredEnergyNetwork(
        state_dim=cfg.state_dim,
        pos_dims=cfg.pos_dims,
        vel_dims=cfg.vel_dims,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        activation=cfg.activation,
    ).to(device)

    max_train = min(len(raw_trajectories_np), 200_000)
    trajs = torch.tensor(raw_trajectories_np[:max_train], dtype=torch.float32, device=device)
    e0 = torch.tensor(np.asarray(energy0_np[:max_train], dtype=np.float32), dtype=torch.float32, device=device)
    e0_scale = float(np.std(energy0_np[:max_train]) + 1e-8)

    loader = DataLoader(TensorDataset(trajs, e0), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = {"loss": []}
    best = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for batch_traj, batch_e0 in loader:
            opt.zero_grad(set_to_none=True)
            loss, _, _ = conservation_loss(model, batch_traj, lambda_var=cfg.lambda_var, epsilon=float(cfg.epsilon))

            B, T, D = batch_traj.shape
            f0 = model(batch_traj[:, 0, :].reshape(B, D))
            align = torch.mean((f0 - batch_e0) ** 2) / (e0_scale**2)
            loss = loss + float(cfg.lambda_energy) * align

            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            opt.step()
            total += float(loss.detach().cpu().item())
            n_batches += 1

        sched.step()
        avg = total / max(1, n_batches)
        history["loss"].append(avg)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), ckpt_path)
        if (epoch + 1) % 50 == 0:
            print(f"[structured {cfg.env_name}] epoch {epoch+1}/{cfg.epochs} loss={avg:.6f} best={best:.6f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, history

