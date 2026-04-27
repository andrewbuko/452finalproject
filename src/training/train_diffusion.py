from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.diffusion_transition import DiffusionTransitionModel


@dataclass
class DiffusionTrainConfig:
    env_name: str
    state_dim: int
    K: int = 50
    hidden_dim: int = 256
    time_emb_dim: int = 64
    lr: float = 2e-4
    epochs: int = 30
    batch_size: int = 4096
    max_train_trajectories: int = 200_000
    device: Optional[str] = None
    save_dir: str = "models"
    log_every: int = 2


def _compute_standardize_stats(trajs: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    flat = trajs.reshape(-1, trajs.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _standardize(trajs: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (trajs - stats["mean"][None, None, :]) / stats["std"][None, None, :]


def _destandardize(trajs_z: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return trajs_z * stats["std"][None, None, :] + stats["mean"][None, None, :]


def build_transition_dataset(trajs_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (s_t, delta_t) pairs from standardized trajectories.
    trajs_z: (N, T, D)
    Returns:
      s_t: (N*(T-1), D)
      delta: (N*(T-1), D)
    """
    s_t = trajs_z[:, :-1, :].reshape(-1, trajs_z.shape[-1])
    delta = (trajs_z[:, 1:, :] - trajs_z[:, :-1, :]).reshape(-1, trajs_z.shape[-1])
    return s_t.astype(np.float32), delta.astype(np.float32)


def train_diffusion_transition(
    trajectories_raw: np.ndarray,
    cfg: DiffusionTrainConfig,
) -> Tuple[DiffusionTransitionModel, Dict[str, np.ndarray], Dict[str, list]]:
    """
    Train a diffusion transition model on one-step deltas (standardized).
    Returns:
      model (on cfg.device),
      standardization stats computed from training subset only,
      history dict
    """
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)

    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f"diffusion_{cfg.env_name}_best.pt")
    stats_path = os.path.join(cfg.save_dir, f"diffusion_{cfg.env_name}_stats.npz")

    n = min(len(trajectories_raw), int(cfg.max_train_trajectories))
    train_subset = trajectories_raw[:n].astype(np.float32, copy=False)

    stats = _compute_standardize_stats(train_subset)
    trajs_z = _standardize(train_subset, stats)
    s_t_np, delta_np = build_transition_dataset(trajs_z)

    ds = TensorDataset(torch.from_numpy(s_t_np), torch.from_numpy(delta_np))
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)

    model = DiffusionTransitionModel(
        state_dim=int(cfg.state_dim),
        K=int(cfg.K),
        hidden_dim=int(cfg.hidden_dim),
        time_emb_dim=int(cfg.time_emb_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))

    history = {"loss": []}
    best = float("inf")

    print(f"Training diffusion transition ({cfg.env_name}) on {device} with {len(ds)} transitions")
    for epoch in range(int(cfg.epochs)):
        model.train()
        total = 0.0
        nb = 0
        for s_t, delta in loader:
            s_t = s_t.to(device)
            delta = delta.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model.training_loss(s_t, delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.detach().cpu().item())
            nb += 1

        avg = total / max(1, nb)
        history["loss"].append(avg)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), ckpt_path)
            np.savez(stats_path, mean=stats["mean"], std=stats["std"])

        if (epoch == 0) or ((epoch + 1) % max(1, int(cfg.log_every)) == 0):
            print(f"[diffusion {cfg.env_name}] epoch {epoch+1:3d}/{cfg.epochs} loss={avg:.6f} best={best:.6f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, stats, history


@torch.no_grad()
def rollout_diffusion(
    model: DiffusionTransitionModel,
    s0_raw: np.ndarray,
    T: int,
    stats: Dict[str, np.ndarray],
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Rollout in standardized space then convert back to raw.
    Args:
      s0_raw: (B, D)
    Returns:
      traj_raw: (B, T, D)
    """
    if device is None:
        device = str(next(model.parameters()).device)
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    s0 = s0_raw.astype(np.float32)
    s0_z = (s0 - stats["mean"][None, :]) / stats["std"][None, :]
    traj_z = model.rollout(torch.from_numpy(s0_z).to(dev), int(T)).detach().cpu().numpy().astype(np.float32)
    traj_raw = _destandardize(traj_z, stats).astype(np.float32)
    return traj_raw

