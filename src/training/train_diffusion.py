from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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
    # if > 0, add extra conditioning dim(s) to the diffusion model
    cond_dim: int = 0


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
    """build (s_t, delta) pairs from standardized trajectories. (N,T,D) -> (N*(T-1),D) each."""
    s_t = trajs_z[:, :-1, :].reshape(-1, trajs_z.shape[-1])
    delta = (trajs_z[:, 1:, :] - trajs_z[:, :-1, :]).reshape(-1, trajs_z.shape[-1])
    return s_t.astype(np.float32), delta.astype(np.float32)


def train_diffusion_transition(
    trajectories_raw: np.ndarray,
    cfg: DiffusionTrainConfig,
    cond_np: Optional[np.ndarray] = None,
) -> Tuple[DiffusionTransitionModel, Dict[str, np.ndarray], Dict[str, list]]:
    """train a ddpm transition model on standardized one-step deltas. returns (model, standardize_stats, history)."""
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
    if cfg.cond_dim:
        if cond_np is None:
            raise ValueError("cond_np must be provided when cfg.cond_dim > 0")
        cond_np = np.asarray(cond_np, dtype=np.float32)
        if cond_np.ndim == 1:
            cond_np = cond_np[:, None]
        # caller may have computed cond_np over the full dataset; trim to the transitions we use here
        if cond_np.shape[0] > s_t_np.shape[0]:
            cond_np = cond_np[: s_t_np.shape[0], :]
        if cond_np.shape[0] != s_t_np.shape[0]:
            raise ValueError(f"cond_np rows {cond_np.shape[0]} != s_t rows {s_t_np.shape[0]}")
        if cond_np.shape[1] != int(cfg.cond_dim):
            raise ValueError(f"cond_np dim {cond_np.shape[1]} != cfg.cond_dim {cfg.cond_dim}")

    if cfg.cond_dim:
        ds = TensorDataset(torch.from_numpy(s_t_np), torch.from_numpy(delta_np), torch.from_numpy(cond_np))
    else:
        ds = TensorDataset(torch.from_numpy(s_t_np), torch.from_numpy(delta_np))
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)

    model = DiffusionTransitionModel(
        state_dim=int(cfg.state_dim),
        K=int(cfg.K),
        hidden_dim=int(cfg.hidden_dim),
        time_emb_dim=int(cfg.time_emb_dim),
        cond_dim=int(cfg.cond_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))

    history = {"loss": []}
    best = float("inf")

    print(f"training diffusion ({cfg.env_name}) on {device} with {len(ds)} transitions")
    for epoch in range(int(cfg.epochs)):
        model.train()
        total = 0.0
        nb = 0
        for batch in loader:
            s_t = batch[0].to(device)
            delta = batch[1].to(device)
            cond = batch[2].to(device) if (cfg.cond_dim and len(batch) > 2) else None
            opt.zero_grad(set_to_none=True)
            loss = model.training_loss(s_t, delta, cond=cond)
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
            print(f"  diffusion {cfg.env_name} ep {epoch+1:3d}/{cfg.epochs} loss={avg:.6f} best={best:.6f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, stats, history


@torch.no_grad()
def rollout_diffusion(
    model: DiffusionTransitionModel,
    s0_raw: np.ndarray,
    T: int,
    stats: Dict[str, np.ndarray],
    device: Optional[str] = None,
    cond_fn_raw: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    energy_model: Optional[torch.nn.Module] = None,
    project_steps: int = 1,
    project_eps: float = 1e-8,
) -> np.ndarray:
    """rollout in standardized space then convert back to raw. s0_raw (B,D) -> traj_raw (B,T,D)."""
    if device is None:
        device = str(next(model.parameters()).device)
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    s0 = s0_raw.astype(np.float32)
    s0_z = (s0 - stats["mean"][None, :]) / stats["std"][None, :]
    s0_z_t = torch.from_numpy(s0_z).to(dev)

    def _cond_fn(s_z: torch.Tensor, _t: int):
        if cond_fn_raw is None:
            return None
        s_raw = (s_z.detach().cpu().numpy() * stats["std"][None, :] + stats["mean"][None, :]).astype(np.float32)
        c = cond_fn_raw(s_raw).astype(np.float32)
        if c.ndim == 1:
            c = c[:, None]
        return torch.from_numpy(c).to(dev)

    H0 = None
    if energy_model is not None and int(project_steps) > 0:
        energy_model = energy_model.to(dev)
        energy_model.eval()
        with torch.no_grad():
            H0 = energy_model(torch.tensor(s0, dtype=torch.float32, device=dev)).detach()  # (B,)

    def _project_fn(s_z: torch.Tensor, _t: int):
        if (energy_model is None) or (H0 is None) or int(project_steps) <= 0:
            return s_z

        mean_t = torch.tensor(stats["mean"], device=s_z.device)[None, :]
        std_t = torch.tensor(stats["std"], device=s_z.device)[None, :]
        s_raw = s_z * std_t + mean_t

        # project onto the level set energy_model(s)=H0 with a first-order step along the gradient
        with torch.enable_grad():
            for _ in range(int(project_steps)):
                s_raw = s_raw.detach().clone().requires_grad_(True)
                H = energy_model(s_raw)  # (B,)
                diff = (H - H0)
                grad = torch.autograd.grad(H.sum(), s_raw, create_graph=False)[0]  # (B, D)
                denom = torch.sum(grad * grad, dim=1, keepdim=True) + float(project_eps)
                s_raw = (s_raw - diff.unsqueeze(1) * grad / denom).detach()

        return (s_raw - mean_t) / std_t

    traj_z = (
        model.rollout(
            s0_z_t,
            int(T),
            cond_fn=_cond_fn if cond_fn_raw is not None else None,
            project_fn=_project_fn if energy_model is not None else None,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    traj_raw = _destandardize(traj_z, stats).astype(np.float32)
    return traj_raw

