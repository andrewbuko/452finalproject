from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Tuple

import numpy as np
import torch

from src.models.cdn import ConservationDiscoveryNetwork
from src.training.train_cdn import cdn_loss


@dataclass
class StreamingCDNConfig:
    env_name: str
    state_dim: int
    hidden_dim: int = 1024
    n_layers: int = 6
    lr: float = 3e-4
    epochs: int = 5
    steps_per_epoch: int = 2000
    lambda_var: float = 0.1
    epsilon: float = 1.0
    var_reg: str = "softplus"
    lambda_scale: float = 0.1
    target_mean: float = 0.0
    target_std: float = 1.0
    std_min: float = 0.8
    std_max: float = 1.2
    lambda_align: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 50
    save_dir: str = "models"
    device: str = "cuda"
    amp: bool = True


BatchFn = Callable[[], Tuple[torch.Tensor, torch.Tensor | None]]


def train_cdn_streaming(make_batch: BatchFn, cfg: StreamingCDNConfig) -> Tuple[ConservationDiscoveryNetwork, Dict[str, list]]:
    device = cfg.device
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f"cdn_{cfg.env_name}_best.pt")

    model = ConservationDiscoveryNetwork(cfg.state_dim, cfg.hidden_dim, cfg.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))

    history: Dict[str, list] = {"loss": [], "consistency": [], "variance": [], "scale_loss": []}
    best = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        run_loss = 0.0
        run_cons = 0.0
        run_var = 0.0
        run_scale = 0.0

        for step in range(cfg.steps_per_epoch):
            traj, e0 = make_batch()
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.startswith("cuda"))):
                loss, cons, var, scale = cdn_loss(
                    model=model,
                    trajectories=traj,
                    energy0=e0,
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

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            run_loss += float(loss.detach().float().cpu().item())
            run_cons += cons
            run_var += var
            run_scale += scale

            if (step + 1) % max(1, cfg.log_every) == 0:
                n = step + 1
                print(
                    f"[{cfg.env_name}] epoch {epoch+1}/{cfg.epochs} step {step+1}/{cfg.steps_per_epoch} "
                    f"loss={run_loss/n:.6f} cons={run_cons/n:.6f} var={run_var/n:.4f} scale={run_scale/n:.6f}"
                )

        avg_loss = run_loss / cfg.steps_per_epoch
        avg_cons = run_cons / cfg.steps_per_epoch
        avg_var = run_var / cfg.steps_per_epoch
        avg_scale = run_scale / cfg.steps_per_epoch

        history["loss"].append(avg_loss)
        history["consistency"].append(avg_cons)
        history["variance"].append(avg_var)
        history["scale_loss"].append(avg_scale)

        if avg_loss < best:
            best = avg_loss
            torch.save(model.state_dict(), ckpt_path)

    np.save(os.path.join(cfg.save_dir, f"cdn_{cfg.env_name}_history.npy"), history)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, history

