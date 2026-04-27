from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard transformer-style sinusoidal embedding.
    Args:
      t: (B,) integer timesteps
      dim: embedding dimension
    Returns:
      (B, dim)
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(0, half, device=device).float() / max(1, half))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=1)
    return emb


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor  # (K,)
    alphas: torch.Tensor  # (K,)
    alphas_cumprod: torch.Tensor  # (K,)
    sqrt_alphas_cumprod: torch.Tensor  # (K,)
    sqrt_one_minus_alphas_cumprod: torch.Tensor  # (K,)


def make_linear_schedule(K: int, beta_start: float = 1e-4, beta_end: float = 2e-2, device: Optional[torch.device] = None):
    betas = torch.linspace(beta_start, beta_end, K, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    )


class TransitionEpsModel(nn.Module):
    """
    Predict diffusion noise eps for a delta-state vector, conditioned on current state s_t and time index t.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, time_emb_dim: int = 64):
        super().__init__()
        self.state_dim = int(state_dim)
        self.time_emb_dim = int(time_emb_dim)

        in_dim = state_dim + state_dim + time_emb_dim  # s_t, x_k (noised delta), emb(k)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, s_t: torch.Tensor, x_k: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
          s_t: (B, D) current state
          x_k: (B, D) noised delta
          k: (B,) diffusion step index in [0, K-1]
        Returns:
          eps_hat: (B, D)
        """
        emb = _sinusoidal_embedding(k, self.time_emb_dim)
        h = torch.cat([s_t, x_k, emb], dim=1)
        return self.net(h)


class DiffusionTransitionModel(nn.Module):
    """
    DDPM-style conditional diffusion model over delta = s_{t+1}-s_t.
    Trained with eps-prediction objective.
    """

    def __init__(self, state_dim: int, K: int = 50, hidden_dim: int = 256, time_emb_dim: int = 64):
        super().__init__()
        self.state_dim = int(state_dim)
        self.K = int(K)
        self.eps_model = TransitionEpsModel(state_dim=state_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim)
        self.register_buffer("_betas", torch.empty(self.K))
        self.register_buffer("_alphas", torch.empty(self.K))
        self.register_buffer("_alphas_cumprod", torch.empty(self.K))
        self.register_buffer("_sqrt_alphas_cumprod", torch.empty(self.K))
        self.register_buffer("_sqrt_one_minus_alphas_cumprod", torch.empty(self.K))
        self.set_schedule(make_linear_schedule(self.K, device=torch.device("cpu")))

    @property
    def schedule(self) -> DiffusionSchedule:
        return DiffusionSchedule(
            betas=self._betas,
            alphas=self._alphas,
            alphas_cumprod=self._alphas_cumprod,
            sqrt_alphas_cumprod=self._sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self._sqrt_one_minus_alphas_cumprod,
        )

    def set_schedule(self, sched: DiffusionSchedule):
        self._betas.copy_(sched.betas)
        self._alphas.copy_(sched.alphas)
        self._alphas_cumprod.copy_(sched.alphas_cumprod)
        self._sqrt_alphas_cumprod.copy_(sched.sqrt_alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod.copy_(sched.sqrt_one_minus_alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, k: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Sample x_k = sqrt(a_bar_k) * x0 + sqrt(1-a_bar_k) * eps
        """
        s1 = self._sqrt_alphas_cumprod[k].unsqueeze(1)
        s2 = self._sqrt_one_minus_alphas_cumprod[k].unsqueeze(1)
        return s1 * x0 + s2 * eps

    def training_loss(self, s_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
          s_t: (B, D)
          delta: (B, D) = s_{t+1}-s_t
        """
        B, D = delta.shape
        device = delta.device
        k = torch.randint(0, self.K, (B,), device=device)
        eps = torch.randn(B, D, device=device)
        x_k = self.q_sample(delta, k, eps)
        eps_hat = self.eps_model(s_t, x_k, k)
        return torch.mean((eps_hat - eps) ** 2)

    @torch.no_grad()
    def sample_delta(self, s_t: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        Reverse diffusion to sample delta ~ p(delta | s_t).
        Uses a simple DDPM sampler (no classifier-free guidance).
        """
        device = s_t.device
        B, D = s_t.shape
        K = int(self.K if n_steps is None else n_steps)
        if K != self.K:
            # keep it simple: require same schedule length for now
            raise ValueError("n_steps must match model.K")

        x = torch.randn(B, D, device=device)
        for k in reversed(range(self.K)):
            kk = torch.full((B,), k, device=device, dtype=torch.long)
            beta = self._betas[kk].unsqueeze(1)
            alpha = self._alphas[kk].unsqueeze(1)
            a_bar = self._alphas_cumprod[kk].unsqueeze(1)

            eps_hat = self.eps_model(s_t, x, kk)
            # Predict x0 (delta) from eps_hat
            x0_hat = (x - torch.sqrt(1.0 - a_bar) * eps_hat) / torch.sqrt(a_bar + 1e-8)
            # DDPM mean
            mean = (1.0 / torch.sqrt(alpha + 1e-8)) * (x - (beta / torch.sqrt(1.0 - a_bar + 1e-8)) * eps_hat)
            if k > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def rollout(self, s0: torch.Tensor, T: int) -> torch.Tensor:
        """
        Roll out a trajectory by repeatedly sampling delta.
        Args:
          s0: (B, D)
          T: number of timesteps
        Returns:
          traj: (B, T, D)
        """
        B, D = s0.shape
        traj = torch.zeros(B, T, D, device=s0.device, dtype=s0.dtype)
        traj[:, 0, :] = s0
        s = s0
        for t in range(1, T):
            delta = self.sample_delta(s)
            s = s + delta
            traj[:, t, :] = s
        return traj

