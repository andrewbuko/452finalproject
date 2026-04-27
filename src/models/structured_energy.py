"""
Structured Energy Network baseline.

Learn a conserved quantity with an inductive bias:
  H(s) = T(v) + V(q)
where T depends only on velocity-like dimensions and V only on position-like
dimensions. This is a strong physics prior for many mechanical systems.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_mlp(in_dim: int, hidden_dim: int, n_layers: int, activation: str) -> nn.Sequential:
    act = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    if n_layers <= 0:
        return nn.Sequential(nn.Linear(in_dim, 1))
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(act())
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act())
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class StructuredEnergyNetwork(nn.Module):
    """
    H(s) = T(v_dims) + V(q_dims)
    """

    def __init__(
        self,
        state_dim: int,
        pos_dims: list[int],
        vel_dims: list[int],
        hidden_dim: int = 128,
        n_layers: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.pos_dims = list(pos_dims)
        self.vel_dims = list(vel_dims)
        if not self.pos_dims or not self.vel_dims:
            raise ValueError("pos_dims and vel_dims must be non-empty")
        if any(d < 0 or d >= self.state_dim for d in (self.pos_dims + self.vel_dims)):
            raise ValueError("pos_dims/vel_dims out of range")

        self.T_net = _make_mlp(len(self.vel_dims), hidden_dim, n_layers, activation)
        self.V_net = _make_mlp(len(self.pos_dims), hidden_dim, n_layers, activation)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.shape[-1] != self.state_dim:
            raise ValueError(f"Expected last dim {self.state_dim}, got {s.shape[-1]}")
        q = s[..., self.pos_dims]
        v = s[..., self.vel_dims]
        T = self.T_net(v).squeeze(-1)
        V = self.V_net(q).squeeze(-1)
        return T + V

