import torch
import torch.nn as nn


class ConservationDiscoveryNetwork(nn.Module):
    """
    MLP mapping state s_t (R^D) -> scalar f(s_t) intended to be conserved along a trajectory.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B, D) tensor of states
        Returns:
            (B,) tensor of scalar values
        """
        return self.net(s).squeeze(-1)

