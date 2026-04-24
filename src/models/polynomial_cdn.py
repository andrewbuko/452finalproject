from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def _fmt_term(coeff: float, term: str) -> str:
    return f"{coeff:+.4f}*{term}"


@dataclass(frozen=True)
class PolynomialSpec:
    var_names: List[str]
    term_names: List[str]


PROJECTILE_SPEC = PolynomialSpec(
    var_names=["x", "y", "vx", "vy"],
    term_names=[
        "1",
        "x",
        "y",
        "vx",
        "vy",
        "x^2",
        "y^2",
        "vx^2",
        "vy^2",
        "x*y",
        "x*vx",
        "x*vy",
        "y*vx",
        "y*vy",
        "vx*vy",
    ],
)


PENDULUM_SPEC = PolynomialSpec(
    var_names=["theta", "omega"],
    term_names=["1", "theta", "omega", "theta^2", "omega^2", "theta*omega", "cos(theta)", "sin(theta)"],
)


class PolynomialConservationModel(nn.Module):
    """
    Linear model over hand-designed features:
      f(s) = sum_i w_i * phi_i(s)
    where the learned weights are directly interpretable as equation coefficients.
    """

    def __init__(self, env_name: str):
        super().__init__()
        if env_name not in ("projectile", "pendulum"):
            raise ValueError(env_name)
        self.env_name = env_name
        self.spec = PROJECTILE_SPEC if env_name == "projectile" else PENDULUM_SPEC
        # We learn weights on *normalized* features so all terms are comparable.
        # Physical-unit coefficients are recovered analytically from (w, mean, std).
        self.weights = nn.Parameter(torch.zeros(len(self.spec.term_names), dtype=torch.float32))

        n_terms = len(self.spec.term_names)
        self.register_buffer("phi_mean", torch.zeros(n_terms, dtype=torch.float32), persistent=True)
        self.register_buffer("phi_std", torch.ones(n_terms, dtype=torch.float32), persistent=True)
        # Keep the raw (unclamped) std so we can identify near-constant terms.
        self.register_buffer("phi_std_raw", torch.ones(n_terms, dtype=torch.float32), persistent=True)

    @property
    def state_dim(self) -> int:
        return 4 if self.env_name == "projectile" else 2

    def features(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
          s: (B, D) raw state (NOT normalized)
        Returns:
          phi: (B, n_terms)
        """
        if s.shape[-1] != self.state_dim:
            raise ValueError(f"Expected last dim {self.state_dim}, got {s.shape[-1]}")

        if self.env_name == "projectile":
            x, y, vx, vy = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
            ones = torch.ones_like(x)
            phi = torch.stack(
                [
                    ones,
                    x,
                    y,
                    vx,
                    vy,
                    x * x,
                    y * y,
                    vx * vx,
                    vy * vy,
                    x * y,
                    x * vx,
                    x * vy,
                    y * vx,
                    y * vy,
                    vx * vy,
                ],
                dim=-1,
            )
            return phi

        # pendulum
        theta, omega = s[..., 0], s[..., 1]
        ones = torch.ones_like(theta)
        phi = torch.stack(
            [
                ones,
                theta,
                omega,
                theta * theta,
                omega * omega,
                theta * omega,
                torch.cos(theta),
                torch.sin(theta),
            ],
            dim=-1,
        )
        return phi

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        phi = self.features(s)
        phi_n = (phi - self.phi_mean) / (self.phi_std + 1e-12)
        return (phi_n * self.weights).sum(dim=-1)

    def set_feature_scaler(self, phi_mean: np.ndarray, phi_std: np.ndarray, std_floor: float = 1e-3):
        """
        Set normalization stats for the feature library.
        We keep the constant feature '1' unnormalized: mean=0, std=1.
        """
        m = np.asarray(phi_mean, dtype=np.float64).copy()
        s = np.asarray(phi_std, dtype=np.float64).copy()
        if m.shape != (len(self.spec.term_names),) or s.shape != (len(self.spec.term_names),):
            raise ValueError("Bad scaler shapes")

        # constant term should stay constant
        m[0] = 0.0
        s[0] = 1.0
        raw_s = s.copy()
        # Clamp tiny stds (common when a feature is effectively constant, e.g. vx fixed).
        s = np.maximum(s, float(std_floor))

        with torch.no_grad():
            self.phi_mean.copy_(torch.tensor(m, dtype=torch.float32, device=self.phi_mean.device))
            self.phi_std.copy_(torch.tensor(s, dtype=torch.float32, device=self.phi_std.device))
            self.phi_std_raw.copy_(torch.tensor(raw_s, dtype=torch.float32, device=self.phi_std_raw.device))

    def physical_coefficients(self) -> List[Tuple[str, float]]:
        """
        Convert normalized-feature weights into coefficients on the *raw* feature library:
          f = sum_i w_i * ((phi_i - mean_i)/std_i)
            = sum_i (w_i/std_i)*phi_i + (w_0 - sum_{i>0} w_i*mean_i/std_i)
        Returns coefficients aligned to self.spec.term_names.
        """
        w = self.weights.detach().cpu().numpy().astype(np.float64)
        m = self.phi_mean.detach().cpu().numpy().astype(np.float64)
        s = self.phi_std.detach().cpu().numpy().astype(np.float64)

        a = w / (s + 1e-12)
        b = a[0]  # constant feature already has mean=0,std=1
        # adjust intercept for mean-shifts of non-constant features
        b = b - np.sum(w[1:] * m[1:] / (s[1:] + 1e-12))
        a[0] = b
        return list(zip(self.spec.term_names, a.tolist()))

    def print_equation(self, threshold: float = 0.01, const_feature_std_eps: float = 1e-6) -> str:
        terms = []
        std_raw = self.phi_std_raw.detach().cpu().numpy().astype(np.float64)
        for i, (name, coeff) in enumerate(self.physical_coefficients()):
            if name == "1":
                if abs(coeff) >= threshold:
                    terms.append(f"{coeff:+.4f}")
                continue
            # If the feature is essentially constant in the dataset, its coefficient is not identifiable.
            if std_raw[i] < const_feature_std_eps:
                continue
            if abs(coeff) < threshold:
                continue
            terms.append(_fmt_term(coeff, name))

        if not terms:
            eq = "f(s) = 0"
        else:
            rhs = " ".join(terms)
            if rhs.startswith("+"):
                rhs = rhs[1:].lstrip()
            eq = f"f(s) = {rhs}"
        return eq

