"""linear model over a polynomial+trig basis. weights are the equation."""

import itertools
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


class PolynomialConservation(nn.Module):
    """f(s) = sum_i w_i * phi_i(s) over a fixed feature library. train on raw (unnormalized) data."""

    def __init__(self, state_dim, degree=2, include_trig_dims=None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.degree = int(degree)
        self.include_trig_dims = list(include_trig_dims or [])

        self.feature_names = self._build_feature_names()
        n_features = len(self.feature_names)

        self.weights = nn.Parameter(torch.zeros(n_features, dtype=torch.float32))
        nn.init.normal_(self.weights, mean=0.0, std=0.01)

    def _build_feature_names(self, var_names=None):
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.state_dim)]
        names = ["1"]
        for name in var_names:
            names.append(name)

        for d in range(2, self.degree + 1):
            for combo in itertools.combinations_with_replacement(range(self.state_dim), d):
                name_parts = [var_names[i] for i in combo]
                counts = Counter(name_parts)
                term = "*".join([f"{v}^{c}" if c > 1 else v for v, c in counts.items()])
                names.append(term)

        for dim in self.include_trig_dims:
            base = var_names[dim]
            names.append(f"cos({base})")
            names.append(f"sin({base})")
        return names

    def _compute_features(self, s):
        B = s.shape[0]
        feats = [torch.ones(B, 1, device=s.device)]
        feats.append(s)

        for d in range(2, self.degree + 1):
            for combo in itertools.combinations_with_replacement(range(self.state_dim), d):
                term = torch.ones(B, 1, device=s.device)
                for dim in combo:
                    term = term * s[:, dim : dim + 1]
                feats.append(term)

        for dim in self.include_trig_dims:
            feats.append(torch.cos(s[:, dim : dim + 1]))
            feats.append(torch.sin(s[:, dim : dim + 1]))

        return torch.cat(feats, dim=1)

    def forward(self, s):
        features = self._compute_features(s)
        return (features * self.weights).sum(dim=1)

    def print_equation(self, var_names=None, threshold=0.01):
        if var_names is not None:
            self.feature_names = self._build_feature_names(var_names)
        w = self.weights.detach().cpu().numpy()
        terms = []
        for name, weight in zip(self.feature_names, w):
            if abs(weight) > threshold:
                if name == "1":
                    terms.append(f"{weight:+.4f}")
                else:
                    terms.append(f"{weight:+.4f}*{name}")
        equation = " ".join(terms)
        print(f"f(s) = {equation}")
        return equation

    def get_equation_dict(self):
        w = self.weights.detach().cpu().numpy()
        return {name: float(weight) for name, weight in zip(self.feature_names, w)}
