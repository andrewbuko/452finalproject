from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


@dataclass
class LassoConfig:
    alpha: float = 1e-3
    max_iter: int = 50_000
    fit_intercept: bool = True


def fit_lasso(Theta: np.ndarray, y: np.ndarray, cfg: LassoConfig) -> Tuple[np.ndarray, float, StandardScaler]:
    """
    Fit LASSO on standardized columns:
      min ||Theta w - y||^2 + alpha * ||w||_1
    Returns coefficients in ORIGINAL (unscaled) Theta units.
    """
    Theta = np.asarray(Theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Theta_s = scaler.fit_transform(Theta)

    model = Lasso(alpha=cfg.alpha, fit_intercept=cfg.fit_intercept, max_iter=cfg.max_iter)
    model.fit(Theta_s, y)

    # Convert coefficients back to original Theta units
    w_scaled = model.coef_.astype(np.float64)
    w = w_scaled / (scaler.scale_ + 1e-12)
    intercept = float(model.intercept_)
    # When using standardized features, intercept corresponds to y - sum(w_scaled * mean_s)
    # but since StandardScaler centers, sklearn handles it; we keep intercept as-is.
    return w, intercept, scaler


def format_equation(names: List[str], w: np.ndarray, intercept: float = 0.0, threshold: float = 1e-2) -> str:
    terms: List[str] = []
    if abs(intercept) >= threshold:
        terms.append(f"{intercept:+.4f}")
    for name, c in zip(names, w.tolist()):
        if abs(c) < threshold:
            continue
        terms.append(f"{c:+.4f}*{name}")

    rhs = " ".join(terms) if terms else "0"
    if rhs.startswith("+"):
        rhs = rhs[1:].lstrip()
    return f"f(s) = {rhs}"

