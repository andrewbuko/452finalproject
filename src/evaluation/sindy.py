from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class STLSQConfig:
    threshold: float = 0.05
    max_iter: int = 15
    normalize_columns: bool = True


def stlsq(Theta: np.ndarray, y: np.ndarray, cfg: STLSQConfig) -> np.ndarray:
    """
    Sequentially Thresholded Least Squares (SINDy-style) for sparse regression:
      min ||Theta w - y||_2 with iterative hard-thresholding on small coefficients.
    """
    Theta = np.asarray(Theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if Theta.ndim != 2:
        raise ValueError("Theta must be 2D")
    if Theta.shape[0] != y.shape[0]:
        raise ValueError("Theta and y must have same number of rows")

    col_scale = np.ones(Theta.shape[1], dtype=np.float64)
    Theta_n = Theta
    if cfg.normalize_columns:
        col_scale = np.linalg.norm(Theta, axis=0) + 1e-12
        Theta_n = Theta / col_scale[None, :]

    # initial least squares
    w = np.linalg.lstsq(Theta_n, y, rcond=None)[0]
    keep = np.ones_like(w, dtype=bool)

    for _ in range(cfg.max_iter):
        small = np.abs(w) < cfg.threshold
        new_keep = keep & (~small)
        if np.all(new_keep == keep):
            break
        keep = new_keep
        if keep.sum() == 0:
            w[:] = 0.0
            break
        w_keep = np.linalg.lstsq(Theta_n[:, keep], y, rcond=None)[0]
        w[:] = 0.0
        w[keep] = w_keep

    # unnormalize back to original Theta columns
    w = w / col_scale
    return w


def build_library_projectile(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    X: (N,4) [x,y,vx,vy]
    Terms match PolynomialConservationModel's projectile library.
    """
    x, y, vx, vy = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    Theta = np.column_stack(
        [
            np.ones_like(x),
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
        ]
    )
    names = [
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
    ]
    return Theta, names


def build_library_pendulum(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    X: (N,2) [theta,omega]
    """
    th, om = X[:, 0], X[:, 1]
    Theta = np.column_stack(
        [
            np.ones_like(th),
            th,
            om,
            th * th,
            om * om,
            th * om,
            np.cos(th),
            np.sin(th),
        ]
    )
    names = ["1", "theta", "omega", "theta^2", "omega^2", "theta*omega", "cos(theta)", "sin(theta)"]
    return Theta, names


def format_sparse_equation(names: List[str], w: np.ndarray, threshold: float = 0.01) -> str:
    terms: List[str] = []
    for name, c in zip(names, w.tolist()):
        if name == "1":
            if abs(c) >= threshold:
                terms.append(f"{c:+.4f}")
            continue
        if abs(c) < threshold:
            continue
        terms.append(f"{c:+.4f}*{name}")
    if not terms:
        rhs = "0"
    else:
        rhs = " ".join(terms)
        if rhs.startswith("+"):
            rhs = rhs[1:].lstrip()
    return f"f(s) = {rhs}"


def sindy_fit_energy(
    X: np.ndarray,
    energy_fn: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    cfg: STLSQConfig,
) -> Tuple[np.ndarray, List[str]]:
    if env_name == "projectile":
        Theta, names = build_library_projectile(X)
        y = energy_fn(X)
    elif env_name == "pendulum":
        Theta, names = build_library_pendulum(X)
        y = energy_fn(X)
    else:
        raise ValueError(env_name)
    w = stlsq(Theta, y, cfg)
    return w, names

