from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit


def probe_single_dimension(
    model,
    base_state: np.ndarray,
    dim_idx: int,
    dim_range: Tuple[float, float],
    n_points: int = 200,
    device: str = "cpu",
):
    dim_values = np.linspace(dim_range[0], dim_range[1], n_points, dtype=np.float64)
    states = np.repeat(base_state[None, :], n_points, axis=0).astype(np.float64)
    states[:, dim_idx] = dim_values

    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(states, dtype=torch.float32, device=device)).detach().cpu().numpy()

    return dim_values, y


def probe_all_dimensions(model, trajs_np: np.ndarray, dim_names, device: str = "cpu"):
    # median state of the dataset is the base; sweep one dimension at a time
    flat = trajs_np.reshape(-1, trajs_np.shape[-1])
    base = np.median(flat, axis=0)
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for i, name in enumerate(dim_names):
        lo = np.percentile(flat[:, i], 5)
        hi = np.percentile(flat[:, i], 95)
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = float(np.nanmin(flat[:, i])), float(np.nanmax(flat[:, i]))
        if lo == hi:
            # dimension is effectively constant (e.g. vx fixed); make a tiny sweep range
            eps = 1e-3 if lo == 0 else abs(lo) * 1e-3
            lo, hi = float(lo - eps), float(hi + eps)
        results[name] = probe_single_dimension(model, base, i, (lo, hi), n_points=200, device=device)

    return results


def fit_symbolic_to_probe(dim_values: np.ndarray, f_values: np.ndarray, mode: str, max_degree: int = 4):
    """fit a closed form to a 1d probe sweep. mode in {constant, linear, quadratic, poly, cosine}."""
    x = dim_values.astype(np.float64)
    y = f_values.astype(np.float64)

    if mode == "constant":
        c = float(np.mean(y))
        yhat = np.full_like(y, c)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)
        return {"mode": mode, "coef": [c], "r2": r2, "yhat": yhat, "eq": f"{c:.4f}"}

    if mode in ("linear", "quadratic", "poly"):
        deg = 1 if mode == "linear" else (2 if mode == "quadratic" else max_degree)
        # nearly-constant x leads to singular lstsq
        if np.std(x) < 1e-10:
            return fit_symbolic_to_probe(dim_values, f_values, mode="constant")
        try:
            coef = np.polyfit(x, y, deg=deg)
        except np.linalg.LinAlgError:
            return fit_symbolic_to_probe(dim_values, f_values, mode="constant")
        yhat = np.polyval(coef, x)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)

        terms = []
        p = deg
        for a in coef:
            if p == 0:
                terms.append(f"{a:.4f}")
            elif p == 1:
                terms.append(f"{a:.4f}*x")
            else:
                terms.append(f"{a:.4f}*x^{p}")
            p -= 1
        eq = " + ".join(terms)
        return {"mode": mode, "coef": coef.tolist(), "r2": r2, "yhat": yhat, "eq": eq}

    if mode == "cosine":
        def fcos(t, a, b):
            return a * np.cos(t) + b

        popt, _ = curve_fit(fcos, x, y, p0=(1.0, 0.0), maxfev=20000)
        yhat = fcos(x, *popt)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)
        a, b = popt
        eq = f"{a:.4f}*cos(x) + {b:.4f}"
        return {"mode": mode, "coef": [float(a), float(b)], "r2": r2, "yhat": yhat, "eq": eq}

    raise ValueError(f"Unknown mode: {mode}")


@dataclass
class ProjectileProbeSummary:
    x_const: float
    y_slope: float
    vx_quad: float
    vy_quad: float
    ratio_slope_over_quad: float


def probe_projectile(model, trajs_scaled: np.ndarray, device: str = "cpu"):
    dim_names = ["x", "y", "vx", "vy"]
    probes = probe_all_dimensions(model, trajs_scaled, dim_names, device=device)

    fits = {}
    fits["x"] = fit_symbolic_to_probe(*probes["x"], mode="constant")
    fits["y"] = fit_symbolic_to_probe(*probes["y"], mode="linear")
    fits["vx"] = fit_symbolic_to_probe(*probes["vx"], mode="quadratic")
    fits["vy"] = fit_symbolic_to_probe(*probes["vy"], mode="quadratic")

    # polyfit returns [a, b, c] for ax^2 + bx + c
    y_slope = float(fits["y"]["coef"][0])
    vx_quad = float(fits["vx"]["coef"][0])
    vy_quad = float(fits["vy"]["coef"][0])
    ratio = float(y_slope / (vx_quad + 1e-12))

    summary = ProjectileProbeSummary(
        x_const=float(fits["x"]["coef"][0]),
        y_slope=y_slope,
        vx_quad=vx_quad,
        vy_quad=vy_quad,
        ratio_slope_over_quad=ratio,
    )
    return probes, fits, summary


@dataclass
class PendulumProbeSummary:
    omega_quad: float
    theta_cos_amp: float
    ratio_amp_over_quad: float


def probe_pendulum(model, trajs_scaled: np.ndarray, device: str = "cpu"):
    dim_names = ["theta", "omega"]
    probes = probe_all_dimensions(model, trajs_scaled, dim_names, device=device)

    fits = {}
    fits["theta"] = fit_symbolic_to_probe(*probes["theta"], mode="cosine")
    fits["omega"] = fit_symbolic_to_probe(*probes["omega"], mode="quadratic")

    omega_quad = float(fits["omega"]["coef"][0])
    theta_amp = float(fits["theta"]["coef"][0])
    ratio = float(theta_amp / (omega_quad + 1e-12))

    return probes, fits, PendulumProbeSummary(omega_quad, theta_amp, ratio)


def plot_probes(probe_results, fits, env_name: str, save_dir: str = "figures"):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})
    os.makedirs(save_dir, exist_ok=True)

    keys = list(probe_results.keys())
    n = len(keys)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        axes = np.array(axes).flatten()

    for ax, k in zip(axes, keys):
        x, y = probe_results[k]
        ax.plot(x, y, "b.", markersize=2, alpha=0.6, label="CDN probe")
        if fits and k in fits:
            ax.plot(x, fits[k]["yhat"], "r--", linewidth=2, label=f"fit (R2={fits[k]['r2']:.3f})")
            ax.set_title(f"{k}: {fits[k]['eq']}")
        else:
            ax.set_title(k)
        ax.set_xlabel(k)
        ax.set_ylabel("f(s)")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(keys) :]:
        ax.axis("off")

    fig.tight_layout()
    out = os.path.join(save_dir, f"probe_{env_name}.png")
    fig.savefig(out)
    plt.close(fig)

