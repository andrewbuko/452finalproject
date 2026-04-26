from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _import_pysr():
    try:
        from pysr import PySRRegressor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PySR is not installed or failed to import. Install with `pip install pysr` "
            "(also requires a working Julia installation)."
        ) from e
    return PySRRegressor


def pysr_available() -> bool:
    # Important: importing `pysr` may trigger Julia initialization (slow/hang on some setups).
    # So only check for package presence here.
    import importlib.util

    return importlib.util.find_spec("pysr") is not None


def run_symbolic_regression(
    trajs_np,
    target_fn,
    variable_names,
    env_name,
    save_dir="results",
    n_samples=50000,
    niterations=500,
    binary_operators=None,
    unary_operators=None,
    maxsize=20,
    seed=42,
):
    """
    Run PySR symbolic regression.
    Args:
      trajs_np: (N, T, D) RAW trajectories
      target_fn: callable(states_2D) -> targets_1D
      variable_names: list[str]
    Returns:
      reg, best_equation_string
    """
    if not pysr_available():
        print("PySR not installed. Run: pip install pysr")
        return None, "PySR not available"

    PySRRegressor = _import_pysr()

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["square"]

    all_states = trajs_np.reshape(-1, trajs_np.shape[-1])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_states), size=min(int(n_samples), len(all_states)), replace=False)
    X = all_states[idx].astype(np.float64)
    y = target_fn(X).astype(np.float64)

    print(f"\nRunning PySR on {env_name}...")
    print(f" X shape: {X.shape}, y shape: {y.shape}")
    print(f" Operators: binary={binary_operators}, unary={unary_operators}")
    print(f" Variables: {variable_names}")
    print(f" Iterations: {niterations}")

    reg = PySRRegressor(
        niterations=int(niterations),
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        variable_names=variable_names,
        maxsize=int(maxsize),
        populations=60,
        population_size=50,
        batching=True,
        batch_size=min(5000, len(X)),
        parsimony=0.003,
        weight_optimize=0.01,
        turbo=True,
        progress=True,
        temp_equation_file=True,
        random_state=int(seed),
        procs=8,
        multithreading=True,
    )
    reg.fit(X, y)

    best_eq = best_equation_string(reg)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    try:
        equations = reg.equations_
        eq_rows = [
            {
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "equation": str(row["equation"]),
            }
            for _, row in equations.iterrows()
        ]
    except Exception:
        eq_rows = []

    out = {
        "environment": env_name,
        "best_equation": best_eq,
        "variable_names": list(variable_names),
        "n_samples": int(len(X)),
        "niterations": int(niterations),
        "equations": eq_rows,
    }
    with open(os.path.join(save_dir, f"pysr_results_{env_name}.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(out, f, indent=2)

    return reg, best_eq


def _eval_model_on_states(model, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return y.astype(np.float64)


def discover_equation_pysr(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: List[str],
    binary_operators: List[str],
    unary_operators: List[str],
    niterations: int = 200,
    model_selection: str = "best",
    equation_file: Optional[str] = None,
):
    PySRRegressor = _import_pysr()
    reg = PySRRegressor(
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        model_selection=model_selection,
        maxsize=30,
        populations=20,
        verbosity=0,
        progress=False,
    )
    reg.fit(X, y, variable_names=variable_names)

    if equation_file is not None:
        os.makedirs(os.path.dirname(equation_file) or ".", exist_ok=True)
        with open(equation_file, "w", encoding="utf-8") as f:
            f.write(str(reg.sympy()) + "\n")

    return reg


def sample_states_and_targets(
    trajectories_raw: np.ndarray,
    n_samples: int,
    target_fn: Callable[[np.ndarray], np.ndarray],
):
    flat = trajectories_raw.reshape(-1, trajectories_raw.shape[-1])
    n = min(n_samples, flat.shape[0])
    idx = np.random.RandomState(0).choice(flat.shape[0], size=n, replace=False)
    X = flat[idx].astype(np.float64)
    y = target_fn(X)
    if y.ndim == 2:
        # if someone passes a trajectory-energy function, take t=0
        y = y[:, 0]
    y = y.astype(np.float64)
    return X, y


def best_equation_string(reg) -> str:
    # Best equation chosen by PySR's model_selection
    try:
        return str(reg.sympy())
    except Exception:
        return str(reg)


@dataclass
class ParetoPoint:
    complexity: int
    loss: float
    equation: str


def analyze_discovered_equations(reg, save_path: str = "figures/pysr_pareto.png"):
    """
    Save a Pareto front plot and print the top candidates.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    eqs = reg.equations_
    pts: List[ParetoPoint] = []
    for _, row in eqs.iterrows():
        pts.append(ParetoPoint(int(row["complexity"]), float(row["loss"]), str(row["equation"])))

    # print a few
    pts_sorted = sorted(pts, key=lambda p: (p.loss, p.complexity))[:10]
    print("\nTop PySR candidates (loss, complexity):")
    for p in pts_sorted:
        print(f"  loss={p.loss:.4e}  complexity={p.complexity:2d}  eq={p.equation}")

    # plot Pareto
    xs = [p.complexity for p in pts]
    ys = [p.loss for p in pts]
    plt.figure(figsize=(7, 5), dpi=150)
    plt.scatter(xs, ys, s=14, alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Complexity")
    plt.ylabel("Loss (log)")
    plt.title("PySR Pareto Front")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Saved", save_path)


def validate_discovered_equation(
    reg,
    trajectories_raw: np.ndarray,
    cdn_model,
    device: str = "cpu",
    save_path: str = "figures/equation_validation.png",
):
    """
    Compare equation output with CDN output and (optionally) analytical energy.
    Plots time-series constancy and correlations.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    T = trajectories_raw.shape[1]
    flat = trajectories_raw.reshape(-1, trajectories_raw.shape[-1])
    y_cdn = _eval_model_on_states(cdn_model, flat, device=device).reshape(-1, T)
    y_eq = reg.predict(flat).reshape(-1, T)

    # per-trajectory std (smaller is better conservation)
    cdn_std = np.std(y_cdn, axis=1)
    eq_std = np.std(y_eq, axis=1)

    plt.figure(figsize=(10, 7), dpi=150)
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(np.mean(y_cdn, axis=0), label="CDN mean")
    ax1.plot(np.mean(y_eq, axis=0), label="Eq mean", linestyle="--")
    ax1.set_title("Mean invariant across time")
    ax1.set_xlabel("t index")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(cdn_std, bins=40, alpha=0.6, label="CDN std per traj")
    ax2.hist(eq_std, bins=40, alpha=0.6, label="Eq std per traj")
    ax2.set_title("Conservation check: std over time per trajectory")
    ax2.set_xlabel("std")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Saved", save_path)

