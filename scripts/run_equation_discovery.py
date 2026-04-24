import os
import warnings

import numpy as np

# Important: PySR uses Julia via juliacall. Importing torch first can crash/hang Julia init on some setups.
# So we delay importing torch until after we decide whether we can run PySR.
warnings.filterwarnings("ignore", message="torch was imported before juliacall")

from src.data_generation.pendulum import compute_energy_pendulum
from src.data_generation.projectile import compute_energy_projectile
from src.evaluation.sparse_regression import LassoConfig, fit_lasso, format_equation
from src.evaluation.sindy import STLSQConfig, format_sparse_equation, sindy_fit_energy
from src.evaluation.symbolic_regression import (
    analyze_discovered_equations,
    best_equation_string,
    discover_equation_pysr,
    pysr_available,
    sample_states_and_targets,
    validate_discovered_equation,
)
from src.models.cdn import ConservationDiscoveryNetwork
from src.evaluation.sindy import build_library_pendulum, build_library_projectile


def _load_cdn(ckpt_path: str, state_dim: int, device: str):
    m = ConservationDiscoveryNetwork(state_dim=state_dim).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    m.eval()
    return m


def _eval_cdn(model, X: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        y = model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return y.astype(np.float64)


def run_projectile(device: str):
    traj = np.load("data/projectile/trajectories.npy")  # RAW (unnormalized)
    flat = traj.reshape(-1, 4)
    E = compute_energy_projectile(traj).reshape(-1)

    print("\nLASSO (sparse regression) — PROJECTILE (target = analytical energy)")
    Theta, names = build_library_projectile(flat)
    w, b, _scaler = fit_lasso(Theta[:, 1:], E, LassoConfig(alpha=5e-4))  # drop constant column; keep intercept separately
    print(format_equation(names[1:], w, intercept=b, threshold=1e-2))
    print("Known: E = 0.5*vx^2 + 0.5*vy^2 + 9.81*y")

    # SINDy-style sparse regression on energy (library + STLSQ pruning)
    w_s, names_s = sindy_fit_energy(
        flat,
        energy_fn=lambda X: (0.5 * (X[:, 2] ** 2 + X[:, 3] ** 2) + 9.81 * X[:, 1]),
        env_name="projectile",
        cfg=STLSQConfig(threshold=0.05, max_iter=15, normalize_columns=True),
    )
    print("\nSINDy (STLSQ) — PROJECTILE (target = analytical energy)")
    print(format_sparse_equation(names_s, w_s, threshold=0.01))

    if not pysr_available():
        print("\nPYSR — PROJECTILE")
        print("PySR not available (requires PySR + Julia). Skipping symbolic regression.")
        return

    # PySR using analytical energy target
    X_e, y_e = sample_states_and_targets(
        traj,
        n_samples=10_000,
        target_fn=lambda X: (0.5 * (X[:, 2] ** 2 + X[:, 3] ** 2) + 9.81 * X[:, 1]),
    )
    reg_e = discover_equation_pysr(
        X=X_e,
        y=y_e,
        variable_names=["x", "y", "vx", "vy"],
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        niterations=200,
    )
    print("\nPYSR — PROJECTILE (target = analytical energy)")
    print("Best equation:", best_equation_string(reg_e))
    analyze_discovered_equations(reg_e, save_path="figures/pysr_pareto_projectile_energy.png")

    # PySR distillation of the CDN output
    cdn = _load_cdn("models/cdn_projectile_best.pt", 4, device)
    X_c = flat[np.random.RandomState(0).choice(flat.shape[0], size=10_000, replace=False)]
    y_c = _eval_cdn(cdn, X_c, device=device)
    # LASSO distillation
    Theta_c, names_c = build_library_projectile(X_c)
    w_c, b_c, _ = fit_lasso(Theta_c[:, 1:], y_c, LassoConfig(alpha=5e-4))
    print("\nLASSO — PROJECTILE (target = CDN output)")
    print(format_equation(names_c[1:], w_c, intercept=b_c, threshold=1e-2))
    reg_c = discover_equation_pysr(
        X=X_c,
        y=y_c,
        variable_names=["x", "y", "vx", "vy"],
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        niterations=200,
    )
    print("\nPYSR — PROJECTILE (target = CDN output)")
    print("Best equation:", best_equation_string(reg_c))
    analyze_discovered_equations(reg_c, save_path="figures/pysr_pareto_projectile_cdn.png")
    validate_discovered_equation(
        reg_c,
        trajectories_raw=traj,
        cdn_model=cdn,
        device=device,
        save_path="figures/equation_validation_projectile.png",
    )


def run_pendulum(device: str):
    traj = np.load("data/pendulum/trajectories.npy")  # RAW (unnormalized)
    flat = traj.reshape(-1, 2)
    E = compute_energy_pendulum(traj).reshape(-1)

    print("\nLASSO (sparse regression) — PENDULUM (target = analytical energy)")
    Theta, names = build_library_pendulum(flat)
    w, b, _ = fit_lasso(Theta[:, 1:], E, LassoConfig(alpha=5e-4))
    print(format_equation(names[1:], w, intercept=b, threshold=1e-2))
    print("Known: H = 0.5*omega^2 - 9.81*cos(theta)")

    w_s, names_s = sindy_fit_energy(
        flat,
        energy_fn=lambda X: compute_energy_pendulum(X.reshape(-1, 1, 2)).reshape(-1),
        env_name="pendulum",
        cfg=STLSQConfig(threshold=0.05, max_iter=15, normalize_columns=True),
    )
    print("\nSINDy (STLSQ) — PENDULUM (target = analytical energy)")
    print(format_sparse_equation(names_s, w_s, threshold=0.01))

    if not pysr_available():
        print("\nPYSR — PENDULUM")
        print("PySR not available (requires PySR + Julia). Skipping symbolic regression.")
        return

    X_e, y_e = sample_states_and_targets(
        traj,
        n_samples=10_000,
        target_fn=lambda X: compute_energy_pendulum(X.reshape(-1, 1, 2)).reshape(-1),
    )
    reg_e = discover_equation_pysr(
        X=X_e,
        y=y_e,
        variable_names=["theta", "omega"],
        binary_operators=["+", "-", "*"],
        unary_operators=["square", "cos"],
        niterations=250,
    )
    print("\nPYSR — PENDULUM (target = analytical energy)")
    print("Best equation:", best_equation_string(reg_e))
    analyze_discovered_equations(reg_e, save_path="figures/pysr_pareto_pendulum_energy.png")

    cdn = _load_cdn("models/cdn_pendulum_best.pt", 2, device)
    X_c = flat[np.random.RandomState(0).choice(flat.shape[0], size=10_000, replace=False)]
    y_c = _eval_cdn(cdn, X_c, device=device)
    Theta_c, names_c = build_library_pendulum(X_c)
    w_c, b_c, _ = fit_lasso(Theta_c[:, 1:], y_c, LassoConfig(alpha=5e-4))
    print("\nLASSO — PENDULUM (target = CDN output)")
    print(format_equation(names_c[1:], w_c, intercept=b_c, threshold=1e-2))
    reg_c = discover_equation_pysr(
        X=X_c,
        y=y_c,
        variable_names=["theta", "omega"],
        binary_operators=["+", "-", "*"],
        unary_operators=["square", "cos"],
        niterations=250,
    )
    print("\nPYSR — PENDULUM (target = CDN output)")
    print("Best equation:", best_equation_string(reg_c))
    analyze_discovered_equations(reg_c, save_path="figures/pysr_pareto_pendulum_cdn.png")
    validate_discovered_equation(
        reg_c,
        trajectories_raw=traj,
        cdn_model=cdn,
        device=device,
        save_path="figures/equation_validation_pendulum.png",
    )


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    import torch  # delayed import; safe if PySR needs to initialize Julia

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    run_projectile(device)
    run_pendulum(device)

