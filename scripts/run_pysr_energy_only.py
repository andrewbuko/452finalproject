"""
PySR-only run for cluster environments.

This script intentionally does NOT import torch to avoid Julia init issues.
It runs PySR to recover:
  - Projectile energy: 0.5*vx^2 + 0.5*vy^2 + 9.81*y
  - Pendulum energy: 0.5*omega^2 - 9.81*cos(theta) (up to constant)
"""

import os

import numpy as np

from src.data_generation.pendulum import compute_energy_pendulum
from src.data_generation.projectile import compute_energy_projectile
from src.evaluation.symbolic_regression import analyze_discovered_equations, best_equation_string, discover_equation_pysr


def run_projectile():
    traj = np.load("data/projectile/trajectories.npy")
    X = traj.reshape(-1, 4)
    y = compute_energy_projectile(traj).reshape(-1)
    reg = discover_equation_pysr(
        X=X,
        y=y,
        variable_names=["x", "y", "vx", "vy"],
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        niterations=300,
    )
    print("\nPYSR — PROJECTILE (energy)")
    print("Best equation:", best_equation_string(reg))
    analyze_discovered_equations(reg, save_path="figures/pysr_pareto_projectile_energy.png")


def run_pendulum():
    traj = np.load("data/pendulum/trajectories.npy")
    X = traj.reshape(-1, 2)
    y = compute_energy_pendulum(traj).reshape(-1)
    reg = discover_equation_pysr(
        X=X,
        y=y,
        variable_names=["theta", "omega"],
        binary_operators=["+", "-", "*"],
        unary_operators=["square", "cos"],
        niterations=400,
    )
    print("\nPYSR — PENDULUM (energy)")
    print("Best equation:", best_equation_string(reg))
    analyze_discovered_equations(reg, save_path="figures/pysr_pareto_pendulum_energy.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    run_projectile()
    run_pendulum()

