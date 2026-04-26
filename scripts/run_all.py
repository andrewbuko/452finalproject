#!/usr/bin/env python3
"""
Master script: runs the entire experiment pipeline.
Usage:
  python scripts/run_all.py --device cuda --data_dir data --save_dir results
On SLURM cluster:
  sbatch cluster/submit_job.sh
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(args):
    device = torch.device(args.device)
    print("=" * 70)
    print("CONSERVATION DISCOVERY AND SYMBOLIC REGRESSION EXPERIMENT")
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Save dir: {args.save_dir}")
    print("=" * 70)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    from src.data_generation.projectile import compute_energy_projectile, generate_projectile_data
    from src.data_generation.pendulum import compute_energy_pendulum, generate_pendulum_data
    from src.data_generation.spring_mass import compute_energy_spring, generate_spring_mass_data
    from src.data_generation.utils import normalize_trajectories, train_val_split
    from src.evaluation.hero_figure import create_hero_figure
    from src.evaluation.probe_cdn import probe_all_dimensions
    from src.evaluation.symbolic_regression import run_symbolic_regression
    from src.evaluation.validate_cdn import validate_conservation_model
    from src.models.cdn import ConservationDiscoveryNetwork
    from src.training.train_cdn import train_conservation_model
    from src.training.train_polynomial import train_polynomial_model

    ENVIRONMENTS = [
        {
            "name": "projectile",
            "state_dim": 4,
            "var_names": ["x", "y", "vx", "vy"],
            "default_n": 1_000_000,
            "poly_degree": 2,
            "trig_dims": None,
            "pysr_unary": ["square"],
            "pysr_binary": ["+", "-", "*"],
            "generate": generate_projectile_data,
            "energy": compute_energy_projectile,
            "known": "E = 0.5*vx^2 + 0.5*vy^2 + 9.81*y",
        },
        {
            "name": "pendulum",
            "state_dim": 2,
            "var_names": ["theta", "omega"],
            "default_n": 500_000,
            "poly_degree": 2,
            "trig_dims": [0],
            "pysr_unary": ["square", "cos"],
            "pysr_binary": ["+", "-", "*"],
            "generate": generate_pendulum_data,
            "energy": compute_energy_pendulum,
            "known": "H = 0.5*omega^2 - 9.81*cos(theta)",
        },
        {
            "name": "spring_mass",
            "state_dim": 2,
            "var_names": ["x", "v"],
            "default_n": 500_000,
            "poly_degree": 2,
            "trig_dims": None,
            "pysr_unary": ["square"],
            "pysr_binary": ["+", "-", "*"],
            "generate": generate_spring_mass_data,
            "energy": compute_energy_spring,
            "known": "E = 5.0*x^2 + 0.5*v^2",
        },
    ]

    start_time = time.time()
    all_results = {}
    discovered_eqs = []
    known_eqs = []
    env_names = []
    coeff_errors = []

    for env in ENVIRONMENTS:
        print("\n" + "#" * 70)
        print(f"# ENVIRONMENT: {env['name'].upper()}")
        print(f"# Known: {env['known']}")
        print("#" * 70)

        env_dir = os.path.join(args.data_dir, env["name"])
        traj_path = os.path.join(env_dir, "trajectories.npy")

        n_traj = int(min(env["default_n"], args.n_trajectories))

        # Phase 1: Data generation / load
        if (not os.path.exists(traj_path)) or args.regenerate:
            print("Generating data...")
            if env["name"] == "pendulum":
                trajs = env["generate"](n_trajectories=n_traj, n_timesteps=args.n_timesteps, dt=args.dt, save_dir=env_dir)
            else:
                trajs = env["generate"](n_trajectories=n_traj, n_timesteps=args.n_timesteps, dt=args.dt, save_dir=env_dir)
        else:
            trajs = np.load(traj_path)
            print(f"Loaded {env['name']} data: {trajs.shape}")

        # Sanity check: energy conservation of generated data
        E = env["energy"](trajs)
        print(f"{env['name']} energy conservation: std={E.std(axis=1).mean():.2e}")

        # Phase 2: CDN training on normalized data
        print("Training CDN (normalized)...")
        trajs_norm, stats = normalize_trajectories(trajs)
        trajs_train, _ = train_val_split(trajs_norm, val_fraction=0.1, seed=42)
        cdn = ConservationDiscoveryNetwork(state_dim=env["state_dim"], hidden_dim=256, n_layers=4).to(device)
        cdn, _ = train_conservation_model(
            model=cdn,
            trajectories_np=trajs_train,
            save_path=os.path.join("models", f"cdn_{env['name']}_best.pt"),
            lr=1e-3,
            epochs=args.cdn_epochs,
            batch_size=512,
            lambda_var=0.5,
            epsilon=10.0,
            device=device,
            model_name=f"CDN {env['name']}",
        )

        # Validate CDN: compare learned f(s) to analytical energy (on raw units)
        def energy_from_norm(t_norm):
            t_raw = t_norm * stats["range"] + stats["min"]
            return env["energy"](t_raw)

        cdn_val = validate_conservation_model(
            cdn,
            trajs_norm,
            energy_from_norm,
            env["name"],
            model_name="CDN",
            save_dir="figures",
            device=device,
        )
        all_results[f"cdn_{env['name']}"] = cdn_val

        # Phase 3: Polynomial model on RAW data
        print("Training polynomial model (RAW)...")
        poly_model, poly_eq = train_polynomial_model(
            trajs,
            state_dim=env["state_dim"],
            env_name=env["name"],
            var_names=env["var_names"],
            save_dir="models",
            degree=env["poly_degree"],
            include_trig_dims=env["trig_dims"],
            lr=args.poly_lr,
            epochs=args.poly_epochs,
            batch_size=args.poly_batch_size,
            device=device,
            warmup_epochs=args.poly_warmup_epochs,
        )
        discovered_eqs.append(poly_eq)
        known_eqs.append(env["known"])
        env_names.append(env["name"])

        poly_val = validate_conservation_model(
            poly_model,
            trajs,
            env["energy"],
            env["name"],
            model_name="Polynomial",
            save_dir="figures",
            device=device,
        )
        all_results[f"poly_{env['name']}"] = poly_val

        # Phase 4: Probing (Polynomial)
        probe = probe_all_dimensions(poly_model, trajs, env["var_names"], device=str(device))
        all_results[f"probe_{env['name']}"] = probe

        # Phase 5: PySR (optional)
        if not args.skip_pysr:
            def target_energy(states_2d):
                return env["energy"](states_2d.reshape(1, 1, -1)).flatten()

            _, pysr_best = run_symbolic_regression(
                trajs,
                target_fn=target_energy,
                variable_names=env["var_names"],
                env_name=env["name"],
                save_dir=args.save_dir,
                n_samples=args.pysr_samples,
                niterations=args.pysr_iterations,
                binary_operators=env["pysr_binary"],
                unary_operators=env["pysr_unary"],
                maxsize=args.pysr_maxsize,
                seed=42,
            )
            all_results[f"pysr_{env['name']}"] = pysr_best

        # Placeholder coefficient error (computed robustly in hero_figure stage later)
        coeff_errors.append(0.0)

    # Hero figure
    create_hero_figure(discovered_eqs, known_eqs, env_names, coeff_errors, save_dir="figures")

    elapsed = time.time() - start_time
    all_results["total_time_seconds"] = elapsed
    with open(os.path.join(args.save_dir, "experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("=" * 70)
    print(f"EXPERIMENT COMPLETE - {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to {args.save_dir}/experiment_results.json")
    print("Figures saved to figures/")
    print("Models saved to models/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--n_trajectories", type=int, default=1_000_000)
    parser.add_argument("--n_timesteps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip_pysr", action="store_true")

    parser.add_argument("--cdn_epochs", type=int, default=200)
    parser.add_argument("--poly_epochs", type=int, default=2000)
    parser.add_argument("--poly_lr", type=float, default=0.005)
    parser.add_argument("--poly_batch_size", type=int, default=4096)
    parser.add_argument("--poly_warmup_epochs", type=int, default=200)

    parser.add_argument("--pysr_iterations", type=int, default=500)
    parser.add_argument("--pysr_samples", type=int, default=50000)
    parser.add_argument("--pysr_maxsize", type=int, default=20)
    main(parser.parse_args())

