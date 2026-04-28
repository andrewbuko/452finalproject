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
    from src.evaluation.validate_diffusion import evaluate_diffusion_rollout
    from src.evaluation.sindy import STLSQConfig, format_sparse_equation, sindy_fit_energy
    from src.training.train_cdn import CDNTrainConfig, train_cdn
    from src.training.train_diffusion import DiffusionTrainConfig, train_diffusion_transition
    from src.training.train_polynomial import train_polynomial_model
    from src.training.train_structured_energy import StructuredEnergyConfig, train_structured_energy

    ENVIRONMENTS = [
        {
            "name": "projectile",
            "state_dim": 4,
            "var_names": ["x", "y", "vx", "vy"],
            "default_n": 1_000_000,
            "poly_degree": 2,
            "trig_dims": None,
            # PySR discovers equations from operators (not a fixed polynomial library)
            "pysr_unary": ["square"],
            "pysr_binary": ["+", "-", "*", "/"],
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
            "pysr_unary": ["square", "sin", "cos"],
            "pysr_binary": ["+", "-", "*", "/"],
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
            "pysr_binary": ["+", "-", "*", "/"],
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
            # Respect --n_trajectories even when loading an existing dataset.
            # (The on-disk dataset may contain many more trajectories.)
            if trajs.shape[0] > n_traj:
                trajs = trajs[:n_traj]
                print(f"Subsampled {env['name']} to: {trajs.shape}")

        # Sanity check: energy conservation of generated data
        E = env["energy"](trajs)
        print(f"{env['name']} energy conservation: std={E.std(axis=1).mean():.2e}")

        # Phase 2: CDN training on normalized data
        print("Training CDN (normalized)...")
        trajs_norm, stats = normalize_trajectories(trajs)
        # Need a consistent split for both trajectories and energy0 alignment targets.
        rng = np.random.RandomState(42)
        nN = trajs_norm.shape[0]
        perm = rng.permutation(nN)
        split = int(nN * 0.9)
        train_idx = perm[:split]
        trajs_train = trajs_norm[train_idx]
        energy0_train = E[train_idx, 0]
        shared_epochs = int(args.epochs_all) if args.epochs_all is not None else None

        # CDN with alignment + scale constraints so f(s) correlates with analytical energy.
        cdn_cfg = CDNTrainConfig(
            env_name=env["name"],
            state_dim=env["state_dim"],
            hidden_dim=256,
            n_layers=4,
            lr=1e-3,
            epochs=int(shared_epochs or args.cdn_epochs),
            batch_size=512,
            lambda_var=float(args.cdn_lambda_var),
            epsilon=float(args.cdn_epsilon),
            var_reg="hinge",
            lambda_scale=float(args.cdn_lambda_scale),
            target_mean=float(args.cdn_target_mean),
            std_min=float(args.cdn_std_min),
            std_max=float(args.cdn_std_max),
            lambda_align=float(args.cdn_lambda_align),
            grad_clip=1.0,
            log_every=50,
            save_dir="models",
            device=str(device),
        )
        cdn, _ = train_cdn(trajs_train, cdn_cfg, energy0_np=energy0_train)

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

        # Phase 2b: Structured Energy Network baseline (optional)
        if args.run_structured_energy:
            print("Training structured energy network (RAW)...")
            if env["name"] == "projectile":
                pos_dims, vel_dims = [0, 1], [2, 3]
            elif env["name"] == "pendulum":
                pos_dims, vel_dims = [0], [1]
            else:  # spring_mass
                pos_dims, vel_dims = [0], [1]

            structured_cfg = StructuredEnergyConfig(
                env_name=env["name"],
                state_dim=env["state_dim"],
                pos_dims=pos_dims,
                vel_dims=vel_dims,
                device=str(device),
                epochs=int(shared_epochs or args.structured_epochs),
                lambda_var=float(args.structured_lambda_var),
                lambda_energy=float(args.structured_lambda_energy),
            )
            structured_model, _ = train_structured_energy(trajs, energy0_np=E[:, 0], cfg=structured_cfg)
            structured_val = validate_conservation_model(
                structured_model,
                trajs,
                env["energy"],
                env["name"],
                model_name="StructuredEnergy",
                save_dir="figures",
                device=device,
            )
            all_results[f"structured_energy_{env['name']}"] = structured_val
        else:
            structured_model = None

        # Phase 3: Polynomial model on RAW data (with energy alignment to pin scale)
        print("Training polynomial model (RAW)...")
        energy0 = E[:, 0].astype(np.float32, copy=False)
        poly_model, poly_eq = train_polynomial_model(
            trajs,
            state_dim=env["state_dim"],
            env_name=env["name"],
            var_names=env["var_names"],
            energy0_np=energy0,
            save_dir="models",
            degree=env["poly_degree"],
            include_trig_dims=env["trig_dims"],
            lr=args.poly_lr,
            epochs=int(shared_epochs or args.poly_epochs),
            batch_size=args.poly_batch_size,
            device=device,
            warmup_epochs=args.poly_warmup_epochs,
            lambda_var=float(args.poly_lambda_var),
            lambda_energy=float(args.poly_lambda_energy),
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

        # Phase 4c: Diffusion transition baseline (RAW -> rollout)
        # Condition on CDN's learned invariant f(s) and apply manifold projection via StructuredEnergyNetwork (if enabled).
        if args.run_diffusion:
            print("Training diffusion transition model (RAW -> rollout, conditioned on CDN f(s))...")
            trajs_train_raw, trajs_val_raw = train_val_split(trajs, val_fraction=0.1, seed=42)

            # Build conditioning values for each transition state in the training set: c = f_cdn(s_t)
            cdn_device = next(cdn.parameters()).device
            cdn.eval()
            flat_raw = trajs_train_raw[:, :-1, :].reshape(-1, env["state_dim"]).astype(np.float32)
            flat_norm = (flat_raw - stats["min"][None, :]) / stats["range"][None, :]
            cond_chunks = []
            bs = 16384
            with torch.no_grad():
                for i in range(0, flat_norm.shape[0], bs):
                    x = torch.tensor(flat_norm[i : i + bs], dtype=torch.float32, device=cdn_device)
                    cond_chunks.append(cdn(x).detach().cpu().numpy().astype(np.float32))
            cond_np = np.concatenate(cond_chunks, axis=0).reshape(-1, 1)

            diff_cfg = DiffusionTrainConfig(
                env_name=env["name"],
                state_dim=env["state_dim"],
                K=args.diffusion_steps,
                hidden_dim=args.diffusion_hidden_dim,
                time_emb_dim=args.diffusion_time_emb_dim,
                lr=args.diffusion_lr,
                epochs=int(shared_epochs or args.diffusion_epochs),
                batch_size=args.diffusion_batch_size,
                max_train_trajectories=args.diffusion_max_train_trajectories,
                device=str(device),
                save_dir="models",
                log_every=max(1, int((shared_epochs or args.diffusion_epochs)) // 10),
                cond_dim=1,
            )
            diff_model, diff_stats, _ = train_diffusion_transition(trajs_train_raw, cfg=diff_cfg, cond_np=cond_np)

            def cond_fn_raw(s_raw_2d: np.ndarray) -> np.ndarray:
                s_norm_2d = (s_raw_2d - stats["min"][None, :]) / stats["range"][None, :]
                with torch.no_grad():
                    x = torch.tensor(s_norm_2d, dtype=torch.float32, device=cdn_device)
                    return cdn(x).detach().cpu().numpy().astype(np.float32)

            diff_metrics = evaluate_diffusion_rollout(
                model=diff_model,
                trajs_raw=trajs_val_raw,
                energy_fn_np=env["energy"],
                env_name=env["name"],
                stats=diff_stats,
                n_rollouts=args.diffusion_eval_rollouts,
                device=str(device),
                save_dir="figures",
                cond_fn_raw=cond_fn_raw,
                energy_model=structured_model if (structured_model is not None and args.diffusion_project_manifold) else None,
                project_steps=int(args.diffusion_project_steps),
            )
            all_results[f"diffusion_{env['name']}"] = diff_metrics

        # Phase 4b: SINDy (STLSQ) sparse regression baseline (optional)
        if args.run_sindy:
            flat = trajs.reshape(-1, env["state_dim"]).astype(np.float64)
            rng = np.random.RandomState(0)
            n = min(int(args.sindy_samples), flat.shape[0])
            X = flat[rng.choice(flat.shape[0], size=n, replace=False)]

            def energy_on_states(X2d: np.ndarray) -> np.ndarray:
                # X2d is (N, D). The env energy function expects (N, T, D),
                # so treat each state as a length-1 trajectory.
                X3d = X2d.reshape(-1, 1, env["state_dim"])
                return env["energy"](X3d)[:, 0].reshape(-1)

            w_s, names_s = sindy_fit_energy(
                X,
                energy_fn=energy_on_states,
                env_name=env["name"],
                cfg=STLSQConfig(threshold=float(args.sindy_threshold), max_iter=15, normalize_columns=True),
            )
            sindy_eq = format_sparse_equation(names_s, w_s, threshold=0.01)
            print("\nSINDy (STLSQ) equation:")
            print(" ", sindy_eq)
            all_results[f"sindy_{env['name']}"] = {"equation": sindy_eq, "weights": w_s.tolist(), "names": names_s}

        # Phase 5: PySR (optional)
        if not args.skip_pysr:
            def target_energy(states_2d):
                X3d = np.asarray(states_2d).reshape(-1, 1, env["state_dim"])
                return env["energy"](X3d)[:, 0].reshape(-1)

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
    parser.add_argument("--run_structured_energy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_sindy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_diffusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--epochs_all",
        type=int,
        default=512,
        help="If set, use this epoch count for all models.",
    )

    parser.add_argument("--cdn_epochs", type=int, default=512)
    parser.add_argument("--cdn_lambda_var", type=float, default=0.5)
    parser.add_argument("--cdn_epsilon", type=float, default=10.0)
    parser.add_argument("--cdn_lambda_align", type=float, default=0.2)
    parser.add_argument("--cdn_lambda_scale", type=float, default=0.05)
    parser.add_argument("--cdn_target_mean", type=float, default=0.0)
    parser.add_argument("--cdn_std_min", type=float, default=0.8)
    parser.add_argument("--cdn_std_max", type=float, default=1.2)
    parser.add_argument("--structured_epochs", type=int, default=512)
    parser.add_argument("--structured_lambda_var", type=float, default=1.0)
    parser.add_argument("--structured_lambda_energy", type=float, default=0.1)
    parser.add_argument("--poly_epochs", type=int, default=512)
    parser.add_argument("--poly_lr", type=float, default=0.005)
    parser.add_argument("--poly_batch_size", type=int, default=4096)
    parser.add_argument("--poly_warmup_epochs", type=int, default=200)
    parser.add_argument("--poly_lambda_var", type=float, default=1.0)
    parser.add_argument("--poly_lambda_energy", type=float, default=0.1)

    parser.add_argument("--pysr_iterations", type=int, default=500)
    parser.add_argument("--pysr_samples", type=int, default=50000)
    parser.add_argument("--pysr_maxsize", type=int, default=20)
    parser.add_argument("--sindy_samples", type=int, default=50000)
    parser.add_argument("--sindy_threshold", type=float, default=0.05)

    # Diffusion transition baseline (trajectory rollouts)
    parser.add_argument("--diffusion_epochs", type=int, default=512)
    parser.add_argument("--diffusion_lr", type=float, default=2e-4)
    parser.add_argument("--diffusion_batch_size", type=int, default=4096)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_time_emb_dim", type=int, default=64)
    parser.add_argument("--diffusion_max_train_trajectories", type=int, default=200000)
    parser.add_argument("--diffusion_eval_rollouts", type=int, default=256)
    parser.add_argument("--diffusion_project_manifold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion_project_steps", type=int, default=1)
    main(parser.parse_args())

