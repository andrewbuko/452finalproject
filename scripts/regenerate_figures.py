#!/usr/bin/env python3
"""regenerate per-model validation figures from saved checkpoints.

loads each saved (cdn|structured_energy|polynomial)_<env>_best.pt and writes
correctly-named scatter plots to <save_dir>. no retraining; runs in ~minutes
on cpu.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.pendulum import compute_energy_pendulum
from src.data_generation.projectile import compute_energy_projectile
from src.data_generation.spring_mass import compute_energy_spring
from src.data_generation.utils import normalize_trajectories
from src.evaluation.validate_cdn import validate_conservation_model
from src.models.cdn import ConservationDiscoveryNetwork
from src.models.polynomial_cdn import PolynomialConservation
from src.models.structured_energy import StructuredEnergyNetwork


ENVS = [
    {
        "name": "projectile",
        "state_dim": 4,
        "pos_dims": [0, 1],
        "vel_dims": [2, 3],
        "trig_dims": None,
        "energy": compute_energy_projectile,
    },
    {
        "name": "pendulum",
        "state_dim": 2,
        "pos_dims": [0],
        "vel_dims": [1],
        "trig_dims": [0],
        "energy": compute_energy_pendulum,
    },
    {
        "name": "spring_mass",
        "state_dim": 2,
        "pos_dims": [0],
        "vel_dims": [1],
        "trig_dims": None,
        "energy": compute_energy_spring,
    },
]


def load_cdn(state_dim, ckpt, device):
    m = ConservationDiscoveryNetwork(state_dim=state_dim, hidden_dim=256, n_layers=4)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.to(device)


def load_structured(state_dim, pos_dims, vel_dims, ckpt, device):
    m = StructuredEnergyNetwork(
        state_dim=state_dim, pos_dims=pos_dims, vel_dims=vel_dims,
        hidden_dim=128, n_layers=2,
    )
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.to(device)


def load_polynomial(state_dim, trig_dims, ckpt, device):
    m = PolynomialConservation(state_dim=state_dim, degree=2, include_trig_dims=trig_dims)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.to(device)


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"device={device}  data={args.data_dir}  models={args.model_dir}  out={args.save_dir}")

    for env in ENVS:
        traj_path = os.path.join(args.data_dir, env["name"], "trajectories.npy")
        if not os.path.exists(traj_path):
            print(f"skip {env['name']}: {traj_path} missing")
            continue
        trajs = np.load(traj_path)
        if trajs.shape[0] > args.n_trajectories:
            trajs = trajs[: args.n_trajectories]
        if args.noise_std and float(args.noise_std) > 0:
            rng = np.random.RandomState(123)
            trajs = trajs + rng.normal(scale=float(args.noise_std), size=trajs.shape).astype(trajs.dtype, copy=False)
        print(f"\n=== {env['name']} ({trajs.shape}) ===")

        cdn_ckpt = os.path.join(args.model_dir, f"cdn_{env['name']}_best.pt")
        if os.path.exists(cdn_ckpt):
            trajs_norm, stats = normalize_trajectories(trajs)

            def energy_from_norm(t_norm, _e=env["energy"], _s=stats):
                return _e(t_norm * _s["range"] + _s["min"])

            cdn = load_cdn(env["state_dim"], cdn_ckpt, device)
            validate_conservation_model(
                cdn, trajs_norm, energy_from_norm, env["name"],
                model_name="CDN", save_dir=args.save_dir, device=device,
            )
        else:
            print(f"  cdn ckpt missing: {cdn_ckpt}")

        se_ckpt = os.path.join(args.model_dir, f"structured_energy_{env['name']}_best.pt")
        if os.path.exists(se_ckpt):
            se = load_structured(env["state_dim"], env["pos_dims"], env["vel_dims"], se_ckpt, device)
            validate_conservation_model(
                se, trajs, env["energy"], env["name"],
                model_name="StructuredEnergy", save_dir=args.save_dir, device=device,
            )
        else:
            print(f"  structured ckpt missing: {se_ckpt}")

        poly_ckpt = os.path.join(args.model_dir, f"polynomial_{env['name']}_best.pt")
        if os.path.exists(poly_ckpt):
            poly = load_polynomial(env["state_dim"], env["trig_dims"], poly_ckpt, device)
            validate_conservation_model(
                poly, trajs, env["energy"], env["name"],
                model_name="Polynomial", save_dir=args.save_dir, device=device,
            )
        else:
            print(f"  polynomial ckpt missing: {poly_ckpt}")

    print(f"\nfigures written to {args.save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--save_dir", default="figures")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_trajectories", type=int, default=200_000,
                   help="cap on trajectories used for validation; matches a subset of training data")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="apply matching gaussian noise so r2 numbers match the json")
    args = p.parse_args()
    main(args)
