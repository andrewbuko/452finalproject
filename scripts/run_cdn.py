import numpy as np

from src.data_generation.pendulum import compute_energy_pendulum
from src.data_generation.projectile import compute_energy_projectile
from src.data_generation.utils import scale_trajectories, train_val_split, unscale_trajectories
from src.evaluation.validate_cdn import validate_cdn
from src.training.train_cdn import CDNTrainConfig, train_cdn


def run_env(env_name: str, state_dim: int):
    raw = np.load(f"data/{env_name}/trajectories.npy")
    scaling = "standardize" if env_name == "pendulum" else "minmax01"
    scaled, stats = scale_trajectories(raw, mode=scaling)
    train_data, _ = train_val_split(scaled, val_fraction=0.1, seed=42)

    # Verify scaling
    flat = train_data.reshape(-1, state_dim)
    if scaling == "standardize":
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        print(f"[{env_name}] scaling=standardize  mean={mu}  std={sd}")
    else:
        mn = flat.min(axis=0)
        mx = flat.max(axis=0)
        print(f"[{env_name}] scaling=minmax01  min={mn}  max={mx}")
    # Basic shape/value sanity checks
    print(f"[{env_name}] raw_shape={raw.shape} scaled_shape={scaled.shape}")
    print(f"[{env_name}] sample_scaled_t0_first3={scaled[:3,0,:]}")

    # Optional: break the "any nonlinear transform of energy" ambiguity
    # by lightly aligning the learned invariant scale to analytical energy at t=0.
    energy0 = None
    cfg = CDNTrainConfig(
        env_name=env_name,
        state_dim=state_dim,
        epochs=80,
        lambda_var=0.5,
        epsilon=1.0,
        var_reg="softplus",
        # Keep f(s0) scale stable: mean~0, std~1
        lambda_scale=0.1,
        target_mean=0.0,
        target_std=1.0,
        std_min=0.8,
        std_max=1.2,
        grad_clip=1.0,
        log_grad_norm=True,
        log_every=1,
    )
    if env_name == "pendulum":
        denorm_train = unscale_trajectories(train_data, stats, mode=scaling)
        energy0 = compute_energy_pendulum(denorm_train)[:, 0]
        cfg.lambda_align = 0.2
    else:
        # Keep projectile fully unsupervised (just conservation + anti-collapse)
        cfg.lambda_align = 0.0

    model, _history = train_cdn(train_data, cfg, energy0_np=energy0)

    if env_name == "projectile":
        energy_fn = lambda t: compute_energy_projectile(
            unscale_trajectories(t, stats, mode=scaling)
        )
    elif env_name == "pendulum":
        energy_fn = lambda t: compute_energy_pendulum(unscale_trajectories(t, stats, mode=scaling))
    else:
        raise ValueError(f"Unknown env: {env_name}")

    r2 = validate_cdn(model, scaled, energy_fn, env_name=env_name)
    return r2


if __name__ == "__main__":
    r2_proj = run_env("projectile", state_dim=4)
    r2_pend = run_env("pendulum", state_dim=2)
    print(f"\nFinal R^2 - Projectile: {r2_proj:.4f}, Pendulum: {r2_pend:.4f}")

