"""
Training script specifically for the polynomial conservation model.
KEY DIFFERENCE FROM CDN TRAINING:
- Uses RAW UNNORMALIZED trajectory data
- Lower learning rate (few parameters, we want precision)
- Prints the discovered equation after training
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.polynomial_cdn import PolynomialConservation
from src.training.train_cdn import conservation_loss


def train_polynomial_model(
    raw_trajectories_np,
    state_dim,
    env_name,
    var_names,
    energy0_np=None,
    save_dir="models",
    degree=2,
    include_trig_dims=None,
    lr=0.005,
    epochs=2000,
    batch_size=4096,
    lambda_var=1.0,
    lambda_energy=0.1,
    epsilon=None,
    device=None,
    warmup_epochs=200,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-set epsilon based on initial-state energy variance on a subset.
    if epsilon is None:
        x = raw_trajectories_np[:5000]
        if state_dim == 4:
            E = 0.5 * (x[:, 0, 2] ** 2 + x[:, 0, 3] ** 2) + 9.81 * x[:, 0, 1]
        elif include_trig_dims:
            E = 0.5 * x[:, 0, 1] ** 2 - 9.81 * np.cos(x[:, 0, 0])
        else:
            E = 0.5 * x[:, 0, 1] ** 2
        epsilon = max(float(E.var() * 0.1), 1.0)
        print(f"Auto epsilon = {epsilon:.2f}")

    model = PolynomialConservation(state_dim=state_dim, degree=degree, include_trig_dims=include_trig_dims).to(device)

    save_path = os.path.join(save_dir, f"polynomial_{env_name}_best.pt")
    os.makedirs(save_dir, exist_ok=True)

    # For very large datasets, we don't need all trajectories to fit 15-20 weights.
    max_train = min(len(raw_trajectories_np), 200_000)
    train_data = raw_trajectories_np[:max_train].astype(np.float32, copy=False)

    trajs_tensor = torch.tensor(train_data, dtype=torch.float32, device=device)
    if energy0_np is not None:
        e0 = np.asarray(energy0_np[:max_train], dtype=np.float32)
        e0_tensor = torch.tensor(e0, dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(trajs_tensor, e0_tensor), batch_size=batch_size, shuffle=True, drop_last=True)
        e0_scale = float(np.std(e0) + 1e-8)
    else:
        loader = DataLoader(TensorDataset(trajs_tensor), batch_size=batch_size, shuffle=True, drop_last=True)
        e0_scale = 1.0

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if warmup_epochs and epoch < warmup_epochs:
            warm_lr = lr * float(epoch + 1) / float(warmup_epochs)
            for pg in opt.param_groups:
                pg["lr"] = warm_lr

        for batch in loader:
            batch_traj = batch[0]
            batch_e0 = batch[1] if len(batch) > 1 else None
            opt.zero_grad(set_to_none=True)
            loss, _, _ = conservation_loss(model, batch_traj, lambda_var=lambda_var, epsilon=epsilon)

            # Energy alignment term pins the scale/offset so coefficients are in physical units.
            # Without this, any affine transform of a conserved quantity is also conserved.
            if (batch_e0 is not None) and (lambda_energy is not None) and (lambda_energy > 0):
                B, T, D = batch_traj.shape
                f0 = model(batch_traj[:, 0, :].reshape(B, D))
                align = torch.mean((f0 - batch_e0) ** 2) / (e0_scale**2)
                loss = loss + float(lambda_energy) * align

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1

        sched.step()
        avg_loss = epoch_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 200 == 0:
            print(f"[poly {env_name}] epoch {epoch+1}/{epochs} loss={avg_loss:.8f} best={best_loss:.8f}")
            model.print_equation(var_names=var_names, threshold=0.005)

    model.load_state_dict(torch.load(save_path, map_location=device))
    print("\n============================================================")
    print(f"FINAL EQUATION - {env_name.upper()}")
    print("============================================================")
    equation = model.print_equation(var_names=var_names, threshold=0.005)
    return model, equation

