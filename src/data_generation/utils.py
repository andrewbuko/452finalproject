import numpy as np


def normalize_trajectories(trajs: np.ndarray):
    """per-feature min-max to [0,1]. returns (normed, {'min','max','range'})."""
    _, _, D = trajs.shape
    flat = trajs.reshape(-1, D)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (trajs - mins) / ranges
    stats = {"min": mins, "max": maxs, "range": ranges}
    return normed, stats


def denormalize_trajectories(normed: np.ndarray, stats):
    return normed * stats["range"] + stats["min"]


def standardize_trajectories(trajs: np.ndarray, eps: float = 1e-8):
    """per-feature standardize to mean 0, std 1. returns (z, {'mean','std'})."""
    _, _, D = trajs.shape
    flat = trajs.reshape(-1, D)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    z = (trajs - mean) / std
    return z, {"mean": mean, "std": std}


def destandardize_trajectories(z: np.ndarray, stats):
    return z * stats["std"] + stats["mean"]


def scale_trajectories(trajs: np.ndarray, mode: str = "minmax01"):
    """mode in {'minmax01', 'standardize'}."""
    mode = mode.lower()
    if mode == "minmax01":
        return normalize_trajectories(trajs)
    if mode == "standardize":
        return standardize_trajectories(trajs)
    raise ValueError(f"unknown scaling mode: {mode}")


def unscale_trajectories(scaled: np.ndarray, stats, mode: str = "minmax01"):
    mode = mode.lower()
    if mode == "minmax01":
        return denormalize_trajectories(scaled, stats)
    if mode == "standardize":
        return destandardize_trajectories(scaled, stats)
    raise ValueError(f"unknown scaling mode: {mode}")


def train_val_split(trajs: np.ndarray, val_fraction: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(trajs)
    indices = rng.permutation(n)
    split = int(n * (1 - val_fraction))
    return trajs[indices[:split]], trajs[indices[split:]]
