"""
Legacy entrypoint kept for convenience.

The main, up-to-date pipeline for equation recovery is:
  python -m scripts.run_equation_discovery
"""

from scripts.run_equation_discovery import run_pendulum, run_projectile  # noqa: F401

if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    run_projectile(device)
    run_pendulum(device)

