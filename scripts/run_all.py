import os
import subprocess
import sys


def _run(cmd):
    print("\n>", " ".join(cmd))
    subprocess.check_call(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def _exists(path):
    return os.path.exists(path)


if __name__ == "__main__":
    py = sys.executable

    # 1) Ensure base CDN models exist
    if not (_exists("models/cdn_projectile_best.pt") and _exists("models/cdn_pendulum_best.pt")):
        _run([py, "-u", "-m", "scripts.run_cdn"])
    else:
        print("Skipping run_cdn.py (checkpoints exist).")

    # 2) Probing figures
    if not (_exists("figures/probe_projectile.png") and _exists("figures/probe_pendulum.png")):
        _run([py, "-u", "-m", "scripts.run_probing"])
    else:
        print("Skipping run_probing.py (probe figures exist).")

    # 3) Hero figure
    if not _exists("figures/equation_comparison.png"):
        _run([py, "-u", "-m", "scripts.make_hero_figure"])
    else:
        print("Skipping make_hero_figure.py (hero figure exists).")

    # 4) Equation discovery (polynomial + PySR) (optional if Julia missing)
    if not _exists("figures/pysr_pareto_projectile_energy.png"):
        print("\nEquation-discovery step not detected; attempting polynomial + PySR.")
        _run([py, "-u", "-m", "scripts.run_equation_discovery"])
    else:
        print("Skipping run_equation_discovery.py (pareto figures exist).")

    print("\nAll done.")

