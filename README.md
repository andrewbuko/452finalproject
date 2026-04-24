# Neural Conservation Discovery + Symbolic Regression (Physics Recovery)

Goal: learn a scalar **conserved quantity** from trajectory data using a Conservation Discovery Network (CDN), then recover a compact closed-form physics equation using **symbolic regression** (PySR).

Hero figure (generated after training):
- `figures/equation_comparison.png`

## Project layout

- **Data**: `data/projectile/`, `data/pendulum/`
- **Models**: `src/models/cdn.py` (baseline CDN), `src/models/polynomial_cdn.py` (readable coefficients)
- **Training**: `src/training/train_cdn.py`, `src/training/train_polynomial_cdn.py`
- **Evaluation**: `src/evaluation/validate_cdn.py`, `src/evaluation/probe_cdn.py`, `src/evaluation/symbolic_regression.py`
- **Runners**: `scripts/run_cdn.py`, `scripts/run_probing.py`, `scripts/run_equation_discovery.py`, `scripts/run_all.py`
- **Notebook**: `notebooks/full_run.ipynb` (end-to-end demo; loads existing data by default)

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:
- **PySR requires Julia**. If you only want neural discovery + probing, you can skip PySR.

## Generate or load data

Generate datasets + sanity plots:

```powershell
python -m src.data_generation.projectile
python -m src.data_generation.pendulum
python -m src.data_generation.visualize_data
```

Outputs:
- `data/projectile/trajectories.npy`
- `data/pendulum/trajectories.npy`
- `figures/data_sanity_projectile.png`
- `figures/data_sanity_pendulum.png`

## Train baseline CDN

```powershell
python -m scripts.run_cdn
```

Outputs:
- `models/cdn_projectile_best.pt`, `models/cdn_pendulum_best.pt`
- `figures/cdn_validation_projectile.png`, `figures/cdn_validation_pendulum.png`

## Equation discovery (readable coefficients + PySR)

This is the main “equation read-out” pipeline:

```powershell
python -m scripts.run_equation_discovery
```

Outputs (examples):
- `models/poly_cdn_projectile_best.pt`, `models/poly_cdn_pendulum_best.pt`
- `figures/pysr_pareto_projectile_energy.png`, `figures/pysr_pareto_pendulum_energy.png`
- `figures/pysr_pareto_projectile_cdn.png`, `figures/pysr_pareto_pendulum_cdn.png`
- `figures/equation_validation_projectile.png`, `figures/equation_validation_pendulum.png`

If PySR/Julia is finicky on your machine or cluster, run the PySR-only script (no torch import):

```powershell
python -m scripts.run_pysr_energy_only
```

## Probing (1D sweeps)

This probes the learned invariant by sweeping one input dimension at a time.

```powershell
python -m scripts.run_probing
```

Outputs:
- `figures/probe_projectile.png`, `figures/probe_pendulum.png`

## End-to-end

Runs the whole pipeline with skip-if-exists logic:

```powershell
python -m scripts.run_all
```

## Running on a server/cluster (git push here → pull there)

- **On your laptop (this repo)**:

```powershell
git status
git add .
git commit -m "update equation discovery pipeline"
git push
```

- **On the cluster login node**:

```bash
git clone <your-repo-url>
cd 452finalproject
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- **Run jobs**:
  - **CPU equation discovery** (fast, no Julia required):

```bash
python -m scripts.run_equation_discovery
```

  - **PySR (Julia)**:
    - Make sure Julia is available (module/juliaup). Then run:

```bash
python -m scripts.run_pysr_energy_only
```

- **SLURM example** (`run_job.sh`):

```bash
#!/bin/bash
#SBATCH -J eqdisc
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00

set -euo pipefail
cd /path/to/452finalproject
source venv/bin/activate
python -m scripts.run_equation_discovery
```

## References

- PySR: `https://github.com/MilesCranmer/PySR`
- SymbolicRegression.jl: `https://github.com/MilesCranmer/SymbolicRegression.jl`
