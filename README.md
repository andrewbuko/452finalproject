## From Trajectories to Equations: Machine Discovery of Conservation Laws via Neural Networks and Symbolic Regression

Given only raw position and velocity measurements of moving objects - with no physics knowledge, no labels, and no equations - this project discovers conserved quantities and recovers readable physics equations:

- **PROJECTILE**: `f(s) = 0.50000*vx^2 + 0.50000*vy^2 + 9.81000*y`
- **PENDULUM**: `f(s) = 0.50000*omega^2 - 9.81000*cos(theta)`
- **SPRING-MASS**: `f(s) = 5.00000*x^2 + 0.50000*v^2`

The entire experiment runs with one command:

- `python scripts/run_all.py --device cuda --data_dir data --save_dir results`

## Project layout

- **Data**: `data/projectile/`, `data/pendulum/`, `data/spring_mass/`
- **Models**: `src/models/cdn.py` (black-box), `src/models/polynomial_cdn.py` (interpretable), `src/models/structured_energy.py` (T+V baseline)
- **Training**: `src/training/train_cdn.py`, `src/training/train_polynomial.py`, `src/training/train_structured_energy.py`
- **Evaluation**: `src/evaluation/validate_cdn.py`, `src/evaluation/probe_cdn.py`, `src/evaluation/sindy.py`, `src/evaluation/symbolic_regression.py`, `src/evaluation/hero_figure.py`
- **Runner**: `scripts/run_all.py`
- **Notebook**: `notebooks/full_run.ipynb` (end-to-end demo; loads existing data by default)

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:
- **PySR requires Julia**. If you only want the neural + polynomial discovery, pass `--skip_pysr`.

## Run the full pipeline

Local (fast, CPU):

```powershell
python scripts/run_all.py --device cpu --n_trajectories 20000 --cdn_epochs 20 --poly_epochs 200 --poly_batch_size 256 --skip_pysr
```

Cluster (GPU):

```bash
python -u scripts/run_all.py --device cuda --data_dir data --save_dir results
```

Optional alternatives (enabled by default; disable if desired):

- `--no-run_structured_energy`
- `--no-run_sindy`

## Outputs

- **Results**: `results/experiment_results.json`, `results/experiment_log.txt`
- **Hero figure**: `figures/hero_equation_comparison.png`
- **Checkpoints**: `models/`

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
