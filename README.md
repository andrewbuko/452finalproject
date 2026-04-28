## From trajectories to equations

This repo is a CPSC 452 final project on **discovering conservation laws** from raw trajectories and recovering **interpretable equations**.

Given only state measurements over time (positions/velocities or angle/angular-velocity), we learn a scalar invariant \(f(s)\) that is conserved along trajectories, and then optionally recover a symbolic expression for it.

### Environments (synthetic ground-truth physics)
- **Projectile** (`state=[x,y,vx,vy]`): \(E = 0.5(v_x^2+v_y^2) + g y\)
- **Pendulum** (`state=[theta,omega]`): \(H = 0.5\omega^2 - g\cos(\theta)\) (with \(m=l=1\))
- **Spring-mass** (`state=[x,v]`): \(E = 0.5k x^2 + 0.5 v^2\) (with \(k=10\))

The generators live in:
- `src/data_generation/projectile.py`
- `src/data_generation/pendulum.py`
- `src/data_generation/spring_mass.py`

## What methods are implemented?

There are two “entrypoints” depending on what you want to show:

- **Notebook entrypoint**: `notebooks/full_run.ipynb`  
  This is the “four-model comparison” notebook (linear, symbolic, diffusion, structured \(K+V\)).

- **Script entrypoint**: `scripts/run_all.py`  
  This runs the full experiment pipeline (includes CDN + optional baselines and plots).

### Models (core baselines)
- **CDN (black-box invariant learner)**: `src/models/cdn.py`  
  An MLP \(s_t \mapsto f_\theta(s_t)\) trained with a temporal conservation loss.

- **Linear polynomial invariant (interpretable)**: `src/models/polynomial_cdn.py`  
  A *linear model over a fixed feature library* (polynomial + optional trig), trained with the same conservation loss + energy alignment to pin scale.

- **Structured energy network (physics prior)**: `src/models/structured_energy.py`  
  Enforces an inductive bias \(H(q,v)=T(v)+V(q)\) using two subnetworks.

- **Diffusion transition model (trajectory baseline)**: `src/models/diffusion_transition.py`  
  A conditional DDPM over one-step deltas \(\Delta s = s_{t+1}-s_t\), rolled out autoregressively.

### Symbolic regression (equation discovery)
- **PySR / SymbolicRegression.jl**: `src/evaluation/symbolic_regression.py`  
  This searches for an equation form given operators (not a fixed polynomial basis).
  **PySR requires Julia**; you can disable it with `--skip_pysr`.

- **SINDy-style sparse regression (hand-designed library)**: `src/evaluation/sindy.py`  
  STLSQ sparse regression over a chosen library (this *is* “hand feeding” a candidate library).

## Installation

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### PySR / Julia (optional)

PySR needs a working Julia installation. If you don’t have Julia set up, you can still run everything else and pass `--skip_pysr` in scripts (or the notebook will simply skip symbolic regression).

## Quickstart

### Run the notebook (four-model comparison)

Open and run:
- `notebooks/full_run.ipynb`

Outputs:
- `results/four_model_results.json`
- model checkpoints in `models/`
- figures in `figures/`

### Run the full script pipeline

GPU (recommended):

```bash
python -u scripts/run_all.py --device cuda --data_dir data --save_dir results
```

CPU (smaller dev run):

```bash
python -u scripts/run_all.py --device cpu --n_trajectories 20000 --n_timesteps 100 --cdn_epochs 50 --poly_epochs 400 --skip_pysr
```

## Running on a cluster (SLURM)

The SLURM submission script is:
- `cluster/submit_job.sh`

Submit:

```bash
sbatch cluster/submit_job.sh
```

Watch logs:

```bash
squeue -u $USER
tail -f results/slurm_<JOBID>.log
tail -n 50 results/slurm_<JOBID>.err
```

### Avoid timeouts (diffusion is the expensive part)

Diffusion training cost scales roughly with:

\[
\text{transitions} \approx \min(N,\ \text{diffusion\_max\_train\_trajectories}) \cdot (T-1)
\]

For a “finish under the time limit” cluster run, add flags like:

```bash
--skip_pysr \
--n_trajectories 20000 \
--n_timesteps 100 \
--diffusion_max_train_trajectories 20000 \
--diffusion_epochs 8 \
--diffusion_eval_rollouts 64
```

Important: when `trajectories.npy` already exists, `scripts/run_all.py` will **load and then subsample** to `--n_trajectories` (so you don’t accidentally train on the full on-disk dataset).

## Outputs

### Files and directories
- **Results JSON**: `results/experiment_results.json`
- **Run log**: `results/experiment_log.txt` (plus SLURM `results/slurm_%j.log`)
- **Figures**: `figures/`  
  Includes conservation validation plots and a “hero” comparison figure.
- **Model checkpoints**: `models/`  
  `.pt` and `.npz` files are written here; see `.gitignore` for what’s tracked.

### Common figure outputs
- `figures/cdn_validation_<env>.png`
- `figures/hero_equation_comparison.png`

## Repository layout (where to look)

- **Data generation**: `src/data_generation/`
  - `projectile.py`, `pendulum.py`, `spring_mass.py`
  - `utils.py` (scaling + train/val split)
- **Models**: `src/models/`
  - `cdn.py`, `polynomial_cdn.py`, `structured_energy.py`, `diffusion_transition.py`
- **Training**: `src/training/`
  - `train_cdn.py`, `train_polynomial.py`, `train_structured_energy.py`, `train_diffusion.py`
- **Evaluation**: `src/evaluation/`
  - `validate_cdn.py`, `validate_diffusion.py`, `probe_cdn.py`, `sindy.py`, `symbolic_regression.py`, `hero_figure.py`
- **Main runner**: `scripts/run_all.py`
- **Cluster**: `cluster/submit_job.sh`, `cluster/environment.yml`

## Troubleshooting

### “My SLURM job produced no log file”
If the job is still pending (`ST=PD`), SLURM won’t create `results/slurm_<JOBID>.log` yet. Wait until the job is running (`ST=R`).

### “My job timed out in diffusion”
Lower diffusion workload:
- `--diffusion_max_train_trajectories` (best lever)
- `--diffusion_epochs`
- `--n_timesteps`

### “PySR isn’t working”
Install Julia and ensure `pysr` imports. Otherwise use `--skip_pysr`.

## References

- PySR: `https://github.com/MilesCranmer/PySR`
- SymbolicRegression.jl: `https://github.com/MilesCranmer/SymbolicRegression.jl`
