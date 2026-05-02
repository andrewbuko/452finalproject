# Discovering Conservation Laws from Trajectories

This repository contains a CPSC 452 final project that explores multiple methods for discovering conservation laws and recovering interpretable symbolic equations directly from raw trajectory data. The goal is to provide a comprehensive comparison of black-box, physics-informed, and symbolic approaches for learning conserved quantities of dynamical systems. All methods are evaluated on three synthetic environments with known ground-truth energies: a 2D **projectile**, a **pendulum**, and a **spring-mass** system.

## 1. CDN (Conservation Dynamics Network)

This approach is a black-box invariant learner. A multilayer perceptron `f(s)` is trained directly on state trajectories with a temporal conservation loss that penalizes changes of `f` along each trajectory. No physics prior is built into the architecture, so any conservation behavior must emerge from the data and the loss.

**Key Features:**

- **Architecture-agnostic invariant**: A standard MLP maps a state vector to a single scalar conserved quantity.
- **Conservation Loss**: Penalizes the variance of `f(s_t)` along each trajectory, encouraging invariance with respect to time.
- **Energy Alignment**: An optional alignment term pins the scalar output to the same scale as a reference energy for evaluation.

## 2. Structured Energy Network

This approach builds in a strong physics prior by enforcing the inductive bias `H(q, v) = T(v) + V(q)`, where the kinetic and potential terms are represented by separate subnetworks. Because all three benchmark systems are of this Hamiltonian form, this model serves as a "best case" reference for what a correctly specified architecture can achieve.

**Key Features:**

- **Separable Architecture**: Independent kinetic `T(v)` and potential `V(q)` subnetworks combined as `T + V`.
- **Physics-Informed Bias**: Restricts the function class to separable Hamiltonians, making the learned invariant directly interpretable as energy.
- **Same Conservation Objective**: Trained with the same temporal conservation loss as the CDN for a fair comparison.

## 3. Polynomial CDN

This approach replaces the MLP with a linear model over a fixed feature library (polynomial terms, with optional trigonometric functions for the pendulum). Because the parameters are coefficients on named monomials, the learned invariant can be read off as a closed-form expression.

**Key Features:**

- **Interpretable Linear Model**: Weights map directly to coefficients of a polynomial / trig expression.
- **Hand-Designed Library**: Feature library is specified per environment (e.g., `x^2, v^2` for spring-mass; `cos(theta), omega^2` for pendulum).
- **Variance + Energy-Alignment Loss**: Combines the conservation loss with an explicit alignment term to recover physical scale.

## 4. Diffusion Transition Model

This approach is a trajectory-prediction baseline rather than an invariant learner. A conditional DDPM is trained over one-step state deltas `Δs = s_{t+1} − s_t` and then rolled out autoregressively. It is included to test whether good trajectory-level prediction implies physical conservation.

**Key Features:**

- **One-Step Delta Modeling**: Learns the distribution of state increments conditioned on the current state.
- **Autoregressive Rollouts**: Generates full trajectories by repeatedly sampling next-step deltas.
- **Conservation Diagnostic**: Energy variance of generated trajectories is compared to ground-truth energy variance, exposing the gap between rollout accuracy and physical conservation.

## 5. Symbolic Regression (SINDy and PySR)

This component performs explicit equation discovery on top of the learned invariants and the analytical energies. Two complementary methods are used.

**Key Features:**

- **SINDy-Style Sparse Regression (STLSQ)**: Sequential thresholded least squares over a hand-designed polynomial / trig library, producing sparse, human-readable expressions.
- **PySR / SymbolicRegression.jl**: Genetic-programming search over a configurable operator set; does not require pre-specifying a fixed basis. Requires a working Julia install and can be disabled with `--skip_pysr`.
- **Coefficient Recovery**: When the true expression lies inside the candidate library, both methods recover the exact analytical coefficients, providing a sanity check on the symbolic pipeline.

## Getting Started

### Prerequisites

- Python 3.x
- torch
- numpy
- scipy
- matplotlib
- tqdm
- pysr (optional, requires Julia)
- sympy
- pandas

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

If you do not have Julia installed, you can skip the PySR step and still run every other method by passing `--skip_pysr` to the runner script.

### Running the Project

The repository provides two entrypoints depending on how you want to run the experiments.

**Notebook entrypoint** — open and run the four-model comparison notebook in any Jupyter-compatible environment:

```
notebooks/full_run.ipynb
```

**Script entrypoint** — run the full pipeline (data generation, training of all models, evaluation, symbolic regression, and figure generation):

```bash
python -u scripts/run_all.py --device cuda --data_dir data --save_dir results
```

For a smaller CPU run useful for development:

```bash
python -u scripts/run_all.py --device cpu --n_trajectories 20000 --n_timesteps 100 --cdn_epochs 50 --poly_epochs 400 --skip_pysr
```

To run on a SLURM cluster, submit the provided batch script:

```bash
sbatch cluster/submit_job.sh
```

Outputs are written to `results/` (metrics JSON and run log), `figures/` (validation and comparison plots), and `models/` (checkpoints).
