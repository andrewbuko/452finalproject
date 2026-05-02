# Discovering Conservation Laws from Trajectories

CPSC 452 final project. Given only state trajectories from three classical systems with closed-form ground-truth energies (2D projectile, simple pendulum, spring-mass), the goal is to recover a scalar invariant that is conserved along trajectories and, where possible, the symbolic expression for it. Implementations live in `src/` and the runner is `scripts/run_all.py`.

## 1. CDN (Conservation Dynamics Network)

A black-box MLP `f_theta: s -> R` trained with a temporal conservation loss that minimizes the variance of `f` along each trajectory, with no architectural physics prior. Because the model is unconstrained, it can in principle learn any conserved quantity (or any function constant on the data manifold), and any conservation behavior is purely an artifact of the loss and the trajectory distribution. An optional energy-alignment term pins the scalar output to the same affine scale as the analytical energy for evaluation; this does not change R^2 or rank-correlation metrics, which are affine-invariant. Defined in `src/models/cdn.py`, trained in `src/training/train_cdn.py`.

## 2. Structured Energy Network

Enforces the inductive bias `H(q, v) = T(v) + V(q)` with two independent MLPs, summed at the output. All three benchmark systems are exactly of this separable Hamiltonian form, so this network represents a deliberately matched function class rather than a general invariant learner; comparisons against CDN should be read with that in mind. Trained with the same temporal conservation loss as CDN, so any difference in performance is attributable to the inductive bias rather than the objective. Defined in `src/models/structured_energy.py`, trained in `src/training/train_structured_energy.py`.

## 3. Polynomial CDN

A linear model over a fixed, hand-designed feature library (polynomial monomials in the state, with `cos(theta)` added for the pendulum), trained end-to-end with the conservation loss plus an energy-alignment term to fix the scale. The architecture is exactly expressive enough to represent the analytical energy for each environment when the basis is chosen correctly, which makes it the cleanest test of whether the conservation surrogate loss can recover the true coefficients on its own. In practice it is sensitive to warm-up schedule and the relative weighting of the variance and alignment terms (`--poly_lambda_energy`, `--poly_lambda_var`, `--poly_warmup_epochs`). Defined in `src/models/polynomial_cdn.py`, trained in `src/training/train_polynomial.py`.

## 4. Diffusion Transition Model

A conditional DDPM over one-step state deltas `Δs = s_{t+1} - s_t`, conditioned on `s_t` and noise level, rolled out autoregressively to generate full trajectories. This is included as a trajectory-prediction baseline rather than an invariant learner: it has no notion of energy in its objective, so any conservation behavior in its rollouts is incidental. The relevant diagnostic is the ratio of generated-trajectory energy variance to ground-truth energy variance, which directly measures how much rollout-level accuracy fails to imply physical conservation. Defined in `src/models/diffusion_transition.py`, trained in `src/training/train_diffusion.py`, evaluated in `src/evaluation/validate_diffusion.py`.

## 5. Symbolic Regression (SINDy and PySR)

Two complementary post-hoc equation-discovery procedures applied to the analytical energies (and, optionally, to the learned invariants). SINDy uses STLSQ over a hand-designed polynomial / trig library defined per environment in `src/evaluation/sindy.py`; when the true expression lies in the library, STLSQ recovers the analytical coefficients to numerical precision, which is a sanity check on the pipeline rather than unsupervised discovery. PySR (`src/evaluation/symbolic_regression.py`) performs genetic-programming search over a configurable operator set without committing to a fixed basis, requires a working Julia install, and can be disabled with `--skip_pysr`. Both are wired into `scripts/run_all.py` and write their recovered expressions to `results/experiment_results.json`.
