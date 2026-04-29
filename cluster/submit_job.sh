#!/bin/bash
#SBATCH --job-name=physics-discovery
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm_%j.log
#SBATCH --error=results/slurm_%j.err

set -euo pipefail

mkdir -p results figures models

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd "$SLURM_SUBMIT_DIR"

echo "Starting physics discovery experiment"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"

python -u scripts/run_all.py \
  --device cuda \
  --data_dir data \
  --save_dir results \
  --n_trajectories 20000 \
  --n_timesteps 100 \
  --cdn_epochs 256 \
  --epochs_rest 256 \
  --poly_lambda_energy 1.0 \
  --poly_lambda_var 1.0 \
  --poly_warmup_epochs 100 \
  --skip_pysr \
  --diffusion_max_train_trajectories 20000 \
  --diffusion_eval_rollouts 64 \
  --noise_std 0.0 \
  2>&1 | tee results/experiment_log.txt
