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

python -u scripts/run_all.py --device cuda --data_dir data --save_dir results_noise_0p01 --noise_std 0.01 --skip_pysr --no-run_diffusion --cdn_epochs 50 --structured_epochs 100 --poly_epochs 200
