#!/bin/bash
#SBATCH --job-name=physics-discovery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
  2>&1 | tee results/experiment_log.txt

