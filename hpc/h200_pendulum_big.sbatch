#!/bin/bash
#SBATCH -J pend_big
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

cd /path/to/452finalproject
source venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1

# If your cluster uses modules:
# module load cuda

python -u -m scripts.run_big_hpc_experiment

