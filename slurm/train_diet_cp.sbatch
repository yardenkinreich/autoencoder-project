#!/bin/bash
#SBATCH --job-name=train-diet-cp
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Full DIET-CP recipe (src/train/diet_cp.py) - the paper's own
# hyperparameters, used as-is (150 epochs, lr=1e-4, weight_decay=0.05,
# label_smoothing=0.3, batch_size=32). No --output_dir passed - the script
# computes its own canonical path (logs/dino/diet-cp/...) and refuses to
# run if that path already has a completed run.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=$(pwd) pixi run -e default python src/train/diet_cp.py
echo "EXIT=$?"
