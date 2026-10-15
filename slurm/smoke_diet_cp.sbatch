#!/bin/bash
#SBATCH --job-name=smoke-diet-cp
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm-%x-%j.out

# Quick smoke test for diet_cp.py - a handful of epochs, just enough to
# confirm the training loop runs end to end (freeze/unfreeze transition,
# datum-index cross-entropy, checkpoint save) before committing to the
# paper's full 150-epoch recipe.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/dino/finetune/_smoke_diet_cp_test"
rm -rf "$OUTPUT_DIR"

PYTHONPATH=$(pwd) pixi run -e default python src/train/diet_cp.py \
  --output_dir "$OUTPUT_DIR" \
  --epochs 6 \
  --freeze_epochs_frac 0.2
echo "EXIT=$?"
