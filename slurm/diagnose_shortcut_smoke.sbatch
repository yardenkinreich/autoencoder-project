#!/bin/bash
#SBATCH --job-name=diagnose-shortcut-smoke
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm-%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
  --checkpoint "data/raw/pretrained/dinov2_vits14_pretrain.pth" \
  --autoencoder-model dino_pretrained --label "stock" --set julie
echo "EXIT=$?"
