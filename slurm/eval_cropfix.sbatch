#!/bin/bash
#SBATCH --job-name=eval-cropfix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Full eval.evaluate run for ep5_cropfix, same protocol as eval_augv3.sbatch
# so it shows up in the dashboard for direct comparison.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_cropfix/eval/final/teacher_checkpoint.pth" \
  --autoencoder-model dino \
  --config configs/eval_suite.yaml \
  --tag dino_cropfix_noproto_ep5
echo "EXIT=$?"
