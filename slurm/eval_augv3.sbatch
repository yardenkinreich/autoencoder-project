#!/bin/bash
#SBATCH --job-name=eval-augv3
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Full eval.evaluate run for ep5_augv3 (dino_craters_augmentation.py's
# GaussianBlur-probability-inversion fix), same protocol as eval_augfix.sbatch
# (ep5_augv2's eval) so both show up in the dashboard for direct comparison.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_augv3/eval/final/teacher_checkpoint.pth" \
  --autoencoder-model dino \
  --config configs/eval_suite.yaml \
  --tag dino_augv3_noproto_ep5
echo "EXIT=$?"
