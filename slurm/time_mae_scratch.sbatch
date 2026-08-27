#!/bin/bash
#SBATCH --job-name=time-mae-scratch
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Timing probe only, not a real run - 5 epochs at the FULL --num_samples
# 50000/--batch_size 128 config to measure actual per-epoch wall time on
# this GPU, so the 1000-epoch launch's --time can be set from a real
# number instead of a guess.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/mae/scratch/_timing_probe"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

START=$(date +%s)
PYTHONPATH=$(pwd) python src/train/train.py \
  --autoencoder_model mae \
  --input data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat \
  --model_output "$OUTPUT_DIR/autoencoder.pth" \
  --loss_plot "$OUTPUT_DIR/loss_curve.png" \
  --num_samples 50000 \
  --epochs 5 \
  --batch_size 128 \
  --mask_ratio 0.75 \
  --warmup_epochs 40
END=$(date +%s)
echo "TOTAL_SECONDS=$((END - START))"
echo "EXIT=$?"
