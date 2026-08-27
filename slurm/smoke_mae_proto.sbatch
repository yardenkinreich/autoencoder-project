#!/bin/bash
#SBATCH --job-name=smoke-mae-proto
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm-%x-%j.out

# Quick smoke test for the new --proto_enabled option in train.py (MAE) -
# NOT a real training run: tiny --num_samples/--epochs, just enough to see
# real train_loss vs proto_train_loss numbers and confirm nothing crashes
# (DDP/GradScaler interaction, embed_for_proto's forward_encoder call,
# LabeledEpisodePool loading) before committing to a full run.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/mae/finetune/_smoke_proto_test"
mkdir -p "$OUTPUT_DIR"

# --pretrained_weights dropped for this smoke test - src/models/mae/
# pretrain_mae_vit_base_full.pth is patch16, but train.py's current MAE
# definition hardcodes patch8 (a pre-existing incompatibility, unrelated to
# proto_enabled - the historical finetune run that used this checkpoint
# must predate that hardcoding). From-scratch here just to validate the
# proto mechanism itself and see real loss magnitudes.
PYTHONPATH=$(pwd) python src/train/train.py \
  --autoencoder_model mae \
  --input data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat \
  --model_output "$OUTPUT_DIR/autoencoder.pth" \
  --loss_plot "$OUTPUT_DIR/loss_curve.png" \
  --num_samples 512 \
  --epochs 2 \
  --batch_size 32 \
  --mask_ratio 0.75 \
  --proto_enabled \
  --proto_loss_weight 0.05
echo "EXIT=$?"
