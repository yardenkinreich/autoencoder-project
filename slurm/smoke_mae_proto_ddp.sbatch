#!/bin/bash
#SBATCH --job-name=smoke-mae-proto-ddp
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm-%x-%j.out

# 2-GPU DDP smoke test for --proto_enabled - the single-GPU smoke tests
# (job 142/143) never exercised DDP's gradient-sync path for the proto
# term (embed_for_proto calls model.module.forward_encoder(), bypassing
# DDP's own wrapped forward() - should still sync correctly since DDP hooks
# are on the parameters themselves, not the forward() call path, but this
# is the first actual multi-GPU check of that).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/mae/finetune/_smoke_proto_ddp_test"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=2 src/train/train.py \
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
