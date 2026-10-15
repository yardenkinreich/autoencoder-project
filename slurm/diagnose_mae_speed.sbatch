#!/bin/bash
#SBATCH --job-name=diagnose-mae-speed
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Diagnostic only: measures actual GPU utilization during training (not just
# wall-clock time) at two batch sizes, to check whether train.py's ~30s/epoch
# at batch_size=128 reflects real H200 saturation or the GPU sitting mostly
# idle between small kernel launches on a modest ViT-Base/128px model.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

run_probe() {
  local bs=$1
  local outdir="logs/mae/scratch/_speed_probe_bs${bs}"
  rm -rf "$outdir"; mkdir -p "$outdir"
  echo "=== batch_size=$bs ==="
  PYTHONPATH=$(pwd) python src/train/train.py \
    --autoencoder_model mae \
    --input data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat \
    --model_output "$outdir/autoencoder.pth" \
    --loss_plot "$outdir/loss_curve.png" \
    --num_samples 50000 \
    --epochs 3 \
    --batch_size "$bs" \
    --mask_ratio 0.75 \
    --warmup_epochs 40 &
  TRAIN_PID=$!
  sleep 8   # let it get past data loading / into steady-state training
  echo "--- nvidia-smi samples (batch_size=$bs) ---"
  for i in $(seq 1 10); do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
    sleep 2
  done
  wait $TRAIN_PID
  rm -rf "$outdir"
}

run_probe 128
run_probe 512
run_probe 1024
echo "ALL_DONE"
