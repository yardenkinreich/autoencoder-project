#!/bin/bash
#SBATCH --job-name=mae-scratch-1000ep
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Plain (no proto loss) MAE from-scratch baseline at a much longer budget
# (1000 epochs) than any prior scratch run (previous longest: ep500) - the
# "as is" comparison point for train_mae_scratch_proto_1000ep.sbatch.
#
# --warmup_epochs 40 (NOT scaled to 10% of 1000 = 100) - the official MAE
# recipe's own absolute value (src/models/mae/util/lr_sched.py, the 400-epoch
# ImageNet schedule) - warmup's job is letting the optimizer's running
# stats stabilize, roughly a function of step count, not a fixed fraction of
# however long the rest of training happens to run. The "~10% of --epochs"
# guidance in train.py's own --warmup_epochs help text is for SHRINKING
# warmup on a shorter-than-official schedule (so it doesn't eat the whole
# budget), not a scaling law to extend proportionally on a longer one.
#
# Resumable: re-submitting this script after a timeout/interruption picks up
# from checkpoint_last.pth automatically (checkpoint_every=20, the default).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/mae/scratch/none/na/wac100m_sigma100_noband/d3-10/n50000/mask0.75/ep1000"
mkdir -p "$OUTPUT_DIR/models"

RESUME_ARGS=()
if [ -f "$OUTPUT_DIR/models/checkpoint_last.pth" ]; then
    echo "Resuming from $OUTPUT_DIR/models/checkpoint_last.pth"
    RESUME_ARGS=(--resume "$OUTPUT_DIR/models/checkpoint_last.pth")
fi

PYTHONPATH=$(pwd) python src/train/train.py \
  --autoencoder_model mae \
  --input data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat \
  --model_output "$OUTPUT_DIR/models/autoencoder.pth" \
  --loss_plot "$OUTPUT_DIR/models/loss_curve.png" \
  --run_dir "$OUTPUT_DIR" \
  --num_samples 50000 \
  --epochs 1000 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --min_lr 1e-8 \
  --val_split 0.2 \
  --mask_ratio 0.75 \
  --warmup_epochs 40 \
  --min_diameter 3.0 --max_diameter 10.0 \
  --data_tag wac100m_sigma100_noband \
  "${RESUME_ARGS[@]}"
echo "EXIT=$?"
