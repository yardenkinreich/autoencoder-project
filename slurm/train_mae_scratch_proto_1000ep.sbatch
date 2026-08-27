#!/bin/bash
#SBATCH --job-name=mae-scratch-proto-1000ep
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same as train_mae_scratch_1000ep.sbatch (identical architecture/data/LR
# schedule) but with the prototypical-loss auxiliary term enabled
# (src/train/prototypical_loss.py) - a controlled A/B, same as the DINO
# proto-vs-noproto comparison. proto_loss_weight left at the CLI default
# (0.05, not DINO's 0.2) - see train.py's --proto_loss_weight help text: the
# same 0.2 DINO uses gave proto_loss ~7-12% of MAE's total loss in a smoke
# test (MAE's reconstruction loss sits on a much smaller scale than DINO's
# combined total), so 0.05 was chosen to land in the same ~1-3%
# gentle-nudge range DINO's weight gave there.
#
# OUTPUT_DIR deliberately differs from the plain run's (adds "_proto") -
# canonical_run_dir() (src/run_layout.py) has no axis for the proto flag, so
# with identical hyperparameters otherwise this would land at the exact same
# directory and overwrite the plain run's checkpoint (same issue hit with
# DINO's proto/noproto runs - see train_dino_finetune_noproto.sbatch).
#
# Resumable: re-submitting this script after a timeout/interruption picks up
# from checkpoint_last.pth automatically (checkpoint_every=20, the default).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/mae/scratch/none/na/wac100m_sigma100_noband/d3-10/n50000/mask0.75/ep1000_proto"
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
  --proto_enabled \
  "${RESUME_ARGS[@]}"
echo "EXIT=$?"
