#!/bin/bash
#SBATCH --job-name=dino-finetune-augfix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same finetune as train_dino_finetune.sbatch (same architecture/LR/schedule,
# same stock-checkpoint warm start) but with two changes, testing the
# brightness-collapse fix in isolation, WITHOUT the prototypical loss (see
# docs/dino_mae_proto_finetune_status.md - collapse needs fixing before
# proto loss is worth re-testing on top of it):
#
#   1. dino_craters_augmentation.py's v2 recipe (solarize on global crop 2,
#      stock-DINOv2-style asymmetric per-crop-slot blur probabilities,
#      instead of the old flat/symmetric one) - this is a CODE change, not a
#      config flag, so every run from now on uses it; run_manifest.json's
#      new "augmentation_recipe" field is the durable marker distinguishing
#      this run from earlier ones that predate the fix.
#   2. --opts proto.enabled=false - the finetune config still has a proto:
#      section (for later re-use), overridden off here.
#
# --epochs 5 (500 iterations, half the earlier 1000) rather than the full
# budget: Phase-0 diagnostics showed the earlier (pre-fix) runs were ALREADY
# fully collapsed by iteration ~199, the very first periodic checkpoint - so
# a shorter run is enough to see whether the fix actually prevents that
# same fast collapse, without paying for the full 1000-iteration cost before
# knowing if it worked. eval_period_iterations=200 in the config still
# gives periodic checkpoints at ~199/399/499(final) to check the trend.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID   # see train_dino_finetune.sbatch's comment

OUTPUT_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_augv2"
if [ -e "$OUTPUT_DIR/run_manifest.json" ]; then
    echo "Refusing to run: $OUTPUT_DIR already has a completed run (run_manifest.json exists)." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=1 \
    src/train/train_dino.py \
    --input data/processed_wac_100m_new/sigma/100/dino_wide/craters_wide.dat \
    --output_dir "$OUTPUT_DIR" \
    --config_file configs/dino_craters_finetune.yaml \
    --pretrained_weights data/raw/pretrained/dinov2_vits14_pretrain.pth \
    --epochs 5 \
    --min_diameter 3.0 --max_diameter 10.0 --data_tag wac100m_sigma100_noband_wide \
    --opts proto.enabled=false
echo "EXIT=$?"
