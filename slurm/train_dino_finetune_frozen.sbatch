#!/bin/bash
#SBATCH --job-name=dino-finetune-frozen
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Item 3 of the "ways forward" list, self-distillation side: same recipe
# as train_dino_finetune_cropfix.sbatch (all prior fixes in place - blur
# probability, crop scale) but now with --freeze_backbone_except_last_n_
# blocks 1 - only the LAST transformer block (of 12) can move at all.
# Tests whether constraining how much of the backbone can move at all
# prevents the drift into the brightness-dominated basin, independent of
# which objective/augmentation/crop-scale detail is driving it - see
# train_dino.py's new --freeze_backbone_except_last_n_blocks flag.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

OUTPUT_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_frozen1"
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
    --freeze_backbone_except_last_n_blocks 1 \
    --opts proto.enabled=false
echo "EXIT=$?"
