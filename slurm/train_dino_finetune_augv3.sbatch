#!/bin/bash
#SBATCH --job-name=dino-finetune-augv3
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Re-run of train_dino_finetune_augfix.sbatch (job that produced ep5_augv2)
# under dino_craters_augmentation.py's CORRECTED v3 recipe. The v2 run's
# GaussianBlur(p=...) calls were silently inverted (see AUGMENTATION_
# RECIPE_VERSION's v3 comment in dino_craters_augmentation.py, and diet_cp.py's
# build_augmentation() docstring for the full measured derivation):
# global_transfo1 (documented "always blurred") actually never blurred;
# global_transfo2 (documented "rarely blurred") actually blurred ~90% of the
# time. ep5_augv2's "didn't fix the collapse" result was measured against
# that wrong recipe. New output dir (ep5_augv3), not overwriting ep5_augv2 -
# keep both for a direct before/after comparison of the blur-probability fix
# specifically.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

OUTPUT_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_augv3"
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
