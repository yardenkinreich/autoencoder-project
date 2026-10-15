#!/bin/bash
#SBATCH --job-name=dino-finetune-cropfix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same finetune as train_dino_finetune_augv3.sbatch (blur-probability fix
# already in place) but with configs/dino_craters_finetune.yaml's crops:
# section now matching stock DINOv2's own local/global scale contrast
# exactly (global [0.32,1.0], local [0.05,0.32] - see that config's own
# comment for the full derivation). The augv3 run showed the blur fix
# alone changed nothing (PC1 vs brightness still r~1.00) - this tests a
# DIFFERENT, independent hypothesis: our previous local_crops_scale
# ([0.25,0.6]) and global_crops_scale ([0.6,1.0]) gave local and global
# crops nearly the same field of view (both always contained ~the whole
# crater), so the self-distillation task never required anything beyond a
# coarse whole-image statistic to solve. New output dir (ep5_cropfix), not
# overwriting ep5_augv2/ep5_augv3 - config.yaml gets saved alongside each
# checkpoint (standard DINOv2 behavior) so the actual crop scale used by
# each run stays traceable even without a dedicated manifest field.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

OUTPUT_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_cropfix"
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
