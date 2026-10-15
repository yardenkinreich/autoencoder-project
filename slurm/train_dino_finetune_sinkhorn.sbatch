#!/bin/bash
#SBATCH --job-name=dino-finetune-sinkhorn
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same recipe as train_dino_finetune_cropfix.sbatch (all prior fixes in
# place) but train.centering=sinkhorn_knopp instead of the stock default
# "centering". Sinkhorn-Knopp enforces a balanced/uniform cluster
# assignment across the WHOLE BATCH (the same equipartition mechanism
# SwAV/Suave use - see dino_finetuning_semi_supervised.pdf Eq. 5-6), a
# meaningfully stronger anti-collapse mechanism than simple centering +
# sharpening. Never overridden from stock in dino_craters_finetune.yaml
# before this run.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

OUTPUT_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_sinkhorn"
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
    --opts proto.enabled=false train.centering=sinkhorn_knopp
echo "EXIT=$?"
