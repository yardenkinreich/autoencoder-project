#!/bin/bash
#SBATCH --job-name=attention-maps
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm-%x-%j.out

# Attention-map diagnostic (see src/eval/attention_maps.py's module docstring)
# for stock + the two existing (collapsed, pre-augmentation-fix) DINO
# checkpoints - a baseline to compare against once the augmentation-fixed
# retrain is ready.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/attention_maps

run_diag() {
  PYTHONPATH=src pixi run -e default python -m eval.attention_maps \
    --checkpoint "$1" --autoencoder-model "$2" --label "$3" --out-dir "$OUT/$3"
}

run_diag "data/raw/pretrained/dinov2_vits14_pretrain.pth" dino_pretrained "stock"
run_diag "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10_noproto/eval/final/teacher_checkpoint.pth" dino "noproto_final_collapsed"
run_diag "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10/eval/final/teacher_checkpoint.pth" dino "proto_final_collapsed"

echo "ALL_DONE"
