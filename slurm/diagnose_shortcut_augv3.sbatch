#!/bin/bash
#SBATCH --job-name=diagnose-shortcut-augv3
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Phase-0 shortcut diagnostic (PC1-vs-brightness) against the augv3
# checkpoints (dino_craters_augmentation.py's blur-probability-inversion
# fix), same protocol as the original shortcut_diagnostics.jsonl entries
# for ep10_noproto - direct before/after comparison of the fix.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_augv3"
DIAG_OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics_augv3.jsonl
rm -f "$DIAG_OUT"

for ckpt in training_199 training_399 final; do
  for set_name in julie new_set_final_test; do
    PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
      --checkpoint "$RUN_DIR/eval/$ckpt/teacher_checkpoint.pth" \
      --autoencoder-model dino --label "augv3@$ckpt" --set "$set_name" --out "$DIAG_OUT"
  done
done
echo "EXIT=$?"
