#!/bin/bash
#SBATCH --job-name=reverify-all-shortcut
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Full project-wide re-verification of the shortcut diagnostic (PC1-vs-
# brightness) after fixing load_dino_backbone()'s block_chunks/init_values
# auto-detection bug (dino_backbone.py) - EVERY "dino"-family checkpoint
# ever produced by this project (finetune, diet-cp, from-scratch, all
# periodic + final checkpoints, going back to 2026-08-10, not just this
# session) was evaluated with ~96% of its transformer blocks left at
# random initialization before this fix. This re-runs the diagnostic on
# every one of them with the fix in place - the first true reading any of
# these checkpoints have ever gotten.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

DIAG_OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics_ALLPOSTFIX.jsonl
rm -f "$DIAG_OUT"

while IFS= read -r ckpt; do
  label=$(echo "$ckpt" | sed -e 's|logs/dino/||' -e 's|/eval/|/|' -e 's|/teacher_checkpoint.pth||' -e 's|/|__|g')
  echo "=== $label ==="
  for set_name in julie new_set_final_test; do
    PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
      --checkpoint "$ckpt" --autoencoder-model dino --label "$label" --set "$set_name" --out "$DIAG_OUT" || echo "FAILED: $ckpt / $set_name"
  done
done < /tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/all_dino_checkpoints.txt
echo "EXIT=0"
