#!/bin/bash
#SBATCH --job-name=diagnose-shortcut-augfix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same Phase-0 diagnostic as diagnose_shortcut.sbatch, run against the new
# augmentation-fixed checkpoint (job 181, ep5_augv2) and its own periodic
# checkpoints, to check whether the fix actually prevented the brightness
# collapse the earlier (pre-fix) runs showed by iteration ~199.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics_augfix.jsonl
rm -f "$OUT"

BASE="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_augv2/eval"

run_diag() {
  PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
    --checkpoint "$1" --autoencoder-model dino --label "$2" --set "$3" --out "$OUT"
}

for set_name in julie new_set_final_test; do
  for it in 199 399; do
    run_diag "$BASE/training_$it/teacher_checkpoint.pth" "augfix@$it" "$set_name"
  done
  run_diag "$BASE/final/teacher_checkpoint.pth" "augfix@final" "$set_name"
done

echo "ALL_DONE"
