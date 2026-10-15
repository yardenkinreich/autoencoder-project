#!/bin/bash
#SBATCH --job-name=diagnose-shortcut
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --output=logs/slurm-%x-%j.out

# Phase-0 diagnostics (docs/dino_mae_proto_finetune_status.md) for the
# brightness-shortcut hypothesis: PC1-vs-brightness correlation + PC1-
# removed residual metrics, for the stock checkpoint and every periodic
# checkpoint of both finetune runs (199/399/599/799/999/final), on both
# julie and new_set_final_test. No training - pure re-embedding + analysis.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics.jsonl
rm -f "$OUT"

NOPROTO_BASE="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10_noproto/eval"
PROTO_BASE="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10/eval"

run_diag() {
  PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
    --checkpoint "$1" --autoencoder-model "$2" --label "$3" --set "$4" --out "$OUT"
}

for set_name in julie new_set_final_test; do
  run_diag "data/raw/pretrained/dinov2_vits14_pretrain.pth" dino_pretrained "stock" "$set_name"
  for it in 199 399 599 799 999; do
    run_diag "$NOPROTO_BASE/training_$it/teacher_checkpoint.pth" dino "noproto@$it" "$set_name"
  done
  run_diag "$NOPROTO_BASE/final/teacher_checkpoint.pth" dino "noproto@final" "$set_name"
  for it in 199 399 599 799 999; do
    run_diag "$PROTO_BASE/training_$it/teacher_checkpoint.pth" dino "proto@$it" "$set_name"
  done
  run_diag "$PROTO_BASE/final/teacher_checkpoint.pth" dino "proto@final" "$set_name"
done

echo "ALL_DONE"
