#!/bin/bash
#SBATCH --job-name=reverify-split-full-eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out
#SBATCH --array=0-4

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRATCH=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad
LIST_FILE=$(printf "%s/eval_split_%02d.txt" "$SCRATCH" "$SLURM_ARRAY_TASK_ID")
HIST_FILE=$(printf "%s/eval_history_split_%02d.csv" "$SCRATCH" "$SLURM_ARRAY_TASK_ID")

while IFS= read -r ckpt; do
  tag=$(echo "$ckpt" | sed -e 's|logs/dino/||' -e 's|/eval/|/|' -e 's|/teacher_checkpoint.pth||' -e 's|/|__|g')
  tag="postfix__${tag}"
  echo "=== $tag ==="
  PYTHONPATH=src pixi run -e default python -m eval.evaluate \
    --checkpoint "$ckpt" --autoencoder-model dino \
    --config configs/eval_suite.yaml --tag "$tag" --history "$HIST_FILE" || echo "FAILED: $ckpt"
done < "$LIST_FILE"
echo "EXIT=0"
