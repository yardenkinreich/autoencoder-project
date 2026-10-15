#!/bin/bash
#SBATCH --job-name=reverify-all-full-eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Full project-wide re-evaluation (eval.evaluate - confusion matrices,
# latent separation plots, attention maps, everything) for every "final"
# checkpoint of a completed run, now that load_dino_backbone()'s
# block_chunks/init_values bug is fixed. See reverify_all_shortcut.sbatch
# for the full context - this is the dashboard-integrated counterpart,
# writing into eval_history.csv / each checkpoint's own eval_results/ so
# the dashboard reflects correct numbers going forward.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while IFS= read -r ckpt; do
  tag=$(echo "$ckpt" | sed -e 's|logs/dino/||' -e 's|/eval/|/|' -e 's|/teacher_checkpoint.pth||' -e 's|/|__|g')
  tag="postfix__${tag}"
  echo "=== $tag ==="
  PYTHONPATH=src pixi run -e default python -m eval.evaluate \
    --checkpoint "$ckpt" --autoencoder-model dino \
    --config configs/eval_suite.yaml --tag "$tag" || echo "FAILED: $ckpt"
done < /tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/final_dino_checkpoints.txt
echo "EXIT=0"
