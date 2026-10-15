#!/bin/bash
#SBATCH --job-name=diagnose-eval-diet-cp
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same Phase-0 shortcut diagnostic (PC1-vs-brightness) as the DINO
# self-distillation checkpoints, plus a full eval.evaluate run, against the
# DIET-CP checkpoint (job 188). DIET-CP never trained on new_test_set at
# all (see diet_cp.py's module docstring), so it's safe to evaluate against
# the FULL new_set here, not just the leak-safe new_set_final_test split -
# no --tag restriction needed the way proto-loss checkpoints require.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CKPT="logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150/eval/final/teacher_checkpoint.pth"
DIAG_OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics_dietcp.jsonl
rm -f "$DIAG_OUT"

echo "=== shortcut diagnostic ==="
for set_name in julie new_set_final_test; do
  PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
    --checkpoint "$CKPT" --autoencoder-model dino --label "diet_cp" --set "$set_name" --out "$DIAG_OUT"
done

echo "=== full eval ==="
PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint "$CKPT" --autoencoder-model dino \
  --config configs/eval_suite.yaml --tag diet_cp_ep150_n1000
echo "EXIT=$?"
