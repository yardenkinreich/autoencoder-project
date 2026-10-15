#!/bin/bash
#SBATCH --job-name=diagnose-shortcut-dietcp-augfix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Phase-0 shortcut diagnostic (PC1-vs-brightness) against the DIET-CP
# augfix checkpoint (diet_cp.py's corrected build_augmentation() - paper's
# Appendix B gating/magnitude + the GaussianBlur-probability-inversion
# fix), same protocol as shortcut_diagnostics_dietcp.jsonl's original
# entries (job 191) - direct before/after comparison of the fix.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CKPT="logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_augfix/eval/final/teacher_checkpoint.pth"
DIAG_OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics_dietcp_augfix.jsonl
rm -f "$DIAG_OUT"

for set_name in julie new_set_final_test; do
  PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
    --checkpoint "$CKPT" --autoencoder-model dino --label "diet_cp_augfix" --set "$set_name" --out "$DIAG_OUT"
done
echo "EXIT=$?"
