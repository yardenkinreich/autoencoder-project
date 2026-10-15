#!/bin/bash
#SBATCH --job-name=train-diet-cp-augfix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Re-run of train_diet_cp.sbatch (job 191, the paper's own 150-epoch
# recipe) under diet_cp.py's corrected build_augmentation(): now matching
# the paper's Appendix B recipe (ColorJitter-equivalent gated behind
# RandomApply(p=0.3) instead of always-applied, stronger blur sigma
# 1.0-2.0), and with the GaussianBlur probability-inversion bug fixed
# (GaussianBlur(p=X) actually blurs with probability (1-X) in this
# environment - see build_augmentation()'s docstring for the measured
# derivation). Explicit --output_dir, not the script's own canonical path,
# since that path (logs/dino/diet-cp/.../ep150) already holds job 191's
# completed run under the old, weaker augmentation - keep both for a direct
# before/after comparison.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_augfix"
if [ -e "$OUTPUT_DIR/run_manifest.json" ]; then
    echo "Refusing to run: $OUTPUT_DIR already has a completed run (run_manifest.json exists)." >&2
    exit 1
fi

PYTHONPATH=$(pwd) pixi run -e default python src/train/diet_cp.py \
  --output_dir "$OUTPUT_DIR"
echo "EXIT=$?"
