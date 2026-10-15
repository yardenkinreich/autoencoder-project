#!/bin/bash
#SBATCH --job-name=train-diet-cp-frozen
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Item 3 of the "ways forward" list: does DIET-CP need to be MORE frozen?
# Same corrected augmentation as train_diet_cp_augfix.sbatch, but
# --unfreeze_last_n_blocks 0 - freeze the ENTIRE backbone, train only the
# datum-classification head W. Tests whether the collapse is specifically
# something the last-2-blocks unfreezing introduces, or whether it's
# already present with zero backbone movement at all (in which case the
# "collapse" language is wrong for this variant - a fully frozen backbone
# can't itself collapse, so a bad result here would mean the STOCK
# features plus W alone can't solve datum classification well, a
# different finding entirely).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR="logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_frozen"
if [ -e "$OUTPUT_DIR/run_manifest.json" ]; then
    echo "Refusing to run: $OUTPUT_DIR already has a completed run (run_manifest.json exists)." >&2
    exit 1
fi

PYTHONPATH=$(pwd) pixi run -e default python src/train/diet_cp.py \
  --output_dir "$OUTPUT_DIR" \
  --unfreeze_last_n_blocks 0
echo "EXIT=$?"
