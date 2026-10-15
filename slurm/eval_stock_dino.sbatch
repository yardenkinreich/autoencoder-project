#!/bin/bash
#SBATCH --job-name=eval-stock-dino
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Fresh full standard eval.evaluate for the frozen off-the-shelf DINOv2
# checkpoint. Two reasons: (1) its last STANDARD eval was 2026-08-27, and
# the supervised-probe run added today (a deliberately reduced schema -
# only new_set_final_test, no alignment/geometry sections) took over the
# "latest eval for this checkpoint" slot build_registry() uses, so the
# dashboard was showing stock through the probe's partial view; (2) it's
# the baseline every re-verified run is compared against, so it should be
# current alongside them. The stock loader (load_stock_pretrained_dino)
# was never affected by the load_dino_backbone() bug - this is about
# dashboard currency/visibility, not correctness of the old numbers.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained \
  --config configs/eval_suite.yaml \
  --tag off_the_shelf
echo "EXIT=$?"
