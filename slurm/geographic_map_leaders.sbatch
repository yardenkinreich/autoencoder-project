#!/bin/bash
#SBATCH --job-name=geographic-map-leaders
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out
#SBATCH --array=0-1

# Global geographic cluster maps (eval/geographic_map.py) for the two
# leading models by the CORRECTED metrics (i.e. after the
# load_dino_backbone() fix, which invalidated every prior DINO evaluation
# this project had):
#
#   task 0: ep10   - the proto-loss finetune, best overall
#                    (new_set_final_test: acc 0.745 / macroF1 0.645 / QWK 0.824)
#   task 1: stock  - frozen off-the-shelf DINOv2, the baseline everything
#                    is measured against (acc 0.636 / macroF1 0.573 / QWK 0.747)
#
# Written to {run_dir}/geographic_map/ - the checkpoint's own run directory,
# NOT a specific eval_results/<run_id>/ folder, so re-evaluating either
# checkpoint later doesn't orphan the map (see dashboard/data.py's
# geographic_map_artifacts docstring).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

EP10_RUN="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10"
STOCK_RUN="logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  CKPT="$EP10_RUN/eval/final/teacher_checkpoint.pth"
  ARCH="dino"
  LABEL="ep10 proto-loss finetune"
  OUT="$EP10_RUN/geographic_map"
else
  CKPT="data/raw/pretrained/dinov2_vits14_pretrain.pth"
  ARCH="dino_pretrained"
  LABEL="frozen off-the-shelf DINOv2"
  OUT="$STOCK_RUN/geographic_map"
fi

echo "checkpoint: $CKPT"
echo "out: $OUT"

PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.geographic_map \
  --checkpoint "$CKPT" \
  --autoencoder-model "$ARCH" \
  --label "$LABEL" \
  --out_dir "$OUT" \
  --k 4 10 100
echo "EXIT=$?"
