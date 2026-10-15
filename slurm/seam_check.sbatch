#!/bin/bash
#SBATCH --job-name=seam-check
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out
#SBATCH --array=0-1

# Is there a constant-longitude (vertical) discontinuity in the imagery or
# the clustering, at the seam boundaries of the retained band? See
# eval/seam_check.py's module docstring. CPU-only: this reads pixel stats
# off the memmap and re-uses the cluster assignments geographic_map.py
# already wrote, so there is no embedding pass here.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=4

EP10_RUN="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10"
STOCK_RUN="logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  RUN="$EP10_RUN"; LABEL="ep10 proto-loss finetune"
else
  RUN="$STOCK_RUN"; LABEL="frozen off-the-shelf DINOv2"
fi

PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.seam_check \
  --clusters_csv "$RUN/geographic_map/geographic_clusters_k4.csv" \
  --label "$LABEL" \
  --out_dir "$RUN/geographic_map"
echo "EXIT=$?"
