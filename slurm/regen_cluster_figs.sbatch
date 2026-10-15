#!/bin/bash
#SBATCH --job-name=regen-clust-figs
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Regenerate the per-cluster figures in the NEW layout: one row per cluster,
# the average crater followed by real members.
#
# The old grid showed averages only, and that hid the thing that matters. On
# morph_sigma100 k=10 every cluster's average looks like the same dark bowl
# with a bright right crescent, differing only in crispness - yet the members
# reveal that the high-sobel clusters are dominated by rugged surrounding
# TERRAIN rather than sharp craters. Averaging hundreds of randomly-oriented
# rough backgrounds washes them to smooth grey, so no grid of means could ever
# have shown it. Members beside the average is the fix.
#
# CPU-only: this reads the crop memmaps and draws, no model involved.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
export OMP_NUM_THREADS=4

S=logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/geomap
F=logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10/data_comparison/geomap

for d in "$S/emp_sigma100__global_wide" \
         "$S/morph_sigma100_div500__global_wide" \
         "$F/morph_sigma100__global_wide" \
         "$F/emp_sigma100__global_wide"; do
  echo "=========== $d"
  PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.cluster_profile \
    --dir "$d" --overwrite
done
echo "EXIT=$?"
