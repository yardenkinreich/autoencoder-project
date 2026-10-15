#!/bin/bash
#SBATCH --job-name=per-band-k4
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# k=4 for morph_sigma100. Embeddings are already cached from the k=10 run, so
# this only re-clusters and redraws - the expensive pass is skipped.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.per_band_clustering \
  --map-dir "logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/geomap/morph_sigma100__global_wide" --k 4
echo "EXIT=$?"
