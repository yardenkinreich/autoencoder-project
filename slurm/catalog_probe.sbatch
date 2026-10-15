#!/bin/bash
#SBATCH --job-name=catalog-probe
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Score all 6 catalogs x 5 normalizations with one frozen DINOv2, ranked by
# macro recall. See src/eval/catalog_probe.py - in particular that this is a
# RANKING instrument: it reliably detects a presentation that destroys signal,
# and only weakly indicates which of the survivors will train best.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=4

PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.catalog_probe
echo "EXIT=$?"
