#!/bin/bash
#SBATCH --job-name=highpass-morph-s200
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# sigma=200 high-pass for the morph source, one of the six catalog
# variants (2 sources x {raw, sigma100, sigma200}). Uses
# src/data/highpass_mosaic.py, whose invalid-pixel guard fills holes with a
# FINITE value - an earlier version wrote NaN and a single NaN pixel
# silently normalized an entire 642-crater dataset to NaN.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.highpass_mosaic \
  --input data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif --output data/raw/wac_mosaic_new_version/sigma/200/highpass_filtered_lunar_mosaic.tif --sigma 200
echo "EXIT=$?"
