#!/bin/bash
#SBATCH --job-name=highpass-emp643
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Apply this project's own sigma=100 high-pass filter to the
# photometrically normalized WAC_EMP_643 mosaic, so it can be compared
# head-to-head against the WAC_GLOBAL morphology mosaic the pipeline
# currently uses. See src/data/highpass_mosaic.py's module docstring, and
# note the nodata guard there - EMP declares nodata=-3.4e38 where
# WAC_GLOBAL declared none, and unguarded that would smear catastrophic
# values across a ~100-pixel radius.
#
# Large --mem on purpose: this reads the whole ~16GB float32 raster into
# RAM (process_full_mosaic.py's own comment warns about exactly this) and
# the filter needs working copies on top.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -m src.data.highpass_mosaic \
  --input data/raw/wac_uniform/WAC_EMP_643_100m_180.tif \
  --output data/raw/wac_emp643/sigma/100/highpass_filtered_emp643.tif \
  --sigma 100
echo "EXIT=$?"
