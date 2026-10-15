#!/bin/bash
#SBATCH --job-name=highpass-div
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# morphology sigma-100 high-pass, PLUS divisive local-contrast normalization
# at sigma=500 px (50 km).
#
# The high-pass is SUBTRACTIVE: it removes the regional albedo offset (maria
# vs highland; terrain identifiability from crop brightness drops 92% -> 62%)
# and leaves a multiplicative GAIN untouched. The seam is a gain difference,
# which is why it survived the sigma-100 filter unchanged. The divisive step
# is the matched correction.
#
# sigma=500 px sits in the gap between crater scale (10-120 px for 1-12 km)
# and tile scale (~27000 px for 90 deg). A scattered estimate of the same
# field over 100k craters found its optimum at a 100 km radius - Gaussian
# sigma ~50 km - independently agreeing.
#
# Measured on that scattered proxy (morph_sigma100, k=10, 100k craters):
#   longitude eta^2     0.0957 -> 0.0001   (-99.9%)
#   bin median ratio     1.995 -> 1.029
#   cluster texture spread 4.135 -> 3.14   (-31.7%)
# i.e. the artifact drops ~3x more than the signal. This job builds the real
# thing so the claim can be tested on the model rather than on a statistic.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -m src.data.highpass_mosaic \
  --input data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif \
  --output data/raw/wac_mosaic_new_version/sigma/100_div500/highpass_filtered_lunar_mosaic.tif \
  --sigma 100 --divisive_sigma 500
echo "EXIT=$?"
