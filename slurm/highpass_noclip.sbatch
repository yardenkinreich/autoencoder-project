#!/bin/bash
#SBATCH --job-name=highpass-noclip
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# Rebuild all four high-pass products WITHOUT the filter's legacy global
# 0.1/99.9 percentile clip.
#
# WHY: local_highpass_filter_gpu used to clip its residual to the whole
# mosaic's 0.1/99.9 percentiles. Those bounds are set by smooth mare and
# highland, which is almost all of the Moon; craters are the high-contrast
# minority, so crater rims and shadows were flattened onto the bounds.
# Measured share of pixels pinned to exactly a bound, on the 642 labeled
# craters:
#
#   morphology sigma-100:  0.99% overall - v-fr 6.04%, r-fr 2.24%,
#                          r-dr 0.16%, v-dr 0.00%
#   EMP        sigma-100: 11.87% overall
#
# Monotonic in degradation and exactly zero for the most degraded class: the
# clip destroyed signal in direct proportion to crater freshness, which is
# the quantity under study. Every DINO/MAE/DIET-CP result in this project's
# history was trained and evaluated on sigma-100 data carrying this.
#
# The old products are MOVED ASIDE, not deleted, so the historical results
# remain reproducible (--legacy_clip regenerates them byte-for-byte in
# principle). Delete data/raw/**/sigma_legacyclip/ only after the retrained
# models have superseded every metric that depends on them.
#
# Outputs land on the SAME paths the catalog builder already reads, so
# slurm/build_catalog.sbatch needs no edit - but note the products are now
# unclipped, which is the whole point.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

case "$SLURM_ARRAY_TASK_ID" in
  0) IN=data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif
     OUT=data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif
     SIGMA=100 ;;
  1) IN=data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif
     OUT=data/raw/wac_mosaic_new_version/sigma/200/highpass_filtered_lunar_mosaic.tif
     SIGMA=200 ;;
  2) IN=data/raw/wac_uniform/WAC_EMP_643_100m_180.tif
     OUT=data/raw/wac_emp643/sigma/100/highpass_filtered_emp643.tif
     SIGMA=100 ;;
  3) IN=data/raw/wac_uniform/WAC_EMP_643_100m_180.tif
     OUT=data/raw/wac_emp643/sigma/200/highpass_filtered_emp643.tif
     SIGMA=200 ;;
  *) echo "bad task id" >&2; exit 1 ;;
esac

# highpass_mosaic.py refuses to overwrite a multi-GB derived product, so the
# clipped one is moved to a sibling *_legacyclip directory first. Keeping it
# means the provenance question "what exactly did the old runs see?" stays
# answerable rather than becoming folklore.
if [ -f "$OUT" ]; then
  LEGACY_DIR="$(dirname "$(dirname "$OUT")")_legacyclip/$(basename "$(dirname "$OUT")")"
  mkdir -p "$LEGACY_DIR"
  echo "moving clipped product aside: $OUT -> $LEGACY_DIR/"
  mv "$OUT" "$LEGACY_DIR/"
fi

echo "input:  $IN"
echo "output: $OUT  (sigma=$SIGMA, NO legacy clip)"
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.highpass_mosaic \
  --input "$IN" --output "$OUT" --sigma "$SIGMA"
echo "EXIT=$?"
