#!/bin/bash
#SBATCH --job-name=build-catalog
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# One stage-1 catalog per array task: 100k craters, 1-12 km, |lat|<=60,
# FULL longitude (the seam band is deliberately KEPT - it is under test),
# fov 1.0, NATIVE size, RAW values. See src/data/build_catalog.py.
#
# Identical crater filter args across all tasks, so preprocess_2's
# filtered.sample(random_state=42) yields the SAME 100k craters in every
# catalog - catalog differences can never be confounded with sampling.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=4

case "$SLURM_ARRAY_TASK_ID" in
  0) MAP=data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif;                          OUT=morph_raw;      F=none ;;
  1) MAP=data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif;      OUT=morph_sigma100; F=gaussian_highpass_sigma100 ;;
  2) MAP=data/raw/wac_mosaic_new_version/sigma/200/highpass_filtered_lunar_mosaic.tif;      OUT=morph_sigma200; F=gaussian_highpass_sigma200 ;;
  3) MAP=data/raw/wac_uniform/WAC_EMP_643_100m_180.tif;                                     OUT=emp_raw;        F=none ;;
  4) MAP=data/raw/wac_emp643/sigma/100/highpass_filtered_emp643.tif;                        OUT=emp_sigma100;   F=gaussian_highpass_sigma100 ;;
  5) MAP=data/raw/wac_emp643/sigma/200/highpass_filtered_emp643.tif;                        OUT=emp_sigma200;   F=gaussian_highpass_sigma200 ;;
  6) MAP=data/raw/wac_mosaic_new_version/sigma/100_div500/highpass_filtered_lunar_mosaic.tif; OUT=morph_sigma100_div500; F=gaussian_highpass_sigma100_then_divisive_sigma500 ;;
  *) echo "bad task id" >&2; exit 1 ;;
esac

echo "mosaic: $MAP"
# configs/catalog_sentinel_craters.csv holds the 5 craters (of 100,000) whose
# crops overlap a WAC_EMP_643 nodata hole. They are excluded from EVERY
# catalog, not just the EMP ones, so all six hold the same crater set - which
# is the premise that makes comparing them meaningful. --labels_csv is always
# excluded in addition to this; see build_catalog.main.
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.build_catalog \
  --map_file "$MAP" --out_dir "data/catalogs/$OUT" --filter_desc "$F" \
  --craters_to_output 100000 \
  --exclude_crater_ids configs/catalog_sentinel_craters.csv
echo "EXIT=$?"
