#!/bin/bash
#SBATCH --job-name=preprocess-emp643-testset
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Materialize the SAME labeled test-set craters from the EMP-643 mosaic, so
# they can be compared crop-for-crop against the WAC_GLOBAL morphology
# versions already on disk. Everything is written under its own
# emp643/-prefixed paths - the existing morphology products are left
# completely untouched, since the whole point is a head-to-head comparison.
#
# Three steps, in order, because --norm_mode global needs percentiles that
# only exist once there is a RAW .dat to measure them from (same two-pass
# flow the morphology pipeline used - see configs/global_scaling.json's own
# "source" field pointing at a *_raw_TEMP dat):
#   1. raw wide sample  -> percentile source
#   2. fit_global_scaling.py -> configs/global_scaling_emp643.json
#   3. the real test-set preprocess, norm_mode global, WITH --save_raw_crops
#      (the crops are what gets looked at)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

MOSAIC="data/raw/wac_emp643/sigma/100/highpass_filtered_emp643.tif"
BASE="data/processed_emp643/sigma/100"
RAW_TMP="$BASE/scaling_raw_TEMP"
TESTSET="$BASE/new_test_set"
SCALING="configs/global_scaling_emp643.json"

if [ ! -e "$MOSAIC" ]; then echo "missing $MOSAIC" >&2; exit 1; fi

# --- 1. raw sample for percentiles -----------------------------------------
if [ ! -e "$SCALING" ]; then
  mkdir -p "$RAW_TMP"
  N_RAW=20000
  PYTHONPATH=$(pwd) pixi run -e default python src/data/preprocess_2.py \
      --map_file "$MOSAIC" \
      --craters_csv data/raw/lunar_crater_database_robbins_2018.csv \
      --output_dir_clean "$RAW_TMP/crater_crops" \
      --np_output_path_clean "$RAW_TMP/craters_wide_raw.dat" \
      --info_output_path "$RAW_TMP/metadata.csv" \
      --min_diameter 3.0 --max_diameter 10.0 \
      --latitude_bounds -60 60 \
      --craters_to_output $N_RAW \
      --clean_offset 1.0 \
      --save_np_array \
      --autoencoder_model mae \
      --exclude_lon_bounds 180 270 \
      --norm_mode raw

  # --- 2. fit the fixed percentiles ---------------------------------------
  PYTHONPATH=$(pwd) pixi run -e default python fit_global_scaling.py \
      --clean-dat "$RAW_TMP/craters_wide_raw.dat" \
      --n $N_RAW \
      --out "$SCALING"
fi
echo "--- scaling ---"; cat "$SCALING"

# --- 3. the labeled test set, globally scaled, crops saved -----------------
# Same crater IDs, diameter band, FOV and latitude bounds as the morphology
# new_test_set (see slurm/preprocess_raw_check.sbatch, which reproduces that
# rule) so the only difference between the two sets is the SOURCE MOSAIC.
mkdir -p "$TESTSET"
PYTHONPATH=$(pwd) pixi run -e default python src/data/preprocess_2.py \
    --map_file "$MOSAIC" \
    --craters_csv data/raw/lunar_crater_database_robbins_2018.csv \
    --output_dir_clean "$TESTSET/crater_crops" \
    --np_output_path_clean "$TESTSET/craters.dat" \
    --info_output_path "$TESTSET/metadata.csv" \
    --min_diameter 1.0 --max_diameter 5.0 \
    --latitude_bounds -60 60 \
    --craters_to_output -1 \
    --only_crater_ids configs/correct_crater_with_labels_final.csv \
    --clean_offset 0.5 \
    --save_raw_crops \
    --save_np_array \
    --autoencoder_model mae \
    --norm_mode global \
    --scaling_json "$SCALING"
echo "EXIT=$?"
