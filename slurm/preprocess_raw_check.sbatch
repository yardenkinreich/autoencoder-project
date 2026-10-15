#!/bin/bash
#SBATCH --job-name=preprocess-raw-check
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Re-materializes new_test_set's 642 craters with --norm_mode raw (no [0,1]
# clip+rescale at all - see preprocess_2.py's normalize()/normalize_global())
# instead of the usual --norm_mode global, to check whether the brightness-
# shortcut dominance we keep finding is a property of the underlying
# imagery itself, or an artifact introduced by the global percentile
# scaling. Same crater IDs, same crop offset (0.5x, narrow FOV, matching
# what julie/new_set/new_set_final_test all actually evaluate against),
# same everything else as the real preprocess_new_test_set rule - only
# --norm_mode differs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

OUT_DIR="data/processed_wac_100m_new/sigma/100/new_test_set_rawcheck"
rm -rf "$OUT_DIR"

PYTHONPATH=$(pwd) pixi run -e default python src/data/preprocess_2.py \
    --map_file data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif \
    --craters_csv data/raw/lunar_crater_database_robbins_2018.csv \
    --output_dir_clean "$OUT_DIR/crater_crops" \
    --np_output_path_clean "$OUT_DIR/craters.dat" \
    --info_output_path "$OUT_DIR/metadata.csv" \
    --min_diameter 1.0 --max_diameter 5.0 \
    --latitude_bounds -60 60 \
    --craters_to_output -1 \
    --only_crater_ids configs/correct_crater_with_labels_final.csv \
    --clean_offset 0.5 \
    --save_raw_crops \
    --save_np_array \
    --autoencoder_model mae \
    --norm_mode raw
echo "EXIT=$?"
