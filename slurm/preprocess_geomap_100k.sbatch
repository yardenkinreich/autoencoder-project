#!/bin/bash
#SBATCH --job-name=preprocess-geomap-100k
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# ~100k craters at 1-12km, wide (~2x diameter) FOV, for the global
# geographic cluster map. Mirrors Snakefile's preprocess_craters_dino rule
# exactly (same mosaic, same global scaling, same seam-band exclusion, same
# latitude coverage limit, same held-out-ID exclusion) with two deliberate
# differences:
#
#   1. --min_diameter 1.0 --max_diameter 12.0 instead of the training pool's
#      3-10km: this map is meant to cover the Moon broadly, and 1-12km is
#      the range discussed for the scale-up. Note the models being mapped
#      were trained on 3-10km, so the 1-3km and 10-12km craters are out of
#      their training distribution - that's intentional here (it's a
#      visualization of how the encoder organizes the surface, not an
#      in-distribution benchmark) but worth remembering when reading it.
#   2. No --save_raw_crops: 100k individual .npy crop files aren't needed
#      for a map (only the memmap + lat/lon metadata are), and skipping
#      them saves both disk and a large amount of per-crater file I/O.
#
# --exclude_lon_bounds 180 270 is kept, so the map will have a deliberate
# 90-degree gap over the mosaic-seam band - that band's incidence-angle
# discontinuities are exactly the kind of imagery artifact that would
# produce meaningless clusters, and the models never saw it in training.
# --craters_to_output uses preprocess_2.py's filtered.sample(random_state=
# 42), i.e. a RANDOM subsample, so the 100k are spatially unbiased rather
# than a catalog-order (and therefore possibly geographically clumped) slice.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

OUT_DIR="data/processed_wac_100m_new/sigma/100/geomap_100k_d1-12"
if [ -e "$OUT_DIR/metadata.csv" ]; then
    echo "Refusing to run: $OUT_DIR already populated." >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

PYTHONPATH=$(pwd) pixi run -e default python src/data/preprocess_2.py \
    --map_file data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif \
    --craters_csv data/raw/lunar_crater_database_robbins_2018.csv \
    --output_dir_clean "$OUT_DIR/crater_crops" \
    --np_output_path_clean "$OUT_DIR/craters_wide.dat" \
    --info_output_path "$OUT_DIR/metadata.csv" \
    --min_diameter 1.0 --max_diameter 12.0 \
    --latitude_bounds -60 60 \
    --craters_to_output 100000 \
    --clean_offset 1.0 \
    --save_np_array \
    --autoencoder_model mae \
    --exclude_lon_bounds 180 270 \
    --exclude_crater_ids configs/correct_crater_with_labels_final.csv \
    --norm_mode global \
    --scaling_json configs/global_scaling.json
echo "EXIT=$?"
