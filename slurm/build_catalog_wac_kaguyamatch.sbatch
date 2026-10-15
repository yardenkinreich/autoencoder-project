#!/bin/bash
#SBATCH --job-name=build-catalog-wac-kmatch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# The classic WAC albedo-removed data (morphology mosaic, subtractive Gaussian
# high-pass sigma=100 px = 10 km) cut on EXACTLY the Kaguya catalogs' craters
# (configs/kaguya_tco_pool_ids.csv, 97,911 - exported from
# data/catalogs/kaguya_tco_raw/index.csv), so every WAC-vs-Kaguya comparison is
# same craters, same framing:
#   1-15 km, FOV 2xD (offset 1.0), FULL longitude (the 180-270 band included),
#   labeled testset = configs/liorr_labels.csv at the same 2xD framing.
# Native WAC resolution (~100 m/px), raw high-pass values; normalization is
# fitted afterwards by slurm/catalog_postprocess.sbatch.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=4

PYTHONPATH=$(pwd) pixi run -e default python -m src.data.build_catalog \
  --map_file data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif \
  --out_dir data/catalogs/morph_sigma100_kaguyamatch \
  --filter_desc gaussian_highpass_sigma100 \
  --min_diameter 1 --max_diameter 15 --clean_offset 1.0 --testset_offset 1.0 \
  --craters_to_output -1 --only_crater_ids configs/kaguya_tco_pool_ids.csv \
  --labels_csv configs/liorr_labels.csv \
  --exclude_crater_ids configs/correct_crater_with_labels_final.csv configs/holdout_crater_ids.csv
echo "EXIT=$?"
