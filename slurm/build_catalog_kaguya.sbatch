#!/bin/bash
#SBATCH --job-name=build-catalog-kaguya
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# Kaguya TC native-resolution catalogs (8.42 m/px, local aeqd, ragged),
# one per array task, SAME craters in the same order:
#   0  kaguya_tco_raw       reflectance
#   1  kaguya_tco_ratio     reflectance / exp(overview log low-pass), sigma=max(10 km, 2D)
#   2  kaguya_tco_sub       reflectance - exp(overview log low-pass) (WAC-style high-pass)
# Needs data/raw/kaguya_tco_overview (kaguya_overview.sbatch) and, for
# REGISTER=robbins, data/raw/kaguya_tco_registration (kaguya_register.sbatch).
#
#   SMOKE=1 sbatch --array=0-1 slurm/build_catalog_kaguya.sbatch   # 500 craters -> *_smoke
#           sbatch --array=0-1 slurm/build_catalog_kaguya.sbatch   # full 100k
# Env: N (craters, default 100000), OFFSET (FOV, default 1.0 = 2xD),
#      REGISTER (robbins|none, default robbins).
#
# Exclusions: the new labeled set (configs/liorr_labels.csv, also built as
# testset/), the old labeled set and the holdout registry - unioned by
# build_catalog. Afterwards: catalog_to_evalset.py + catalog_postprocess for
# these catalogs, as for the WAC ones.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1

N=${N:-100000}; OFFSET=${OFFSET:-1.0}; REGISTER=${REGISTER:-robbins}; SUFFIX=""
if [ "${SMOKE:-0}" = "1" ]; then N=500; SUFFIX="_smoke"; fi

case "$SLURM_ARRAY_TASK_ID" in
  0) ALB=none;     OUT=kaguya_tco_raw$SUFFIX;      F=none ;;
  1) ALB=ratio;    OUT=kaguya_tco_ratio$SUFFIX;    F=ratio_sigma_max10km_2D ;;
  2) ALB=subtract; OUT=kaguya_tco_sub$SUFFIX;      F=highpass_subtract_sigma_max10km_2D ;;
  *) echo "bad task id" >&2; exit 1 ;;
esac

PYTHONPATH=$(pwd) pixi run -e default python -m src.data.build_catalog \
  --source kaguya_tco --albedo "$ALB" --register "$REGISTER" \
  --out_dir "data/catalogs/$OUT" --filter_desc "$F" \
  --min_diameter 1 --max_diameter 15 --clean_offset "$OFFSET" \
  --craters_to_output "$N" --workers 32 --max_frac_lowsat "${MAX_LOWSAT:-0.2}" \
  --labels_csv configs/liorr_labels.csv \
  --exclude_crater_ids configs/correct_crater_with_labels_final.csv configs/holdout_crater_ids.csv
echo "EXIT=$?"
