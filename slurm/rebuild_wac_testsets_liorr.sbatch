#!/bin/bash
#SBATCH --job-name=wac-testsets-liorr
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# Re-cut every WAC catalog's labeled testset/ for the BASELINE labels
# (configs/liorr_labels.csv, 1,162 craters; 1,037 inside the mosaic's +-60 deg)
# - DATA ONLY, no model evaluation. Per catalog (one array task each):
#   1. testset/, seedset/ and the catalog's labeled evalsets move to *_legacy642
#      (the old 642-crater set stays available for comparison with past runs),
#      and the old generated eval config to configs/legacy642/ (paths rewritten)
#   2. build_catalog --testset_only: main pool, scaling.json, norm_stats untouched
#   3. the testset's loader.json is restored (stats_from = parent, unchanged)
#   4. catalog_to_evalset regenerates data/evalsets/<cat>__<norm> + the config
#   sbatch --array=0-6 slurm/rebuild_wac_testsets_liorr.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=4

CATS=(morph_raw morph_sigma100 morph_sigma200 emp_raw emp_sigma100 emp_sigma200 morph_sigma100_div500)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"
D="data/catalogs/$CAT"
MAP=$(pixi run -e default python -c "import json;print(json.load(open('$D/catalog_manifest.json'))['map_file'])")
FILTER=$(pixi run -e default python -c "import json;print(json.load(open('$D/catalog_manifest.json'))['filter'])")
NORMS="global global_wide zscore robust asinh"
echo "=== $CAT  map=$MAP  filter=$FILTER"

# 1. keep the old set
[ -d "$D/testset" ] && [ ! -d "$D/testset_legacy642" ] && mv "$D/testset" "$D/testset_legacy642"
[ -d "$D/seedset" ] && [ ! -d "$D/seedset_legacy642" ] && mv "$D/seedset" "$D/seedset_legacy642"
mkdir -p data/evalsets_legacy642 configs/legacy642
for e in data/evalsets/${CAT}__*; do
  case "$e" in *__pool) continue ;; esac
  [ -d "$e" ] && mv "$e" data/evalsets_legacy642/
done
CFG="configs/eval_suite_catalog_${CAT}.yaml"
[ -f "$CFG" ] && [ ! -f "configs/legacy642/$(basename $CFG)" ] && \
  sed 's|data/evalsets/|data/evalsets_legacy642/|g' "$CFG" > "configs/legacy642/$(basename $CFG)"

# 2. new testset (no diameter band: the label file defines the set)
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.build_catalog \
  --map_file "$MAP" --out_dir "$D" --filter_desc "$FILTER" --testset_only \
  --labels_csv configs/liorr_labels.csv

# 3. same loading policy as before (inherits the parent's constants)
cp "$D/testset_legacy642/loader.json" "$D/testset/loader.json"

# 4. eval sets + config for the new labels
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.catalog_to_evalset \
  --catalogs "$D" --norms $NORMS --labels-csv configs/liorr_labels.csv \
  --out-root data/evalsets --config-out "$CFG"
echo "EXIT=$?"
