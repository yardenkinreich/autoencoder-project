#!/bin/bash
#SBATCH --job-name=div-pipeline
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Everything downstream of the divisive mosaic, in the one order that works.
# Runs after slurm/highpass_divisive.sbatch has produced
# data/raw/wac_mosaic_new_version/sigma/100_div500/.
#
#   1. catalog          100k craters, raw + native, same crater set as the other six
#   2. norm constants   norm_stats.json + loader.json (asinh for raw, robust otherwise)
#   3. DATA PROCESSING  processing_compare -> the app's Data Processing page:
#                       example craters per class, per-class clipping, distributions.
#                       Re-run over ALL variants, not just the new one, so the new
#                       column appears beside the existing six on identical craters.
#   4. eval suite       the 642 labeled craters x 5 normalizations, full metric suite
#   5. geomap           100k craters, k=4/10/100, + gpkg/qml + cluster profile
#
# The decision this exists to settle: does divisive normalization reduce the
# longitude artifact (lon_eta, Cramer's V) proportionally MORE than it costs
# class separation (silhouette_true, mAP, ARI)? A scattered proxy over 100k
# craters said artifact -99.9% vs signal -31.7%; this is the real test.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CAT=morph_sigma100_div500
NORM=global_wide
P=$(pwd)

echo "=== 1. catalog ==="
if [ ! -f "data/catalogs/$CAT/index.csv" ]; then
  PYTHONPATH=$P pixi run -e default python -m src.data.build_catalog \
    --map_file data/raw/wac_mosaic_new_version/sigma/100_div500/highpass_filtered_lunar_mosaic.tif \
    --out_dir "data/catalogs/$CAT" \
    --filter_desc gaussian_highpass_sigma100_then_divisive_sigma500 \
    --craters_to_output 100000 \
    --exclude_crater_ids configs/catalog_sentinel_craters.csv
else
  echo "reusing data/catalogs/$CAT"
fi

echo; echo "=== 2. normalization constants + loader policy ==="
PYTHONPATH=$P pixi run -e default python -m src.data.norm_stats --catalog "data/catalogs/$CAT"
PYTHONPATH=$P pixi run -e default python - "data/catalogs/$CAT" <<'PYX'
import json, os, sys
cat = sys.argv[1]
# robust, not asinh: this is a high-pass residual that has additionally been
# divided by its own local contrast, so its tail is already compressed twice
# over. asinh is for raw reflectance, whose skew no affine map can fix.
policy = {"norm": "robust", "resize": None, "stats_from": None,
          "diam_min": None, "diam_max": None, "lat_abs_max": None,
          "num_channels": 1,
          "_why": "sigma-100 high-pass THEN divisive local-contrast normalization "
                  "(sigma=500 px). robust: no clip, and the divisive step has "
                  "already flattened the contrast distribution."}
json.dump(policy, open(os.path.join(cat, "loader.json"), "w"), indent=2)
ts = os.path.join(cat, "testset")
if os.path.isdir(ts):
    p = dict(policy); p["stats_from"] = cat
    json.dump(p, open(os.path.join(ts, "loader.json"), "w"), indent=2)
print("wrote loader.json (+ testset)")
PYX

echo; echo "=== 3. data-processing evaluation (the app's Data Processing page) ==="
# --no-cache: the cached raw crops predate this variant, and a stale cache
# would silently compare new statistics against old pixels.
PYTHONPATH=$P pixi run -e default python -m src.eval.processing_compare --no-cache

echo; echo "=== 4. eval suite: 642 labeled craters x 5 normalizations ==="
CFG="configs/eval_suite_catalog_${CAT}.yaml"
PYTHONPATH=$P pixi run -e default python -m src.data.catalog_to_evalset \
  --catalogs "data/catalogs/$CAT" --norms global global_wide zscore robust asinh \
  --out-root data/evalsets --config-out "$CFG"
PYTHONPATH=src:$P pixi run -e default python -m eval.evaluate \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained --config "$CFG" \
  --out "logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/eval/$CAT" \
  --history logs/catalog_eval_history.csv --tag "catalog_data_comparison_${CAT}" \
  --n-boot 2000

echo; echo "=== 5. geographic map over the 100k pool ==="
SET="${CAT}__${NORM}__pool"
if [ ! -f "data/evalsets/$SET/craters.dat" ]; then
  PYTHONPATH=$P pixi run -e default python -m src.data.catalog_to_evalset \
    --catalogs "data/catalogs/$CAT" --norms "$NORM" --use-catalog-root --no-crops \
    --out-root data/evalsets --config-out "configs/eval_suite_pool_${CAT}.yaml"
fi
OUT="logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/geomap/${CAT}__${NORM}"
PYTHONPATH=$P pixi run -e default python -m src.eval.geographic_map \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained \
  --dat_path "data/evalsets/$SET/craters.dat" \
  --metadata_csv "data/evalsets/$SET/metadata.csv" \
  --k 4 10 100 --label "frozen DINOv2 — ${CAT} / ${NORM}" --out_dir "$OUT"
PYTHONPATH=$P pixi run -e default python -m src.eval.backfill_geojson --dir "$OUT" --overwrite
PYTHONPATH=$P pixi run -e default python -m src.eval.cluster_profile --dir "$OUT" --overwrite
PYTHONPATH=$P pixi run -e default python -m src.eval.sobel_gain_test --map-dir "$OUT" --k 10 --radii-km 100
echo "EXIT=$?"
