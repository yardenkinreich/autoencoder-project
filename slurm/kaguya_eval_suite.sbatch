#!/bin/bash
#SBATCH --job-name=kaguya-eval-suite
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# Full frozen-DINOv2 evaluation of one Kaguya catalog per array task, at
# 224 px (DINOv2's own input size; 128 would throw away Kaguya's resolution -
# a 1 km crater at 2xD is 237 native px). Crops are area-averaged when shrunk.
#   0  kaguya_tco_raw    norm=asinh   (catalog loader policy)
#   1  kaguya_tco_ratio  norm=robust
# Steps:
#   1. labeled eval set: the 1,030 baseline-labeled craters (configs/liorr_labels.csv)
#   2. eval suite on it: silhouette/mAP/ARI/NMI/QWK + confound checks, PCA &
#      UMAP latent plots, k-means cluster plots with crater thumbnails
#   3. full ~98k pool eval set (no display crops)
#   4. geographic cluster maps at k = 4, 10, 100 (PNG, CSV, GeoJSON for QGIS)
#   5. per-cluster profiles: mean crater image, example craters, brightness/
#      texture/size/geography per cluster
# sbatch --array=0-1 slurm/kaguya_eval_suite.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
# single-threaded BLAS/OpenMP, as in catalog_eval_suite.sbatch: with 4 threads the
# first run (job 1047) deadlocked at 100% CPU inside the metric suite
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

case "$SLURM_ARRAY_TASK_ID" in
  0) CAT=kaguya_tco_raw;   NORM=asinh ;;
  1) CAT=kaguya_tco_ratio; NORM=robust ;;
  2) CAT=kaguya_tco_sub;   NORM=robust ;;
  *) echo "bad task id" >&2; exit 1 ;;
esac
SIZE=224
ROOT=data/evalsets/s$SIZE
CKPT=data/raw/pretrained/dinov2_vits14_pretrain.pth
TAG="${CAT}__${NORM}__s${SIZE}"
# The dashboard's Deep Dive -> "data comparison" section reads the STOCK
# checkpoint's run dir: data_comparison/eval/<catalog>/ (metric suite) and
# data_comparison/geomap/<catalog>__<norm>/ (maps + cluster profiles).
STOCK=logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison
echo "=== $TAG"

# (outputs land where the dashboard looks - see STOCK above)
# 1. labeled eval set
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.catalog_to_evalset \
  --catalogs "data/catalogs/$CAT" --norms "$NORM" --size $SIZE --resize-interp auto \
  --labels-csv configs/liorr_labels.csv \
  --out-root $ROOT --config-out "configs/eval_suite_catalog_${CAT}_s${SIZE}.yaml"

# 2. eval suite (frozen stock DINOv2)
PYTHONPATH=src:$(pwd) pixi run -e default python -m eval.evaluate \
  --checkpoint $CKPT --autoencoder-model dino_pretrained \
  --config "configs/eval_suite_catalog_${CAT}_s${SIZE}.yaml" \
  --out "$STOCK/eval/$CAT" \
  --history logs/catalog_eval_history.csv \
  --tag "kaguya_frozen_dino_${TAG}" --n-boot 2000

# 3. full pool eval set
POOL="$ROOT/${CAT}__${NORM}__pool"
if [ ! -f "$POOL/craters.dat" ]; then
  PYTHONPATH=$(pwd) pixi run -e default python -m src.data.catalog_to_evalset \
    --catalogs "data/catalogs/$CAT" --norms "$NORM" --size $SIZE --resize-interp auto \
    --use-catalog-root --no-crops --out-root $ROOT \
    --config-out "configs/eval_suite_pool_${CAT}_s${SIZE}.yaml"
fi

# 4. geographic cluster maps
OUT="$STOCK/geomap/${CAT}__${NORM}"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.geographic_map \
  --checkpoint $CKPT --autoencoder-model dino_pretrained \
  --dat_path "$POOL/craters.dat" --metadata_csv "$POOL/metadata.csv" \
  --k 4 10 100 --label "frozen DINOv2 — Kaguya ${CAT#kaguya_tco_} / ${NORM} / ${SIZE}px" \
  --out_dir "$OUT"

# 5. per-cluster profiles (size comes from the eval set's manifest)
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.cluster_profile --dir "$OUT" --overwrite
echo "EXIT=$?"
