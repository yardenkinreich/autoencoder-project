#!/bin/bash
#SBATCH --job-name=geomap-catalogs
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# Geographic cluster maps over the FULL ~100k-crater pool, frozen
# off-the-shelf DINOv2, for one catalog per array task.
#
# WHY BOTH morph_sigma100 AND emp_sigma100 ARE (RE)RUN HERE
# --------------------------------------------------------
# A geographic map on morph_sigma100 already exists, but it was produced
# through the old preprocess_2 path with the historical [0,1] p1/p99 global
# scaling. Comparing a NEW emp_sigma100 map against it would confound two
# changes at once - the source mosaic AND the normalization. Both are
# therefore rebuilt from the catalogs with the SAME normalization and the
# SAME 100k crater set, so the only difference between the two maps is the
# source mosaic. That is the comparison being asked for.
#
# NORMALIZATION: global_wide. On the 30-cell labeled comparison it gave the
# best class separation of any normalization (silhouette 0.131 vs 0.082 for
# robust) while clipping essentially nothing - unlike `global`, whose 1st/99th
# window destroys 6-9% of the freshest craters' pixels and 0% of the most
# degraded. It is the bounded-range option without the label-correlated
# damage. See the dashboard's Data Processing page.
#
# BASEMAP: the plain morphology mosaic for BOTH maps, deliberately, even for
# the EMP run. Its job is to show the reader where they are looking; a
# high-pass residual has no maria/highland contrast and is unreadable as a
# map. See geographic_map._DEFAULT_BASEMAP.
#
# Outputs per k (4, 10, 100): a coloured PNG, a summary CSV, a per-crater CSV,
# and a GeoJSON for QGIS (points + cluster id, style as a categorical field).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

CATS=(morph_sigma100 emp_sigma100)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"
NORM=global_wide
SET="${CAT}__${NORM}__pool"

echo "=== $CAT / $NORM ==="

# ~6.5 GB memmap per catalog; --no-crops because 100k display crops are only
# used by labeled-set confound checks, which do not apply to an unlabeled pool.
if [ ! -f "data/evalsets/$SET/craters.dat" ]; then
  PYTHONPATH=$(pwd) pixi run -e default python -m src.data.catalog_to_evalset \
    --catalogs "data/catalogs/$CAT" --norms "$NORM" \
    --use-catalog-root --no-crops \
    --out-root data/evalsets \
    --config-out "configs/eval_suite_pool_${CAT}.yaml"
else
  echo "reusing existing data/evalsets/$SET"
fi

OUT="logs/data_profiles/geomap/${CAT}__${NORM}"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.geographic_map \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained \
  --dat_path "data/evalsets/$SET/craters.dat" \
  --metadata_csv "data/evalsets/$SET/metadata.csv" \
  --k 4 10 100 \
  --label "frozen DINOv2 — ${CAT} / ${NORM}" \
  --out_dir "$OUT"
echo "EXIT=$?"
