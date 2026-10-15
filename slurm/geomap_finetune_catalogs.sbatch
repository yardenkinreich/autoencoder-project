#!/bin/bash
#SBATCH --job-name=geomap-ft
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# The FINETUNED DINOv2 checkpoint over the same catalog-derived 100k pools the
# frozen checkpoint already has, so the two become comparable on equal terms.
#
# WHY THIS IS NEEDED: the finetuned run's only existing geographic map came
# from the older preprocess_2 pipeline, which EXCLUDED a 90-degree longitude
# band around the mosaic seam. Comparing that against the frozen model's
# full-longitude catalog maps would compare two different craters sets and two
# different longitude coverages at once - the dashboard warns about exactly
# this when two runs resolve to different variants. Same pool, same
# normalization, same k values here; only the checkpoint differs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CATS=(morph_sigma100 emp_sigma100)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"
NORM=global_wide
SET="${CAT}__${NORM}__pool"
CKPT=logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10/eval/final/teacher_checkpoint.pth
RUN_DIR=logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10

# The pools were materialized by slurm/geomap_catalogs.sbatch; reuse them.
if [ ! -f "data/evalsets/$SET/craters.dat" ]; then
  echo "missing data/evalsets/$SET - run slurm/geomap_catalogs.sbatch first" >&2
  exit 1
fi

OUT="$RUN_DIR/data_comparison/geomap/${CAT}__${NORM}"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.geographic_map \
  --checkpoint "$CKPT" --autoencoder-model dino \
  --dat_path "data/evalsets/$SET/craters.dat" \
  --metadata_csv "data/evalsets/$SET/metadata.csv" \
  --k 4 10 100 \
  --label "DINOv2 finetuned ep10 — ${CAT} / ${NORM}" \
  --out_dir "$OUT"

PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.backfill_geojson --dir "$OUT" --overwrite
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.cluster_profile --dir "$OUT" --overwrite
echo "EXIT=$?"
