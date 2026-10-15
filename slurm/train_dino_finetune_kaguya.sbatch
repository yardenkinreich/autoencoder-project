#!/bin/bash
#SBATCH --job-name=dino-ft-kaguya
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# DINOv2 finetune (stock ViT-S/14 warm start) on a Kaguya catalog, then the
# full evaluation, all in one job so the results land in the app:
#   train  : configs/dino_craters_finetune.yaml - context(2xD)+zoom(1xD) teacher
#            views; OBJECTIVE=proto adds the prototypical loss on the train_pool
#            half of configs/liorr_labels_split.csv (from <catalog>/testset);
#            OBJECTIVE=ssl is self-distillation only
#   eval   : eval.evaluate on configs/eval_suite.yaml (Kaguya sets first; a proto
#            checkpoint is scored on the final_test half only - the full sets
#            hold its training craters)
#   maps   : 98k-pool geographic clusters k=4/10/100 + GeoPackage + cluster profiles
#            under <run>/data_comparison/geomap/<catalog>__<norm>
# Env: OBJECTIVE=proto|supcon|ssl (default proto; supcon = ordinal SupCon, proto off),
#      KOLEO=<weight> (override dino.koleo_loss_weight, run dir gets _koleo<w>),
#      SUFFIX=<tag> (appended to the run dir, e.g. tw5 for the 5-epoch teacher warmup),
#      EXTRA_OPTS="key=value ..." (more config overrides, e.g. loss weights), CATALOG (default kaguya_tco_ratio),
#      NORM (pool eval norm, default robust), EPOCHS (default 10 x 100 iterations),
#      SMOKE=1 (20 iterations into logs/dino_smoke/, no eval) .
#   OBJECTIVE=proto sbatch slurm/train_dino_finetune_kaguya.sbatch
#   OBJECTIVE=ssl   sbatch slurm/train_dino_finetune_kaguya.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID          # torchrun env, not srun's - see train_dino_finetune.sbatch
export PYTHONUNBUFFERED=1

OBJECTIVE=${OBJECTIVE:-proto}; CATALOG=${CATALOG:-kaguya_tco_ratio}; NORM=${NORM:-robust}
EPOCHS=${EPOCHS:-10}; CAT_DIR=data/catalogs/$CATALOG
PRETRAINED=data/raw/pretrained/dinov2_vits14_pretrain.pth
# proto: ssl + prototypical | supcon: ssl + ordinal SupCon (proto off) | ssl: neither
PROTO_FLAG=""; OBJ_OPTS=(); OBJ_NAME="ssl+proto"
case "$OBJECTIVE" in
  ssl)    PROTO_FLAG="--no_proto"; OBJ_NAME="ssl" ;;
  supcon) PROTO_FLAG="--no_proto"; OBJ_OPTS=(supcon.enabled=true); OBJ_NAME="ssl+supcon" ;;
esac
# KOLEO=<w> overrides dino.koleo_loss_weight (default 0.1) and tags the run dir
[ -n "${KOLEO:-}" ] && OBJ_OPTS+=(dino.koleo_loss_weight=$KOLEO)
# EXTRA_OPTS="a.b=1 c.d=2": any further config overrides (e.g. loss weights)
[ -n "${EXTRA_OPTS:-}" ] && OBJ_OPTS+=($EXTRA_OPTS)
DIAM=$(pixi run -e default python -c "import json;d=json.load(open('$CAT_DIR/catalog_manifest.json'))['diameter_km'];print(d[0],d[1])")
read DMIN DMAX <<< "$DIAM"

if [ "${SMOKE:-0}" = "1" ]; then
  RUN_DIR=logs/dino_smoke/${CATALOG}_${OBJECTIVE}_$(date +%Y%m%d-%H%M%S)
  EXTRA=(--epochs 1 --opts train.OFFICIAL_EPOCH_LENGTH=20 evaluation.eval_period_iterations=20 train.num_workers=16 "${OBJ_OPTS[@]}")
else
  RUN_DIR=$(PYTHONPATH=src pixi run -e default python -c "
from run_layout import canonical_run_dir
print(canonical_run_dir({'family':'dino','source':'finetune','pretrained_weights':'$PRETRAINED',
  'epochs':$EPOCHS,'diameter_range':[$DMIN,$DMAX],'data_tag':'$CATALOG','freeze_until':None,
  'dino_objective':'$OBJ_NAME'}))")
  [ -n "${KOLEO:-}" ] && RUN_DIR="${RUN_DIR}_koleo${KOLEO}"
  [ -n "${SUFFIX:-}" ] && RUN_DIR="${RUN_DIR}_${SUFFIX}"
  EXTRA=(--epochs "$EPOCHS" --opts train.num_workers=16 "${OBJ_OPTS[@]}")
fi
mkdir -p "$RUN_DIR"
echo "=== train: $CATALOG objective=$OBJECTIVE -> $RUN_DIR"

export OMP_NUM_THREADS=1
PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=1 src/train/train_dino.py \
  --input "$CAT_DIR" $PROTO_FLAG --output_dir "$RUN_DIR" \
  --config_file configs/dino_craters_finetune.yaml --pretrained_weights "$PRETRAINED" \
  --min_diameter "$DMIN" --max_diameter "$DMAX" --data_tag "$CATALOG" "${EXTRA[@]}"
CKPT="$RUN_DIR/eval/final/teacher_checkpoint.pth"
[ "${SMOKE:-0}" = "1" ] && { ls -la "$CKPT"; echo "EXIT=0 (smoke)"; exit 0; }

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "=== eval suite"
PYTHONPATH=src:$(pwd) pixi run -e default python -m eval.evaluate \
  --checkpoint "$CKPT" --autoencoder-model dino --config configs/eval_suite.yaml \
  --tag "kaguya_finetune_${CATALOG}_${OBJECTIVE}" --n-boot 2000 --skip-holdout

echo "=== geographic maps (98k pool)"
POOL=data/evalsets/s224/${CATALOG}__${NORM}__pool
OUT="$RUN_DIR/data_comparison/geomap/${CATALOG}__${NORM}"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.geographic_map \
  --checkpoint "$CKPT" --autoencoder-model dino \
  --dat_path "$POOL/craters.dat" --metadata_csv "$POOL/metadata.csv" --k 4 10 100 \
  --label "DINOv2 finetuned ($OBJECTIVE, ${EPOCHS}x100 it) — Kaguya ${CATALOG#kaguya_tco_} / $NORM / 224px" \
  --out_dir "$OUT"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.backfill_geojson --dir "$OUT" --overwrite
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.cluster_profile --dir "$OUT" --overwrite
echo "EXIT=0"
