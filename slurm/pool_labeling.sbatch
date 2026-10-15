#!/bin/bash
#SBATCH --job-name=pool-label
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Label the 100k pool by a SUPERVISED method instead of KMeans, propagating
# the 642 human labels out to the whole pool, and write a complete parallel
# set of artifacts per method under label_<method>/.
#
# The labelled set and the pool MUST be the same catalog and the same
# normalization, or the nearest neighbours are being measured across two
# different data presentations. That is why LAB is derived from CAT/NORM here
# rather than being a free argument.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NORM=global_wide
METHODS="${METHODS:-knn}"
CKPT=data/raw/pretrained/dinov2_vits14_pretrain.pth
G=logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/geomap

# EMP is deliberately NOT in the default list. It remains buildable -
# pass CATS/VARIANTS explicitly to include it - but the routine
# comparison is morphology vs its divisive-corrected version.
CATS="${CATS:-morph_sigma100 morph_sigma100_div500}"
for CAT in $CATS; do
  POOL="data/evalsets/${CAT}__${NORM}__pool"
  LAB="data/evalsets/${CAT}__${NORM}"
  OUT="$G/${CAT}__${NORM}"
  if [ ! -f "$POOL/craters.dat" ] || [ ! -f "$LAB/craters.dat" ]; then
    echo "!! skipping $CAT - missing $POOL or $LAB"; continue
  fi
  for M in $METHODS; do
    echo "=========== $CAT / $NORM / method=$M"
    PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_labeling \
      --method "$M" \
      --pool-dat  "$POOL/craters.dat"  --pool-meta  "$POOL/metadata.csv" \
      --labelled-dat "$LAB/craters.dat" --labelled-meta "$LAB/metadata.csv" \
      --checkpoint "$CKPT" --model dino_pretrained \
      --out-dir "$OUT" \
      --pool-embed-cache "$OUT/pool_embeddings.npy" \
      --label "frozen DINOv2 — ${CAT} / ${NORM}"
  done
done
echo "EXIT=$?"
