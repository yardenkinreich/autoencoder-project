#!/bin/bash
#SBATCH --job-name=pool-all
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# Every labelling method x every variant, into its own label_<tag>/ dir.
# Pool embeddings are cached per variant, so re-running to add a figure costs
# minutes rather than the full embedding pass.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
G=logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/geomap
NORM=global_wide
# EMP is deliberately NOT in the default list. It remains buildable -
# pass CATS/VARIANTS explicitly to include it - but the routine
# comparison is morphology vs its divisive-corrected version.
CATS="${CATS:-morph_sigma100 morph_sigma100_div500}"
for CAT in $CATS; do
  POOL="data/evalsets/${CAT}__${NORM}__pool"; LAB="data/evalsets/${CAT}__${NORM}"
  OUT="$G/${CAT}__${NORM}"
  C="--pool-dat $POOL/craters.dat --pool-meta $POOL/metadata.csv \
     --labelled-dat $LAB/craters.dat --labelled-meta $LAB/metadata.csv \
     --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
     --model dino_pretrained --out-dir $OUT \
     --pool-embed-cache $OUT/pool_embeddings.npy \
     --label frozen-DINOv2-${CAT}"
  echo "########## $CAT :: knn (4-class, plain)"
  PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_labeling --method knn $C
  echo "########## $CAT :: knn_balanced (4-class, prior-corrected)"
  PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_labeling --method knn_balanced $C
  echo "########## $CAT :: knn_binary (fresh vs degraded)"
  PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_labeling --method knn $C \
    --merge-classes "0,1 2,3" --class-names fresh degraded --tag knn_binary
done
echo "EXIT=$?"
