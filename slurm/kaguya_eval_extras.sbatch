#!/bin/bash
#SBATCH --job-name=kaguya-eval-extras
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# The evaluation pieces train_dino_finetune_kaguya.sbatch does NOT do itself,
# for one finished Kaguya finetune (RUN=<run dir>, CATALOG, NORM):
#   1. full-pool k-NN labels (liorr labels -> ~98k craters): map, PCA (class /
#      band), UMAP, examples, CV accuracy   -> geomap/<cat>__<norm>/label_knn/
#   2. the pool in PCA/UMAP coloured by the map's k-means k=4/10 clusters
#   3. learning curve over the saved snapshots on the final_test half
#      -> <run>/learning_curve/
#   RUN=... CATALOG=kaguya_tco_raw NORM=asinh sbatch slurm/kaguya_eval_extras.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8 PYTHONUNBUFFERED=1
CKPT=$RUN/eval/final/teacher_checkpoint.pth
POOL=data/evalsets/s224/${CATALOG}__${NORM}__pool; LAB=data/evalsets/s224/${CATALOG}__${NORM}
OUT=$RUN/data_comparison/geomap/${CATALOG}__${NORM}; mkdir -p "$OUT"
NAME=$(echo "$RUN" | sed 's/.*unknown\///; s/\/d1-15\/nunknown//')
echo "=== $NAME"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_labeling --method knn \
  --pool-dat $POOL/craters.dat --pool-meta $POOL/metadata.csv \
  --labelled-dat $LAB/craters.dat --labelled-meta $LAB/metadata.csv \
  --labels-csv configs/liorr_labels.csv --checkpoint "$CKPT" --model dino \
  --out-dir "$OUT" --pool-embed-cache "$OUT/pool_embeddings.npy" --label "$NAME"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_projection --dir "$OUT"
PYTHONPATH=src:$(pwd) pixi run -e default python -m eval.learning_curve \
  --set ${CATALOG}__${NORM}_final_test --run-dir "$RUN"
echo "EXIT=0"
