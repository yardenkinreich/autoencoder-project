#!/bin/bash
#SBATCH --job-name=kaguya-pool-label
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# Full-pool (~98k Kaguya craters) views for one model per array task: k-NN
# label propagation from the 1,030 liorr-labelled craters (+ its CV accuracy),
# PCA (by class and by acquisition band), UMAP, class map and examples - written
# into that model's data_comparison/geomap/<catalog>__<norm>/label_knn/, which
# the app shows under the geographic section.  sbatch --array=0-7 ...
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8 PYTHONUNBUFFERED=1
STOCK=logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown
FT=logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown
SP=data/raw/pretrained/dinov2_vits14_pretrain.pth
#      label                     model            checkpoint                                                   run dir                                      catalog          norm
RUNS=("frozen_raw              dino_pretrained  $SP                                                          $STOCK                                       kaguya_tco_raw   asinh"
      "frozen_ratio            dino_pretrained  $SP                                                          $STOCK                                       kaguya_tco_ratio robust"
      "ft_ratio_ssl_proto      dino  $FT/kaguya_tco_ratio/d1-15/nunknown/ssl_proto/ep30/eval/final/teacher_checkpoint.pth $FT/kaguya_tco_ratio/d1-15/nunknown/ssl_proto/ep30 kaguya_tco_ratio robust"
      "ft_ratio_ssl            dino  $FT/kaguya_tco_ratio/d1-15/nunknown/ssl/ep30/eval/final/teacher_checkpoint.pth       $FT/kaguya_tco_ratio/d1-15/nunknown/ssl/ep30       kaguya_tco_ratio robust"
      "ft_raw_ssl_proto        dino  $FT/kaguya_tco_raw/d1-15/nunknown/ssl_proto/ep30/eval/final/teacher_checkpoint.pth   $FT/kaguya_tco_raw/d1-15/nunknown/ssl_proto/ep30   kaguya_tco_raw   asinh"
      "ft_raw_ssl              dino  $FT/kaguya_tco_raw/d1-15/nunknown/ssl/ep30/eval/final/teacher_checkpoint.pth         $FT/kaguya_tco_raw/d1-15/nunknown/ssl/ep30         kaguya_tco_raw   asinh"
      "ft_raw_ssl_supcon       dino  $FT/kaguya_tco_raw/d1-15/nunknown/ssl_supcon/ep30/eval/final/teacher_checkpoint.pth  $FT/kaguya_tco_raw/d1-15/nunknown/ssl_supcon/ep30  kaguya_tco_raw   asinh"
      "ft_raw_supcon_koleo002  dino  $FT/kaguya_tco_raw/d1-15/nunknown/ssl_supcon/ep30_koleo0.02/eval/final/teacher_checkpoint.pth  $FT/kaguya_tco_raw/d1-15/nunknown/ssl_supcon/ep30_koleo0.02  kaguya_tco_raw   asinh")
read LABEL MODEL CKPT RUN CAT NORM <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
POOL=data/evalsets/s224/${CAT}__${NORM}__pool; LAB=data/evalsets/s224/${CAT}__${NORM}
OUT=$RUN/data_comparison/geomap/${CAT}__${NORM}; mkdir -p "$OUT"
echo "=== $LABEL: $CKPT on $CAT/$NORM -> $OUT"
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_labeling --method knn \
  --pool-dat $POOL/craters.dat --pool-meta $POOL/metadata.csv \
  --labelled-dat $LAB/craters.dat --labelled-meta $LAB/metadata.csv \
  --labels-csv configs/liorr_labels.csv --checkpoint "$CKPT" --model $MODEL \
  --out-dir "$OUT" --pool-embed-cache "$OUT/pool_embeddings.npy" \
  --label "$LABEL (Kaguya ${CAT#kaguya_tco_}, 224px)"
# the same pool, coloured by the map's own k-means clusters (k=4, 10)
PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.pool_projection --dir "$OUT"
echo "EXIT=$?"
