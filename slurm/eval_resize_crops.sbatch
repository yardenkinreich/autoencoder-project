#!/bin/bash
#SBATCH --job-name=eval-resize-crops
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Re-evaluates every currently-tracked checkpoint (except stock MAE, still
# blocked on the unresolved NaN-embedding bug) so all of them get the
# corrected unsupervised-cluster crop sizes: 0.18 zoom for new_set/
# new_set_final_test (already applied to the 3 DINO checkpoints in job 141,
# but job 140's 4 checkpoints predate that fix and still have the old 0.35),
# and a NEW julie-only override (0.28, configs/eval_suite.yaml's
# unsupervised_crop_zoom) since 0.18 read too small on julie's smaller
# (150-crater) point cloud.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_eval() {
  echo "=== $1 ==="
  PYTHONPATH=src pixi run -e default python -m eval.evaluate \
    --checkpoint "$1" --autoencoder-model "$2" \
    --config configs/eval_suite.yaml --tag "$3"
  echo "EXIT=$? ($1)"
}

run_eval "data/raw/pretrained/dinov2_vits14_pretrain.pth" dino_pretrained off_the_shelf
run_eval "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10_noproto/eval/final/teacher_checkpoint.pth" dino dino_comparison_noproto_finetune
run_eval "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10/eval/final/teacher_checkpoint.pth" dino dino_comparison_proto_finetune
run_eval "logs/mae/scratch/none/na/wac100m_sigma100_noband_global/d3-10/n25000/mask0.75/ep500/20260810-221537/models/autoencoder.pth" mae unsupervised_cluster_backfill
run_eval "logs/mae/scratch/none/na/wac100m_sigma100/d3-10/n50000/mask0.75/ep150/20260505-173914/models/autoencoder.pth" mae unsupervised_cluster_backfill
run_eval "logs/dino/scratch/none/na/wac100m_sigma100/d3-10/nunknown/na/ep2/20260810-193059/eval/final/teacher_checkpoint.pth" dino unsupervised_cluster_backfill
run_eval "logs/mae/finetune/mae_pretrain_mae_vit_base_full/unknown/wac100m_sigma100/d3-10/n50000/mask0.75/ep50/20260426-155516/models/autoencoder.pth" mae unsupervised_cluster_backfill

echo "ALL_DONE"
