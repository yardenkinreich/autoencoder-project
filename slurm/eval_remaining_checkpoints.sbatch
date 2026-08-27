#!/bin/bash
#SBATCH --job-name=eval-remaining
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Re-evaluates every checkpoint already tracked in logs/eval_history.csv
# that the proto-comparison job (stock/noproto/proto DINO) didn't cover, so
# they all get the new unsupervised over-clustering plots (evaluate.py's
# evaluate_labeled_set(), see visualize.py's plot_unsupervised_cluster_
# dots/_images) too. Not every checkpoint under logs/ (28 total, including
# an old facebook_vit-mae-large freeze-sweep and CAE latent-dim sweep that
# were never part of this eval harness) - just the 5 already in the
# tracked comparison set.

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
    --config configs/eval_suite.yaml --tag unsupervised_cluster_backfill
  echo "EXIT=$? ($1)"
}

run_eval "logs/mae/scratch/none/na/wac100m_sigma100_noband_global/d3-10/n25000/mask0.75/ep500/20260810-221537/models/autoencoder.pth" mae
run_eval "src/models/mae/pretrain_mae_vit_base_full.pth" mae_pretrained
run_eval "logs/mae/scratch/none/na/wac100m_sigma100/d3-10/n50000/mask0.75/ep150/20260505-173914/models/autoencoder.pth" mae
run_eval "logs/dino/scratch/none/na/wac100m_sigma100/d3-10/nunknown/na/ep2/20260810-193059/eval/final/teacher_checkpoint.pth" dino
run_eval "logs/mae/finetune/mae_pretrain_mae_vit_base_full/unknown/wac100m_sigma100/d3-10/n50000/mask0.75/ep50/20260426-155516/models/autoencoder.pth" mae

echo "ALL_DONE"
