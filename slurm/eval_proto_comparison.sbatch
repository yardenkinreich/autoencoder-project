#!/bin/bash
#SBATCH --job-name=eval-proto-comparison
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Evaluates stock frozen DINOv2 vs. the plain finetune vs. the prototypical-
# loss finetune, all through the same eval_suite.yaml suite, so all three
# rows land in logs/eval_history.csv comparably. Run via sbatch (not a bare
# background process) because a direct, unscheduled GPU process gets SIGKILL
# on this cluster - confirmed directly: eval.evaluate died the same way
# train_dino.py did earlier before that got moved to sbatch too.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID   # see train_dino_finetune.sbatch's comment

# Belt-and-suspenders alongside head.py's KNeighborsClassifier(n_jobs=1) fix -
# job 136 hit a garbage-index IndexError out of sklearn's compiled k-NN
# backend on new_set_final_test, the signature of a BLAS/joblib threading
# race under a multi-CPU SLURM allocation. n_jobs=1 controls joblib's own
# parallelism; these cover the BLAS layer underneath numpy/sklearn too.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== STOCK ==="
PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained \
  --config configs/eval_suite.yaml \
  --tag off_the_shelf
echo "STOCK_EXIT=$?"

echo "=== NOPROTO FINETUNE ==="
PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10_noproto/eval/final/teacher_checkpoint.pth" \
  --autoencoder-model dino \
  --config configs/eval_suite.yaml \
  --tag dino_comparison_noproto_finetune
echo "NOPROTO_EXIT=$?"

echo "=== PROTO FINETUNE ==="
PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint "logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10/eval/final/teacher_checkpoint.pth" \
  --autoencoder-model dino \
  --config configs/eval_suite.yaml \
  --tag dino_comparison_proto_finetune
echo "PROTO_EXIT=$?"

echo "ALL_DONE"
