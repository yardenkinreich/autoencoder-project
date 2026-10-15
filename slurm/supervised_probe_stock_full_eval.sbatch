#!/bin/bash
#SBATCH --job-name=supervised-probe-full-eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm-%x-%j.out

# Full dashboard-integrated eval for the supervised probe (item 1 of the
# "ways forward" list) - writes into the frozen stock checkpoint's own
# eval_results/ dir and appends to logs/eval_history.csv, same as any
# other eval.evaluate run, tagged "supervised_probe" so it's identifiable
# in the dashboard's run picker.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src pixi run -e default python -m eval.supervised_probe \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained \
  --full-eval \
  --tag supervised_probe
echo "EXIT=$?"
