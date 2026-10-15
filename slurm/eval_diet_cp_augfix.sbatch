#!/bin/bash
#SBATCH --job-name=eval-diet-cp-augfix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

# Full eval.evaluate run for ep150_augfix (diet_cp.py's corrected
# build_augmentation() - paper's Appendix B gating/magnitude + the
# GaussianBlur-probability-inversion fix), same protocol as
# diagnose_and_eval_diet_cp.sbatch's eval section (job 191's eval) so both
# show up in the dashboard for direct comparison. DIET-CP never trains on
# new_test_set (see diet_cp.py's module docstring), so evaluating against
# the FULL new_set here is safe, same as the original run.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint "logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_augfix/eval/final/teacher_checkpoint.pth" \
  --autoencoder-model dino \
  --config configs/eval_suite.yaml \
  --tag diet_cp_augfix_ep150_n1000
echo "EXIT=$?"
