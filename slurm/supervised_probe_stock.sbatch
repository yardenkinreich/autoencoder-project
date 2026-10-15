#!/bin/bash
#SBATCH --job-name=supervised-probe-stock
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm-%x-%j.out

# Item 1 of the "ways forward" list: a small supervised MLP probe trained
# directly on the true degradation labels, on top of frozen stock DINOv2
# embeddings (no backbone training at all). Isolates "does frozen DINOv2's
# representation contain the class signal" from "does Hungarian-matched
# k-means happen to recover it" - see supervised_probe.py's module
# docstring. train_pool (321) trains the probe, final_test (321, disjoint,
# never touched) is the one evaluation.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src pixi run -e default python -m eval.supervised_probe \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained \
  --label stock \
  --out /tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/supervised_probe_results.jsonl
echo "EXIT=$?"
