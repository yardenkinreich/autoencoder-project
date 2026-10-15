#!/bin/bash
#SBATCH --job-name=diagnose-eval-dino-frozen1
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CKPT="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_frozen1/eval/final/teacher_checkpoint.pth"
DIAG_OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics_dino_frozen1.jsonl
rm -f "$DIAG_OUT"

for set_name in julie new_set_final_test; do
  PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
    --checkpoint "$CKPT" --autoencoder-model dino --label "dino_frozen1" --set "$set_name" --out "$DIAG_OUT"
done

PYTHONPATH=src pixi run -e default python -m eval.evaluate \
  --checkpoint "$CKPT" --autoencoder-model dino \
  --config configs/eval_suite.yaml --tag dino_frozen1_noproto_ep5
echo "EXIT=$?"
