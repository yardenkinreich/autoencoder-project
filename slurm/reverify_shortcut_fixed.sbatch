#!/bin/bash
#SBATCH --job-name=reverify-shortcut-fixed
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Re-run the shortcut diagnostic on the key checkpoints from this entire
# session, now that load_dino_backbone()'s block_chunks/init_values
# auto-detection bug is fixed (dino_backbone.py) - every prior result
# (this whole session AND likely this project's entire DINO-finetuning
# history before it) was computed against a ~96%-random backbone. This is
# the first time any of these checkpoints have actually been evaluated on
# their real trained weights.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

DIAG_OUT=/tmp/claude-1001/-home-kinreich-autoencoder-project/f40296c3-b9d1-4aab-8ea1-ba0382d24406/scratchpad/shortcut_diagnostics_POSTFIX.jsonl
rm -f "$DIAG_OUT"

declare -A CKPTS
CKPTS[ep10_noproto_orig]="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10_noproto/eval/final/teacher_checkpoint.pth"
CKPTS[ep5_cropfix]="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_cropfix/eval/final/teacher_checkpoint.pth"
CKPTS[ep5_sinkhorn]="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_sinkhorn/eval/final/teacher_checkpoint.pth"
CKPTS[diet_cp_augfix]="logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_augfix/eval/final/teacher_checkpoint.pth"
CKPTS[diet_cp_frozen]="logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_frozen/eval/final/teacher_checkpoint.pth"

for label in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$label]}"
  for set_name in julie new_set_final_test; do
    PYTHONPATH=src pixi run -e default python -m eval.diagnose_shortcut \
      --checkpoint "$ckpt" --autoencoder-model dino --label "$label" --set "$set_name" --out "$DIAG_OUT"
  done
done
echo "EXIT=$?"
