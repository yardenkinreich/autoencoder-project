#!/bin/bash
#SBATCH --job-name=verify-dino-backbone-fix
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -c "
import torch
from src.models.dino_backbone import load_dino_backbone, DINO_VIT_KWARGS, _detect_dino_arch
from src.models.dinov2.dinov2.utils.utils import load_pretrained_weights

ckpts = {
    'diet_cp_frozen': 'logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_frozen/eval/final/teacher_checkpoint.pth',
    'dino_cropfix': 'logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_cropfix/eval/final/teacher_checkpoint.pth',
    'dino_sinkhorn': 'logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_sinkhorn/eval/final/teacher_checkpoint.pth',
}

for name, ckpt in ckpts.items():
    print(f'=== {name} ===')
    detected = _detect_dino_arch(ckpt)
    print('detected:', detected)
    model = load_dino_backbone(ckpt)
    sd_raw = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd_raw['teacher'] if 'teacher' in sd_raw else sd_raw
    sd = {k.replace('module.', '').replace('backbone.', ''): v for k, v in sd.items()}
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    missing = model_keys - ckpt_keys
    print(f'model params: {len(model_keys)}, checkpoint params: {len(ckpt_keys)}')
    print(f'MISSING (would stay random): {len(missing)}  {sorted(missing) if missing else \"(none)\"}')
    print()
"
echo "EXIT=$?"
