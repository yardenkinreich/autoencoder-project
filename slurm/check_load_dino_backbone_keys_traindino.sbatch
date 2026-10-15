#!/bin/bash
#SBATCH --job-name=check-load-dino-backbone-keys-traindino
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same check as check_load_dino_backbone_keys.sbatch, but on a
# train_dino.py-produced checkpoint (SSLMetaArch, NOT diet_cp.py) - does
# THIS family of checkpoints suffer the same load_dino_backbone() bug, or
# does train_dino.py's own construction happen to match load_dino_backbone
# ()'s unoverridden defaults (block_chunks=1) closely enough that only
# init_values differs, or neither?

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -c "
import torch
from src.models.dino_backbone import load_dino_backbone, DINO_VIT_KWARGS, _detect_dino_arch
from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.utils.utils import load_pretrained_weights

ckpt = 'logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep5_cropfix/eval/final/teacher_checkpoint.pth'

detected = _detect_dino_arch(ckpt)
print('detected arch:', detected)
kwargs = {**DINO_VIT_KWARGS, **detected}
print('final kwargs load_dino_backbone() builds vit_small() with:', kwargs)

model = vit_small(**kwargs)

before = {k: v.clone() for k, v in model.state_dict().items()}
sd_raw = torch.load(ckpt, map_location='cpu', weights_only=False)
sd = sd_raw['teacher'] if 'teacher' in sd_raw else sd_raw
sd = {k.replace('module.', '').replace('backbone.', ''): v for k, v in sd.items()}
model_keys = set(model.state_dict().keys())
ckpt_keys = set(sd.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys
print()
print(f'model has {len(model_keys)} params, checkpoint has {len(ckpt_keys)} params')
print(f'MISSING (stay at random init): {len(missing)}  sample: {sorted(missing)[:6]}')
print(f'UNEXPECTED (silently dropped): {len(unexpected)}  sample: {sorted(unexpected)[:6]}')

load_pretrained_weights(model, ckpt, 'teacher')
after = model.state_dict()
changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
print()
print(f'params that CHANGED after load (real trained values loaded): {changed}/{len(before)}')
for k in [k for k in before if 'blocks.0' in k or k in ('cls_token','pos_embed')][:8]:
    print(f'  {k}: changed={not torch.equal(before[k], after[k])}')
"
echo "EXIT=$?"
