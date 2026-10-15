#!/bin/bash
#SBATCH --job-name=check-load-dino-backbone-keys
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm-%x-%j.out

# Does load_dino_backbone() (the function EVERY diagnose_shortcut.py /
# eval.evaluate run has used for every "dino"-family checkpoint this whole
# session) actually load the transformer blocks/LayerScale for a checkpoint
# saved with block_chunks=0, init_values=1.0 (diet_cp.py's and train_dino.
# py's own convention)? DINO_VIT_KWARGS never overrides block_chunks/
# init_values and _detect_dino_arch() doesn't detect them either, so
# vit_small() would fall back to DinoVisionTransformer's own defaults
# (block_chunks=1, init_values=None) - a different block-key naming
# convention and no LayerScale module at all. strict=False inside
# load_pretrained_weights would silently drop any mismatched keys instead
# of raising. This checks exactly that, directly, on a real checkpoint.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -c "
import torch
from src.models.dino_backbone import load_dino_backbone, DINO_VIT_KWARGS, _detect_dino_arch
from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.utils.utils import load_pretrained_weights

ckpt = 'logs/dino/diet-cp/none/na/diet_cp_random_sample/drange_unknown/n1000/na/ep150_frozen/eval/final/teacher_checkpoint.pth'

detected = _detect_dino_arch(ckpt)
print('detected arch:', detected)
kwargs = {**DINO_VIT_KWARGS, **detected}
print('final kwargs load_dino_backbone() builds vit_small() with:', kwargs)

model = vit_small(**kwargs)
print()
print('constructed model.blocks type:', type(model.blocks))
print('constructed model.blocks[0] type:', type(model.blocks[0]))
print('constructed model.blocks.0.ls1 type:', type(model.blocks[0].ls1) if hasattr(model.blocks[0], 'ls1') else 'N/A (block_chunks>0, indexing differs)')

# Now actually load and report missing/unexpected keys ourselves (not
# relying on load_pretrained_weights' own internal logging, which may be
# silent) - compare model.state_dict() keys before/after vs checkpoint keys
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
print(f'MISSING from checkpoint (model params that will stay at random init): {len(missing)}')
print('  sample:', sorted(missing)[:10])
print(f'UNEXPECTED in checkpoint (silently dropped, never loaded): {len(unexpected)}')
print('  sample:', sorted(unexpected)[:10])

load_pretrained_weights(model, ckpt, 'teacher')
after = model.state_dict()
changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
unchanged_still_random = sum(1 for k in before if torch.equal(before[k], after[k]))
print()
print(f'params that CHANGED after load_pretrained_weights (real trained values loaded): {changed}/{len(before)}')
print(f'params that stayed at their PRE-LOAD (random init) value: {unchanged_still_random}/{len(before)}')
for k in list(before.keys())[:5]:
    print(f'  {k}: changed={not torch.equal(before[k], after[k])}')
for k in [k for k in before if \"blocks.0.\" in k][:5]:
    print(f'  {k}: changed={not torch.equal(before[k], after[k])}')
"
echo "EXIT=$?"
