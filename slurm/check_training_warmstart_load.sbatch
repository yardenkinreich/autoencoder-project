#!/bin/bash
#SBATCH --job-name=check-training-warmstart-load
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm-%x-%j.out

# Was TRAINING itself corrupted by the same bug found in load_dino_backbone
# () (used only for evaluation)? Training's own model construction goes
# through a DIFFERENT path - build_model_from_cfg() (dinov2/models/
# __init__.py), which reads block_chunks/layerscale straight from cfg.
# student, not from a separate, disconnected default dict the way
# load_dino_backbone() did. Our finetune config never overrides student.
# layerscale/student.block_chunks, so they inherit ssl_default_config.
# yaml's own values (layerscale: 1.0e-05, block_chunks: 0) - already
# matching what the stock checkpoint needs. This directly simulates
# train_dino.py's actual warm-start load (SSLMetaArch's own construction +
# _resize_pretrained_pos_embed + strict=False load) against the real stock
# checkpoint, using dinov2's own setup()/config merge exactly as
# train_dino.py does, to confirm rather than assume.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -c "
import argparse
import torch
from src.models.dinov2.dinov2.utils.config import setup
from src.models.dinov2.dinov2.train.ssl_meta_arch import SSLMetaArch

setup_args = argparse.Namespace(
    config_file='configs/dino_craters_finetune.yaml',
    output_dir='/tmp/_warmstart_check',
    opts=['train.dataset_path=Craters:root=data/processed_wac_100m_new/sigma/100/dino_wide/craters_wide.dat',
          f'student.pretrained_weights=data/raw/pretrained/dinov2_vits14_pretrain.pth'],
    seed=0,
)
cfg = setup(setup_args)
print('cfg.student.layerscale:', cfg.student.layerscale)
print('cfg.student.block_chunks:', cfg.student.block_chunks)
print('cfg.student.pretrained_weights:', cfg.student.pretrained_weights)

model = SSLMetaArch(cfg).to('cuda')

stock_sd = torch.load('data/raw/pretrained/dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=False)
stock_keys = set(stock_sd.keys())
student_backbone_sd = model.student.backbone.state_dict()
student_keys = set(student_backbone_sd.keys())

# after construction+warm-start (SSLMetaArch.__init__ already did the load
# internally) - compare against a FRESH randomly-initialized backbone of
# the same arch to see which params actually differ from random init
import copy
before_load_state = {k: v.clone() for k, v in student_backbone_sd.items()}

# reconstruct what a random-init (never loaded) version would look like,
# same seed convention won't match exactly, so instead just check how many
# student.backbone keys have a counterpart in the RAW checkpoint at all
# (missing => could not possibly have loaded, regardless of random values)
missing_from_checkpoint = student_keys - stock_keys
print()
print(f'student.backbone has {len(student_keys)} params; stock checkpoint has {len(stock_keys)} params')
print(f'student.backbone keys with NO matching key in the stock checkpoint at all: {len(missing_from_checkpoint)}')
print('sample:', sorted(missing_from_checkpoint)[:10] if missing_from_checkpoint else '(none)')

# direct value check: does student.backbone's pos_embed/cls_token/block-0
# weights match the (possibly pos_embed-resized) stock values, i.e. did the
# warm-start load actually run and take effect - compare against a SEPARATE
# freshly-loaded reference model built the same way DIET-CP verified against
from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.train.ssl_meta_arch import _resize_pretrained_pos_embed
ref_sd = {k: v.clone() for k, v in stock_sd.items()}
ref_model = vit_small(img_size=224, patch_size=14, in_chans=3, block_chunks=0, init_values=1.0)
_resize_pretrained_pos_embed(ref_sd, ref_model.pos_embed)
ref_model.load_state_dict(ref_sd, strict=True)

for k in ['cls_token', 'patch_embed.proj.weight', 'blocks.0.attn.qkv.weight', 'blocks.11.mlp.fc2.weight', 'blocks.0.ls1.gamma']:
    a = student_backbone_sd.get(k)
    b = dict(ref_model.state_dict()).get(k)
    if a is None or b is None:
        print(f'{k}: MISSING (a is None: {a is None}, b is None: {b is None})')
        continue
    same = torch.allclose(a.cpu(), b.cpu(), atol=1e-4)
    print(f'{k}: matches known-correctly-loaded reference = {same}')
"
echo "EXIT=$?"
