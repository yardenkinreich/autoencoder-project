#!/bin/bash
#SBATCH --job-name=check-pos-embed-paths
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm-%x-%j.out

# Direct test: does DINOv2's dynamic interpolate_pos_encoding() (stock's
# loading path - img_size=518, native pos_embed, interpolated fresh at
# every forward call) produce the SAME output as _resize_pretrained_
# pos_embed()'s one-time precomputed interpolation (every trained/frozen
# checkpoint's loading path - img_size=224, pos_embed baked in at load
# time)? Triggered by finding that "diet_cp_frozen" (backbone NEVER
# unfrozen, confirmed byte-identical to stock on every non-pos_embed
# weight) still shows the same brightness collapse stock doesn't - the
# ONLY difference between these two checkpoints is which pos_embed path
# was used, so if that path materially changes the CLS embedding, it
# would explain the collapse independent of every training-recipe fix
# tried so far.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -c "
import torch
from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.train.ssl_meta_arch import _resize_pretrained_pos_embed

device = 'cuda'
stock_sd = torch.load('data/raw/pretrained/dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=False)

model_a = vit_small(img_size=518, patch_size=14, in_chans=3, block_chunks=0, init_values=1.0)
model_a.load_state_dict(stock_sd, strict=True)
model_a.to(device).eval()

sd_b = {k: v.clone() for k, v in stock_sd.items()}
model_b = vit_small(img_size=224, patch_size=14, in_chans=3, block_chunks=0, init_values=1.0)
_resize_pretrained_pos_embed(sd_b, model_b.pos_embed)
model_b.load_state_dict(sd_b, strict=True)
model_b.to(device).eval()

print('model_a pos_embed shape (stored):', model_a.pos_embed.shape)
print('model_b pos_embed shape (stored):', model_b.pos_embed.shape)

torch.manual_seed(0)
x = torch.rand(8, 3, 224, 224, device=device)

with torch.no_grad():
    out_a = model_a(x, is_training=True)
    out_b = model_b(x, is_training=True)

cls_a = out_a['x_norm_clstoken']
cls_b = out_b['x_norm_clstoken']
print('CLS embeddings identical (atol=1e-5):', torch.allclose(cls_a, cls_b, atol=1e-5))
print('max abs diff:', (cls_a - cls_b).abs().max().item())
cos = torch.nn.functional.cosine_similarity(cls_a, cls_b, dim=1)
print('per-sample cosine similarity:', cos.tolist())
"
echo "EXIT=$?"
