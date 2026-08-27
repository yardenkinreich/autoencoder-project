"""
mae_pretrained.py
──────────────────
Load the official Meta ImageNet-pretrained MAE checkpoint for frozen feature
extraction — parallel to src/models/dino_backbone.py's
load_stock_pretrained_dino(). Does NOT touch src/train/train.py or any
crater-specific MAE training code.

Reuses the vendored repo's OWN checkpoint-loading recipe from
main_linprobe.py (models_vit.vit_base_patch16 + util.pos_embed's
interpolate_pos_embed) rather than hand-rolling it — that script's job is
"load an MAE-pretrained checkpoint into a plain ViT for feature extraction,
then train a linear head on top." We only want the first half (frozen
features), so we stop right after load_state_dict and never touch the head
or run any training loop.

One thing that IS load-bearing, not just linprobe-specific plumbing:
main_linprobe.py inserts a non-affine BatchNorm1d between the frozen CLS
token and the (trainable) head - "hack: revise model's head with BN"
(main_linprobe.py:222) - meaning the official recipe never hands the raw
CLS token to anything downstream, it standardizes it first. KMeans
clustering is similarly scale-sensitive, so we apply the same
standardization here before returning embeddings, rather than assuming the
raw CLS token is already what's meant to be used.

Checkpoint: src/models/mae/pretrain_mae_vit_base_full.pth (official Meta
ImageNet-1k MAE pretrain, ViT-B/16 - same embed_dim/depth/num_heads our own
crater-MAE already uses in train.py, so the comparison is architecture-
capacity-equal; only patch_size/in_chans/img_size differ, same as the
DINOv2 comparison).
"""
import torch
import torch.nn.functional as F

from src.models.mae.models_vit import vit_base_patch16
from src.models.mae.util.pos_embed import interpolate_pos_embed

STOCK_MAE_CHECKPOINT = "src/models/mae/pretrain_mae_vit_base_full.pth"
STOCK_MAE_FORWARD_SIZE = 224   # matches the checkpoint's native resolution exactly


def load_stock_pretrained_mae(checkpoint_path: str = STOCK_MAE_CHECKPOINT, device="cpu"):
    model = vit_base_patch16(num_classes=0, global_pool=False,
                             img_size=STOCK_MAE_FORWARD_SIZE, in_chans=3)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict.get(k, torch.empty(0)).shape:
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)   # no-op here (224 matches exactly)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # only the (unused, num_classes=0 -> nonexistent) head is allowed missing
    assert set(msg.missing_keys) <= {"head.weight", "head.bias"}, msg.missing_keys

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_stock_pretrained_mae(model, imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (B, 1, H, W). Channel-replicated to 3 and resized to
    STOCK_MAE_FORWARD_SIZE. forward_features() returns the final-norm CLS
    token (global_pool=False) - same CLS-token convention as DINOv2 and our
    own crater-MAE's encode_for_clustering(). Standardized per-dimension
    (mean 0, std 1 across the batch) afterward, matching main_linprobe.py's
    non-affine BatchNorm1d before the head - see module docstring.
    """
    x = F.interpolate(imgs, size=(STOCK_MAE_FORWARD_SIZE, STOCK_MAE_FORWARD_SIZE),
                      mode="bicubic", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    with torch.no_grad():
        feats = model.forward_features(x)
        feats = (feats - feats.mean(dim=0, keepdim=True)) / (feats.std(dim=0, keepdim=True) + 1e-6)
    return feats
