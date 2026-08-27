"""
mae_bottleneck.py
──────────────────
Project-specific extension of the vendored Facebook MAE (src/models/mae/).

New training runs (src/train/train.py) use the plain, bottleneck-free
MaskedAutoencoderViT directly — no dependency on this file.

This module exists to keep OLD checkpoints (everything trained before the
bottleneck was removed from new runs) loadable and usable for clustering.
MaskedAutoencoderViTBottleneck reproduces the exact architecture those
checkpoints were trained with: an information bottleneck on the CLS token,
routed through a small (default 64-dim) linear layer before decoding.

load_mae() picks the right class automatically by inspecting the checkpoint's
state_dict — callers don't need to know in advance whether a given .pth is
an old bottleneck checkpoint or a new plain one.

Keeping this in a subclass (rather than editing models_mae.py directly)
keeps the vendored file a clean, unmodified copy of upstream Facebook MAE.
"""

import sys
import os

sys.path.append(os.path.abspath("src/models/mae"))   # models_mae.py needs util/ on sys.path

import torch
import torch.nn as nn

from src.models.mae.models_mae import MaskedAutoencoderViT


class MaskedAutoencoderViTBottleneck(MaskedAutoencoderViT):
    def __init__(self, *args, bottleneck_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.cls_token.shape[-1]

        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, bottleneck_dim),
        )
        self.bottleneck_expand = nn.Linear(bottleneck_dim, embed_dim)
        self.bottleneck_dim = bottleneck_dim

    def encode_cluster(self, x):
        """
        Inference-time method for k-means clustering.
        Returns bottleneck_dim-dim embedding with no masking.
        Use this instead of forward_encoder for extracting clustering embeddings.
        """
        with torch.no_grad():
            z, _, _ = self.forward_encoder(x, mask_ratio=0)
            cls = z[:, 0, :]                        # (N, embed_dim)
            return self.bottleneck(cls)              # (N, bottleneck_dim)

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        # --- Bottleneck on CLS token ---
        cls = latent[:, 0, :]                               # (N, embed_dim)
        cls_bottleneck = self.bottleneck(cls)               # (N, bottleneck_dim)
        cls_expanded = self.bottleneck_expand(cls_bottleneck)  # (N, embed_dim)

        # Replace CLS token with bottlenecked version
        # Patch tokens are unaffected — decoder still has full patch context
        latent_bottlenecked = torch.cat([
            cls_expanded.unsqueeze(1),
            latent[:, 1:, :]
        ], dim=1)

        pred = self.forward_decoder(latent_bottlenecked, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        # Return cls_bottleneck so the training loop can log its norm
        return loss, pred, mask, cls_bottleneck


def _checkpoint_has_bottleneck(state: dict) -> bool:
    return any(k.startswith("bottleneck.") or k.startswith("bottleneck_expand.")
               for k in state.keys())


def _checkpoint_bottleneck_dim(state: dict) -> int:
    # bottleneck.2 is the final Linear(256, bottleneck_dim) in the Sequential
    return state["bottleneck.2.weight"].shape[0]


def _checkpoint_patch_and_img_size(state: dict) -> tuple[int, int] | tuple[None, None]:
    """Auto-detect patch_size (patch_embed's conv kernel size) and img_size
    (from pos_embed's patch-token count, assuming a square image) directly
    from the checkpoint. cluster.py's MAE_KWARGS hardcodes this project's
    current convention (patch_size=8, img_size=128) for every "mae"
    checkpoint regardless of what it actually was trained with - fine for
    checkpoints trained under that convention, but a checkpoint trained
    before it (e.g. facebook/vit-mae-base's original patch_size=16 @ 224px
    default) then fails to load at all (pos_embed/patch_embed shape
    mismatch) rather than silently loading wrong. Returns (None, None) if
    the checkpoint doesn't have these keys at all (shouldn't happen for a
    real MAE checkpoint, but callers should fall back to their own kwargs)."""
    w = state.get("patch_embed.proj.weight")
    pe = state.get("pos_embed")
    if w is None or pe is None:
        return None, None
    patch_size = int(w.shape[-1])          # (embed_dim, in_chans, patch, patch)
    num_patches = int(pe.shape[1]) - 1     # pos_embed includes the CLS token
    img_size = round(num_patches ** 0.5) * patch_size
    return patch_size, img_size


def load_mae(checkpoint_path: str = None, device="cpu", **mae_kwargs):
    """
    Build a MAE model for this checkpoint, auto-detecting architecture.

    - checkpoint_path is None : fresh training run -> plain MaskedAutoencoderViT
      (the current default architecture; no bottleneck).
    - checkpoint has bottleneck.* weights : reconstructs
      MaskedAutoencoderViTBottleneck with the checkpoint's own bottleneck_dim,
      so old checkpoints load exactly as they were trained.
    - checkpoint has no bottleneck.* weights : plain MaskedAutoencoderViT.
    - patch_size/img_size are auto-detected from the checkpoint itself
      (see _checkpoint_patch_and_img_size), overriding whatever mae_kwargs
      passed in - the resulting model's own model.patch_embed.img_size is
      the source of truth callers should resize input images to (see
      eval/pipeline.py's embed() and eval/holdout.py's embed_holdout()),
      not a hardcoded constant.
    """
    if checkpoint_path is None:
        return MaskedAutoencoderViT(**mae_kwargs).to(device)

    state = torch.load(checkpoint_path, map_location=device)

    patch_size, img_size = _checkpoint_patch_and_img_size(state)
    if patch_size is not None:
        mae_kwargs = {**mae_kwargs, "patch_size": patch_size, "img_size": img_size}

    if _checkpoint_has_bottleneck(state):
        model = MaskedAutoencoderViTBottleneck(
            bottleneck_dim=_checkpoint_bottleneck_dim(state), **mae_kwargs
        )
    else:
        model = MaskedAutoencoderViT(**mae_kwargs)

    model.load_state_dict(state, strict=True)
    return model.to(device)


def encode_for_clustering(model, imgs):
    """
    Per-image embedding for clustering, whichever architecture `model` is.
    Bottleneck models   : encode_cluster() -> CLS token through trained bottleneck.
    Plain models        : raw CLS token (no learned compression available).
    """
    if hasattr(model, "encode_cluster"):
        return model.encode_cluster(imgs)
    with torch.no_grad():
        latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
    return latent[:, 0, :]
