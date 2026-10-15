"""
attention_maps.py — CLS-token attention-map diagnostic (the "Domain-Specific
Vision Foundation Model for Mars" paper's check: inspect whether attention
in the final transformer block stays distributed along the object of
interest - crater rims, for us - versus collapsing onto a few arbitrary
patches. A faster, visual read on shortcut learning than a PCA/confound
analysis, and independent of it - the two diagnostics can disagree, and
disagreeing would itself be informative.

Why this needs custom code at all: DINOv2's own Attention/MemEffAttention
(src/models/dinov2/dinov2/layers/attention.py) route through fused kernels
(xFormers' memory_efficient_attention, or torch's own
scaled_dot_product_attention) that never materialize the full (N, N)
attention matrix in memory - that's precisely what makes them fast, but it
also means there is no way to read attention weights back out of a normal
forward pass. The original DINO repo had a get_last_selfattention() method
for exactly this reason; this DINOv2 fork dropped it when it moved to fused
kernels. This module recomputes ONLY the last block's attention explicitly
(plain matmul + softmax, bypassing the fused kernel) - a duplicate,
side-channel computation purely for visualization, using the exact same
norm1/qkv/num_heads/scale the real forward pass uses so it stays faithful
to what the model actually computes.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _flat_blocks(model) -> list:
    """model.blocks is either a flat nn.ModuleList of Block objects
    (block_chunks=0, e.g. the stock checkpoint's STOCK_DINO_VIT_KWARGS) or a
    ModuleList of BlockChunk objects (block_chunks>0, the default vit_small()
    uses whenever it isn't overridden - true for every checkpoint we've
    trained, see dino_backbone.DINO_VIT_KWARGS). BlockChunk itself subclasses
    nn.ModuleList and is padded with nn.Identity() placeholders to keep block
    indices consistent under chunking (vision_transformer.py's own
    construction code) - so flattening means iterating each chunk's own
    children and dropping the Identity placeholders, not just taking
    model.blocks[-1] directly (which would return an entire BlockChunk, not
    a single Block, whenever chunking is in play)."""
    if getattr(model, "chunked_blocks", False):
        return [b for chunk in model.blocks for b in chunk if not isinstance(b, torch.nn.Identity)]
    return list(model.blocks)


@torch.no_grad()
def last_block_cls_attention(model, x: torch.Tensor) -> torch.Tensor:
    """
    model: a loaded DINO backbone (dino_backbone.load_dino_backbone's output).
    x: (B, C, H, W) input batch, already at the model's expected resolution/
       channel count.

    Returns (B, num_heads, num_patches): how much the CLS token (as QUERY)
    attends to each PATCH (as KEY) in the LAST transformer block. Averaging
    over heads gives the single attention-map-per-image convention DINO's
    own well-known visualizations use.
    """
    blocks = _flat_blocks(model)
    # prepare_tokens_with_masks does patch_embed + concat CLS (+ register
    # tokens) + add pos_embed - the same prep every block's input goes
    # through in the model's own forward_features() - reused directly so
    # this stays in sync with whatever convention THIS checkpoint's
    # architecture uses (patch14/in_chans3 finetune vs patch8/in_chans1
    # from-scratch, register tokens or not) rather than reimplementing it
    # and risking a mismatch.
    tokens = model.prepare_tokens_with_masks(x)
    for blk in blocks[:-1]:
        tokens = blk(tokens)

    last_block = blocks[-1]
    normed = last_block.norm1(tokens)
    attn = last_block.attn
    B, N, C = normed.shape
    qkv = attn.qkv(normed).reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
    q, k, _v = qkv.unbind(2)
    q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, heads, N, head_dim)
    attn_weights = (q @ k.transpose(-2, -1)) * attn.scale  # (B, heads, N, N)
    attn_weights = attn_weights.softmax(dim=-1)

    num_prefix = 1 + getattr(model, "num_register_tokens", 0)  # CLS (+ any register tokens)
    cls_to_patch = attn_weights[:, :, 0, num_prefix:]  # CLS as query, patches only as keys
    return cls_to_patch


def attention_map_grid(cls_to_patch: torch.Tensor, grid_size: int) -> np.ndarray:
    """(B, heads, num_patches), averaged over heads -> (B, grid_size,
    grid_size) 2D map, ready to upsample/overlay on the original image."""
    avg = cls_to_patch.mean(dim=1)  # (B, num_patches)
    return avg.reshape(-1, grid_size, grid_size).cpu().numpy()


def save_attention_maps_for_df(model, df, imgs_dir: str, out_dir: str,
                               n_per_class: int = 2, seed: int = 0) -> list[str]:
    """Save attention-map overlays for a handful of sample craters from an
    already-built predictions dataframe (crater_id/true_label/filename - the
    same schema evaluate.py's evaluate_labeled_set() already has after
    P.build_predictions()). Reuses visualize.py's own _load_crop_for_display
    so this works for either labeled-set shape (julie's .png crops or a
    memmap-based set's .npy display crops) without a third copy of that
    format-detection logic.

    model must already be loaded (e.g. via dino_backbone.load_dino_backbone) -
    unlike main()'s standalone CLI path, this does NOT load its own model,
    since evaluate.py already has one loaded for embedding and reloading it
    again per labeled set would be wasteful. DINO-family checkpoints only -
    attention maps aren't meaningful for MAE/CAE's different architectures,
    and this hasn't been tried against them.

    Returns the list of written paths (empty if imgs_dir/filenames aren't
    found - e.g. a set with no display crops saved)."""
    from eval.visualize import _load_crop_for_display

    patch_size = model.patch_embed.proj.kernel_size[0]
    target_size = model.patch_embed.img_size[0] if hasattr(model.patch_embed, "img_size") else 224
    in_chans = model.patch_embed.proj.in_channels
    grid_size = target_size // patch_size
    device = next(model.parameters()).device

    sample_rows = (df.groupby("true_label", group_keys=False)
                   .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=seed)))

    written = []
    for _, row in sample_rows.iterrows():
        img_path = os.path.join(imgs_dir, row["filename"])
        if not os.path.exists(img_path):
            continue
        raw = _load_crop_for_display(img_path)  # (H, W), model's own [0,1]-ish scale, native resolution

        model_input = torch.from_numpy(raw).float()[None, None]  # (1,1,H,W)
        model_input = F.interpolate(model_input, size=(target_size, target_size),
                                    mode="bicubic", align_corners=False)
        if in_chans > 1:
            model_input = model_input.repeat(1, in_chans, 1, 1)
        x = model_input.to(device)

        cls_to_patch = last_block_cls_attention(model, x)
        grid = attention_map_grid(cls_to_patch, grid_size)[0]
        out_path = os.path.join(out_dir, f"{row['crater_id']}_class{row['true_label']}.png")
        plot_attention_overlay(raw, grid, out_path,
                               title=f"{row['crater_id']} (class {row['true_label']})")
        written.append(out_path)
    return written


def main():
    """CLI driver: run the diagnostic against Julie's PNG set (the same
    labels_from_png_dir/_png_to_global scaling pipeline.embed() uses, reused
    directly here rather than duplicated) for a handful of sample craters,
    one per true class where possible so the overlays span the degradation
    scale rather than clustering around one state by chance."""
    import argparse
    import pandas as pd
    from PIL import Image
    from models.dino_backbone import load_dino_backbone, load_stock_pretrained_dino, STOCK_DINO_FORWARD_SIZE
    from eval import pipeline as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--autoencoder-model", required=True, choices=["dino", "dino_pretrained"])
    ap.add_argument("--imgs-dir", default="data/raw/craters_for_danny")
    ap.add_argument("--scaling-json", default="configs/global_scaling.json")
    ap.add_argument("--png-max", type=float, default=255.0)
    ap.add_argument("--n-per-class", type=int, default=2)
    ap.add_argument("--label", default=None, help="free-text label for filenames/titles")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    label = args.label or args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.autoencoder_model == "dino_pretrained":
        model = load_stock_pretrained_dino(args.checkpoint, device=device)
        target_size = STOCK_DINO_FORWARD_SIZE
        patch_size = model.patch_embed.patch_size[0] if isinstance(model.patch_embed.patch_size, tuple) else model.patch_embed.patch_size
    else:
        model = load_dino_backbone(args.checkpoint, device=device)
        patch_size = model.patch_embed.proj.kernel_size[0]
        target_size = model.patch_embed.img_size[0] if hasattr(model.patch_embed, "img_size") else 224
    in_chans = model.patch_embed.proj.in_channels
    grid_size = target_size // patch_size

    df = P.labels_from_png_dir(args.imgs_dir)
    rng = np.random.default_rng(args.seed)
    sample_rows = (df.groupby("true_label", group_keys=False)
                   .apply(lambda g: g.sample(n=min(args.n_per_class, len(g)),
                                             random_state=args.seed)))

    lo, hi, png_max = P._resolve_scaling({"scaling_json": args.scaling_json, "png_max": args.png_max})

    for _, row in sample_rows.iterrows():
        img = Image.open(os.path.join(args.imgs_dir, row["filename"])).convert("L").resize(
            (target_size, target_size))
        arr = np.array(img, dtype=np.float32)
        scaled = P._png_to_global(arr, lo, hi, png_max)
        x = torch.from_numpy(scaled).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
        if in_chans > 1:
            x = x.repeat(1, in_chans, 1, 1)

        cls_to_patch = last_block_cls_attention(model, x)
        grid = attention_map_grid(cls_to_patch, grid_size)[0]
        out_path = os.path.join(args.out_dir, f"{label}_{row['crater_id']}_class{row['true_label']}.png")
        plot_attention_overlay(scaled, grid, out_path,
                               title=f"{label} — {row['crater_id']} (class {row['true_label']})")
        print(f"Saved → {out_path}")


def plot_attention_overlay(img: np.ndarray, attn_grid: np.ndarray, out_path: str,
                           title: str = "") -> None:
    """img: (H, W) grayscale crop in [0,1]. attn_grid: (grid_size, grid_size)
    raw attention map (already averaged over heads, NOT yet resized) - upsampled
    to img's resolution here (bicubic, matching how this project resizes
    everywhere else) and overlaid as a semi-transparent heatmap alongside the
    plain image, side by side, so the reader can compare directly rather than
    guess what the heatmap is occluding."""
    H, W = img.shape
    attn_t = torch.from_numpy(attn_grid)[None, None].float()
    attn_up = F.interpolate(attn_t, size=(H, W), mode="bicubic", align_corners=False)[0, 0].numpy()
    attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("crop")
    axes[0].axis("off")
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(attn_up, cmap="inferno", alpha=0.55)
    axes[1].set_title("CLS attention (last block)")
    axes[1].axis("off")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
