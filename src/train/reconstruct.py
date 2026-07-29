"""
reconstruct.py
──────────────
Visualise MAE / CAE reconstructions from a trained checkpoint.

Snakemake call (example):
    python src/train/reconstruct.py \
        --autoencoder_model mae \
        --input  data/processed_wac_100m_new/sigma/100/craters_aug.dat \
        --model  logs/mae/.../models/autoencoder.pth \
        --device cpu \
        --file_out logs/mae/.../models/reconstructions.png \
        --num_images 8 \
        --mask_ratio 0.75
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src/models/mae"))   # needed for util.pos_embed
from src.models.autoencoder import ConvAutoencoder
from src.models.mae.models_mae import MaskedAutoencoderViT
from src.models.mae_bottleneck import load_mae


# ── architecture (must mirror train.py exactly) ───────────────────────────────
MAE_KWARGS = dict(
    img_size=128,
    patch_size=8,
    in_chans=1,
    norm_pix_loss=True,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4.0,
)


# ── memmap loader (mirrors train.py) ─────────────────────────────────────────
def load_memmap(path: str, num_channels: int) -> np.memmap:
    file_size    = os.path.getsize(path)
    total_floats = file_size // 4
    for size in [128]:
        if total_floats % (num_channels * size * size) == 0:
            N = total_floats // (num_channels * size * size)
            print(f"  Memmap shape: ({N}, {num_channels}, {size}, {size})")
            return np.memmap(path, dtype=np.float32, mode="r",
                             shape=(N, num_channels, size, size))
    raise ValueError(
        f"Cannot infer shape from {path} "
        f"({total_floats} floats, {num_channels} channel(s))."
    )


# ── MAE helpers ───────────────────────────────────────────────────────────────

def unpatchify(model: MaskedAutoencoderViT,
               pred: torch.Tensor) -> torch.Tensor:
    """(B, n_patches, patch²·C) → (B, C, H, W)  using the model's own method."""
    return model.unpatchify(pred)          # built into MaskedAutoencoderViT


def make_masked_image(model: MaskedAutoencoderViT,
                      imgs: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """
    Replace masked patches with 0.5 (mid-grey) so the viewer can see
    which regions the model had to reconstruct.
    mask: (B, n_patches)  1 = masked, 0 = visible
    """
    B, C, H, W = imgs.shape
    p          = model.patch_embed.patch_size[0]   # patch size
    h = w      = H // p                            # patches per side

    # expand mask to pixel space: (B, n_patches) → (B, C, H, W)
    mask_pixel = mask.reshape(B, h, w)              # (B, h, w)
    mask_pixel = mask_pixel.unsqueeze(1)            # (B, 1, h, w)
    mask_pixel = mask_pixel.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
    mask_pixel = mask_pixel.expand_as(imgs)         # (B, C, H, W)

    masked = imgs.clone()
    masked[mask_pixel == 1] = 0.5
    return masked


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_reconstructions(
    originals:      np.ndarray,   # (N, H, W)  float32 [0,1]
    masked_inputs:  np.ndarray,   # (N, H, W)
    reconstructions: np.ndarray,  # (N, H, W)
    save_path: str,
    model_type: str,
):
    N    = len(originals)
    rows = 3 if model_type == "mae" else 2
    labels = (["Original", "Masked input", "Reconstruction"]
              if model_type == "mae"
              else ["Original", "Reconstruction"])

    fig, axes = plt.subplots(rows, N, figsize=(2.2 * N, 2.5 * rows),
                             facecolor="#0d1117")

    grids = [originals, masked_inputs, reconstructions] if rows == 3 \
            else [originals, reconstructions]

    for r, (row_imgs, label) in enumerate(zip(grids, labels)):
        for c in range(N):
            ax = axes[r, c]
            ax.imshow(row_imgs[c], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(label, color="white", fontsize=9,
                              rotation=0, labelpad=55, va="center")

    fig.suptitle("Crater reconstructions", color="#e6edf3", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅  Saved → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device       = torch.device(args.device)
    num_channels = 1

    # ── load data ─────────────────────────────────────────────────────────
    print(f"Loading data : {args.input}")
    data = load_memmap(args.input, num_channels)

    # pull held-out samples only (never seen in training)
    split_dir = os.path.dirname(args.model)
    pool_name = f"{args.split}_idx.npy"
    pool_path = os.path.join(split_dir, pool_name)

    if os.path.exists(pool_path):
        pool = np.load(pool_path)
        # test pool can be empty if num_samples was None → fall back to val
        if len(pool) == 0 and args.split == "test":
            print("  Test pool empty (num_samples was None); falling back to val.")
            pool = np.load(os.path.join(split_dir, "val_idx.npy"))
        print(f"  Sampling from '{args.split}' pool ({len(pool)} held-out images)")
    else:
        raise FileNotFoundError(
            f"{pool_path} not found. Re-run training to save split indices, "
            f"or point --model at the checkpoint dir that has *_idx.npy."
        )

    n_pick  = min(args.num_images, len(pool))
    pick    = np.random.default_rng(0).choice(pool, n_pick, replace=False)
    imgs_np = data[pick].copy()                          # (N, 1, H, W)
    imgs    = torch.from_numpy(imgs_np).to(device)

    # ── build model + load weights ───────────────────────────────────────
    # load_mae auto-detects whether this checkpoint has the clustering
    # bottleneck (old runs) or not (current default) and builds accordingly.
    print(f"Building model : {args.autoencoder_model}")
    if args.autoencoder_model == "cae":
        model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
        print(f"Loading weights: {args.model}")
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state, strict=True)
    else:
        print(f"Loading weights: {args.model}")
        model = load_mae(checkpoint_path=args.model, device=device, **MAE_KWARGS)

    model.eval()
    print("  Weights loaded ✅")

    # ── inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        if args.autoencoder_model == "cae":
            recon = model(imgs).cpu().numpy()            # (N, 1, H, W)
            originals       = imgs_np[:, 0]              # (N, H, W)
            masked_inputs   = None
            reconstructions = recon[:, 0]

        else:  # mae
            # forward returns (loss, pred, mask) for plain models, or
            # (loss, pred, mask, cls_bottleneck) for old bottleneck checkpoints.
            # pred : (B, n_patches, patch²·C)  — normalized per patch when norm_pix_loss=True
            # mask : (B, n_patches)  1=masked
            output = model(imgs, mask_ratio=args.mask_ratio)
            pred, mask = output[1], output[2]

            # ── denormalize pred (reverses norm_pix_loss normalization) ──
            # model.patchify: (B,C,H,W) → (B, n_patches, patch²·C)
            patches = model.patchify(imgs)               # original patches
            mean    = patches.mean(dim=-1, keepdim=True)
            var     = patches.var(dim=-1,  keepdim=True)
            pred_denorm = pred * (var + 1e-6).sqrt() + mean

            recon  = unpatchify(model, pred_denorm).cpu()  # (N, 1, H, W)
            masked = make_masked_image(model, imgs.cpu(), mask.cpu())

            originals       = imgs_np[:, 0]
            masked_inputs   = masked.numpy()[:, 0]
            reconstructions = recon.numpy()[:, 0]

    # ── clip to [0, 1] (norm_pix_loss can produce values outside) ─────────
    reconstructions = np.clip(reconstructions, 0, 1)

    # ── plot ──────────────────────────────────────────────────────────────
    plot_reconstructions(
        originals, masked_inputs, reconstructions,
        save_path=args.file_out,
        model_type=args.autoencoder_model,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise crater reconstructions from a trained MAE / CAE"
    )
    parser.add_argument("--autoencoder_model", choices=["cae", "mae"], default="mae")
    parser.add_argument("--input",      required=True,
                        help="Path to .dat memmap (clean or augmented)")
    parser.add_argument("--model",      required=True,
                        help="Path to saved .pth checkpoint")
    parser.add_argument("--file_out",   required=True,
                        help="Output PNG path")
    parser.add_argument("--device",     default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--num_images", type=int,   default=8,
                        help="Number of examples to visualise")
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                        help="MAE mask ratio (must match training)")
    parser.add_argument("--latent_dim", type=int,   default=64,
                        help="CAE latent dim (ignored for MAE)")
    args = parser.parse_args()
    main(args)