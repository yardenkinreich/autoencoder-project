"""
cluster.py
──────────
Encode craters into latent space and visualise with PCA / t-SNE.

Usage:
    python src/cluster/cluster.py encode   --imgs-dir ...  --model ... --autoencoder-model mae ...
    python src/cluster/cluster.py plot-dots --latents ...  --states ... --out-png ...
    python src/cluster/cluster.py plot-imgs --latents ...  --imgs-dir ... --out-png ...
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src/models/mae"))   # for util.pos_embed
from src.models.autoencoder import ConvAutoencoder
from src.models.mae_bottleneck import load_mae, encode_for_clustering
from src.models.dino_backbone import load_dino_backbone

# ── architecture — must mirror train.py exactly ───────────────────────────────
INPUT_SIZE = 128      # ← was 224; matches OUTPUT_SIZE in preprocess.py

MAE_KWARGS = dict(
    img_size=128,
    patch_size=8,     # ← was 16; matches train.py
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

STATE_LABELS = {1: "New Crater", 2: "Semi New Crater",
                3: "Semi Old Crater", 4: "Old Crater"}
STATE_COLORS = {1: "tab:blue", 2: "tab:green",
                3: "tab:orange", 4: "tab:red"}


# ── model builder ─────────────────────────────────────────────────────────────

def build_model(autoencoder_model: str, bottleneck: int,
                model_path: str, device: torch.device):
    if autoencoder_model == "cae":
        model = ConvAutoencoder(latent_dim=bottleneck)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        return model

    if autoencoder_model == "dino":
        # bottleneck is unused here — DINO's embedding dim is fixed by the
        # architecture (384 for vit_small), not user-settable via this flag.
        return load_dino_backbone(model_path, device=device)

    # mae — load_mae auto-detects whether this checkpoint has the clustering
    # bottleneck (old runs) or not (current default) and builds accordingly.
    model = load_mae(checkpoint_path=model_path, device=device, **MAE_KWARGS)
    model.eval()
    return model


# ── encoding helpers ──────────────────────────────────────────────────────────
# thin wrappers so build_model()/encode_images() below can dispatch on
# autoencoder_model without an if/else at every call site.
def _encode_mae_batch(model, imgs: torch.Tensor) -> torch.Tensor:
    return encode_for_clustering(model, imgs)


def _encode_dino_batch(model, imgs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(imgs)   # normalized CLS token (see src/models/dino_backbone.py)


def _encode_cae_batch(model: ConvAutoencoder,
                      imgs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.encoder(imgs)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_images(imgs_dir: str) -> tuple:
    """Load Julie's PNG dataset, resize to INPUT_SIZE (128)."""
    files  = sorted(f for f in os.listdir(imgs_dir) if f.endswith(".png"))
    states = np.array([int(f.split("_")[1].split(".")[0]) for f in files])

    imgs_list = []
    for f in files:
        img = Image.open(os.path.join(imgs_dir, f)).convert("L")
        img = img.resize((INPUT_SIZE, INPUT_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs_list.append(torch.from_numpy(arr).unsqueeze(0))   # (1, H, W)

    imgs = torch.stack(imgs_list).float()   # (N, 1, H, W)
    return imgs, states, files


# ── encode entry point ────────────────────────────────────────────────────────

def encode_images(
    inputs,            # (N,1,H,W) tensor  OR  a DataLoader
    model_path: str,
    bottleneck: int,
    device: torch.device,
    out_latents: str,
    out_states: str,
    states=None,
    autoencoder_model: str = "mae",
    is_dataloader: bool = False,
):
    model  = build_model(autoencoder_model, bottleneck, model_path, device)
    encode = {
        "mae":  _encode_mae_batch,
        "dino": _encode_dino_batch,
        "cae":  _encode_cae_batch,
    }[autoencoder_model]

    if is_dataloader:
        latents_list = []
        n = len(inputs)
        with torch.no_grad():
            for i, batch in enumerate(inputs):
                if i % 100 == 0:
                    print(f"  Batch {i}/{n}")
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                latents_list.append(encode(model, x.to(device)).cpu())
        latents = torch.cat(latents_list, dim=0).numpy()
    else:
        with torch.no_grad():
            latents = encode(model, inputs.to(device)).cpu().numpy()

    os.makedirs(os.path.dirname(out_latents) or ".", exist_ok=True)
    np.save(out_latents, latents)
    if states is not None:
        np.save(out_states, states)
    print(f"  Latents saved → {out_latents}  shape={latents.shape}")


# ── dimensionality reduction ──────────────────────────────────────────────────

def _reduce(latents: np.ndarray, technique: str):
    if technique == "pca":
        pca    = PCA(n_components=2)
        coords = pca.fit_transform(latents)
        ev     = pca.explained_variance_ratio_
        xlabel = f"PC1 ({ev[0]*100:.1f}%)"
        ylabel = f"PC2 ({ev[1]*100:.1f}%)"
        print(f"[PCA] Variance explained: {ev.sum()*100:.2f}%")
    elif technique == "tsne":
        coords = TSNE(n_components=2, random_state=42).fit_transform(latents)
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    else:
        raise ValueError(f"Unknown technique: {technique}")
    return coords, xlabel, ylabel


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_dots(latents_path: str, states_path: str,
              out_png: str, technique: str, model_name: str):
    latents = np.load(latents_path)
    states  = np.load(states_path).squeeze()
    coords, xlabel, ylabel = _reduce(latents, technique)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"{technique.upper()} — Latent Space coloured by Deformation State\n{model_name}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    for s in np.unique(states):
        mask = states == s
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=STATE_LABELS.get(s, f"state {s}"),
                   c=STATE_COLORS.get(s, "gray"),
                   alpha=0.7, s=20)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  Saved → {out_png}")


def plot_imgs(latents_path: str, imgs_dir: str,
              out_png: str, technique: str, model_name: str):
    latents = np.load(latents_path)
    coords, xlabel, ylabel = _reduce(latents, technique)

    files = sorted(f for f in os.listdir(imgs_dir) if f.endswith(".png"))
    assert len(files) == len(latents), (
        f"Mismatch: {len(files)} images vs {len(latents)} latents"
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"{technique.upper()} — {model_name} latent image clustering")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    for (x, y), fname in zip(coords, files):
        img = Image.open(os.path.join(imgs_dir, fname)).convert("L")
        ab  = AnnotationBbox(OffsetImage(img, zoom=0.2, cmap="gray"),
                             (x, y), frameon=False)
        ax.add_artist(ab)

    ax.update_datalim(coords)
    ax.autoscale()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  Saved → {out_png}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest="command")

    # encode
    enc = sub.add_parser("encode")
    enc.add_argument("--imgs-dir",          required=True)
    enc.add_argument("--model",             required=True)
    enc.add_argument("--autoencoder-model", choices=["cae", "mae", "dino"], required=True)
    enc.add_argument("--bottleneck",        type=int,   default=6)
    enc.add_argument("--out-latents",       required=True)
    enc.add_argument("--out-states",        required=True)

    # plot-dots
    dots = sub.add_parser("plot-dots")
    dots.add_argument("--latents",    required=True)
    dots.add_argument("--states",     required=True)
    dots.add_argument("--out-png",    required=True)
    dots.add_argument("--technique",  choices=["pca", "tsne"], default="pca")
    dots.add_argument("--model-name", required=True)

    # plot-imgs
    imgs = sub.add_parser("plot-imgs")
    imgs.add_argument("--latents",    required=True)
    imgs.add_argument("--imgs-dir",   required=True)
    imgs.add_argument("--out-png",    required=True)
    imgs.add_argument("--technique",  choices=["pca", "tsne"], default="pca")
    imgs.add_argument("--model-name", required=True)

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "encode":
        inputs, states, _ = load_images(args.imgs_dir)
        encode_images(inputs, args.model, args.bottleneck, device,
                      args.out_latents, args.out_states, states,
                      autoencoder_model=args.autoencoder_model)

    elif args.command == "plot-dots":
        plot_dots(args.latents, args.states, args.out_png,
                  args.technique, args.model_name)

    elif args.command == "plot-imgs":
        plot_imgs(args.latents, args.imgs_dir, args.out_png,
                  args.technique, args.model_name)

    else:
        parser.print_help()