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
from src.models.dino_backbone import (
    load_dino_backbone, load_stock_pretrained_dino, encode_stock_pretrained_dino,
    STOCK_DINO_CHECKPOINT,
)
from src.models.mae_pretrained import (
    load_stock_pretrained_mae, encode_stock_pretrained_mae, STOCK_MAE_CHECKPOINT,
)

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
        # Auto-detect latent_dim from the checkpoint's own encoder bottleneck
        # layer rather than trusting a caller-supplied `bottleneck` - that
        # value has silently drifted between callers (this module's own CLI
        # defaults to 6; eval/pipeline.py's default is 64) and previously
        # caused a hard state_dict shape-mismatch crash on any checkpoint
        # trained with a different latent_dim (e.g. a lat40 run evaluated
        # with the eval harness's default 64). Same auto-detect philosophy
        # load_mae() below already uses for MAE checkpoints.
        state_dict = torch.load(model_path, map_location=device)
        detected = state_dict.get("encoder.9.weight")
        latent_dim = int(detected.shape[0]) if detected is not None else bottleneck
        model = ConvAutoencoder(latent_dim=latent_dim)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model

    if autoencoder_model == "dino":
        # bottleneck is unused here — DINO's embedding dim is fixed by the
        # architecture (384 for vit_small), not user-settable via this flag.
        return load_dino_backbone(model_path, device=device)

    if autoencoder_model == "dino_pretrained":
        # frozen, stock (unmodified architecture) Meta ImageNet/LVD-142M
        # checkpoint - no crater-specific training at all. See
        # src/models/dino_backbone.py's STOCK_DINO_* for why. model_path is
        # optional here (falls back to the default download location) since
        # there's no user-trained checkpoint to point at.
        return load_stock_pretrained_dino(
            model_path or STOCK_DINO_CHECKPOINT, device=device)

    if autoencoder_model == "mae_pretrained":
        # frozen, stock official Meta ImageNet MAE checkpoint - the same
        # "use it as-is, no crater training" comparison as dino_pretrained.
        # See src/models/mae_pretrained.py.
        return load_stock_pretrained_mae(
            model_path or STOCK_MAE_CHECKPOINT, device=device)

    # mae — load_mae auto-detects whether this checkpoint has the clustering
    # bottleneck (old runs) or not (current default) and builds accordingly,
    # and also auto-detects patch_size/img_size from the checkpoint itself
    # (overriding MAE_KWARGS's img_size=128/patch_size=8 when a checkpoint
    # predates that convention - see model_input_size() below, which callers
    # use to resize their input images to match).
    model = load_mae(checkpoint_path=model_path, device=device, **MAE_KWARGS)
    model.eval()
    return model


def model_input_size(model, default: int = INPUT_SIZE) -> int:
    """The square image size `model` actually expects, read off the model
    itself rather than assumed. MAE checkpoints can have their own native
    resolution auto-detected at load time (see mae_bottleneck.load_mae) that
    differs from this project's current INPUT_SIZE=128 convention - callers
    (eval/pipeline.py's embed(), eval/holdout.py's embed_holdout()) resize
    to whatever this returns, not a hardcoded constant, so an
    older-resolution checkpoint can still be evaluated correctly. Falls back
    to `default` for architectures that don't expose this the same way
    (CAE has no patch embedding at all; DINO's own backbone is handled
    separately and always uses its own fixed resolution)."""
    patch_embed = getattr(model, "patch_embed", None)
    img_size = getattr(patch_embed, "img_size", None)
    if img_size is not None:
        return int(img_size[0])
    return default


# ── encoding helpers ──────────────────────────────────────────────────────────
# thin wrappers so build_model()/encode_images() below can dispatch on
# autoencoder_model without an if/else at every call site.
def _encode_mae_batch(model, imgs: torch.Tensor) -> torch.Tensor:
    return encode_for_clustering(model, imgs)


def _encode_dino_batch(model, imgs: torch.Tensor) -> torch.Tensor:
    # imgs arrives single-channel (grayscale craters) regardless of
    # architecture - the old from-scratch "dino" convention (patch8/
    # in_chans1) wants that as-is, but configs/dino_craters_finetune.yaml's
    # checkpoint-matching architecture (patch14/in_chans3, see
    # dino_backbone.py's _detect_dino_arch) expects 3 channels, same
    # grayscale-replicated-3x convention CraterAugmentationDINO already uses
    # at training time (src/data/dino_craters_augmentation.py) and
    # encode_stock_pretrained_dino already does for the stock checkpoint.
    # Read the model's OWN expected channel count rather than assuming -
    # same auto-detect philosophy as load_dino_backbone()'s architecture
    # detection, so this stays correct for either "dino" architecture.
    in_chans = model.patch_embed.proj.in_channels
    if imgs.shape[1] != in_chans:
        imgs = imgs.repeat(1, in_chans, 1, 1)
    with torch.no_grad():
        return model(imgs)   # normalized CLS token (see src/models/dino_backbone.py)


def _encode_dino_pretrained_batch(model, imgs: torch.Tensor) -> torch.Tensor:
    return encode_stock_pretrained_dino(model, imgs)


def _encode_mae_pretrained_batch(model, imgs: torch.Tensor) -> torch.Tensor:
    return encode_stock_pretrained_mae(model, imgs)


def _encode_cae_batch(model: ConvAutoencoder,
                      imgs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.encoder(imgs)


# public dispatch table — reused by src/eval/pipeline.py so there's one
# place that knows how to turn (autoencoder_model, model, imgs) into an
# embedding, instead of a third copy of this dispatch logic.
ENCODE_FNS = {
    "mae":             _encode_mae_batch,
    "dino":            _encode_dino_batch,
    "dino_pretrained": _encode_dino_pretrained_batch,
    "mae_pretrained":  _encode_mae_pretrained_batch,
    "cae":             _encode_cae_batch,
}


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
    encode = ENCODE_FNS[autoencoder_model]

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