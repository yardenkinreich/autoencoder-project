import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import random
import sys

sys.path.append(os.path.abspath("src/models/mae"))
from src.models.autoencoder import ConvAutoencoder
from src.models.mae.models_mae import MaskedAutoencoderViT
from src.helper_functions import *

# ── constants ─────────────────────────────────────────────────────────────
# Preprocessing now outputs tensors at OUTPUT_SIZE directly (default 128).
# There is no longer an intermediate EXTRACT_SIZE on disk — the DataLoader
# receives ready-to-use (C, OUTPUT_SIZE, OUTPUT_SIZE) tensors.
# Derive INPUT_SIZE from the memmap shape at runtime (see load_memmap).


# ── dataset ───────────────────────────────────────────────────────────────
class CraterDataset(Dataset):
    """
    Wraps a pre-processed memmap of shape (N, C, H, W).

    Training  (augment=True)
    ─────────────────────────
    Applies lightweight on-the-fly augmentation on top of the stored tensor:
      • Random H-flip  p=0.5
      • Random V-flip  p=0.5

    If you passed the *augmented* .dat (craters_aug.dat) as input, each
    sample already has a unique random rotation baked in from preprocessing.
    The flips add two more degrees of freedom without any resampling cost.

    If you passed the *clean* .dat (craters_clean.dat), you may want to add
    rotation here as well — see the commented-out block below.

    Validation (augment=False)
    ───────────────────────────
    Tensors are returned as-is. No randomness.
    """

    def __init__(self, data: np.ndarray, augment: bool = True):
        self.data    = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].copy())   # (C, H, W)

        if self.augment:
            angle = random.uniform(0.0, 360.0)
            mean_val = img.mean().item()
            img = TF.rotate(img, angle,
                            interpolation=TF.InterpolationMode.BILINEAR,
                            fill=mean_val)

        return (img,)


# ── helpers ───────────────────────────────────────────────────────────────
def load_memmap(path: str, num_channels: int) -> np.memmap:
    """
    Infer N from file size.  Works for any OUTPUT_SIZE because we read
    the spatial dimension from the file size rather than hardcoding it.
    """
    file_size   = os.path.getsize(path)
    total_floats = file_size // 4          # float32 = 4 bytes
    # total_floats = N × C × H × W
    # We know C; we need to find N and H (H == W assumed square).
    # Try OUTPUT_SIZE values that are divisible by 16 (ViT patch constraint).
    for size in [128]:
        if total_floats % (num_channels * size * size) == 0:
            N = total_floats // (num_channels * size * size)
            print(f"  Detected memmap shape : ({N}, {num_channels}, {size}, {size})")
            return np.memmap(path, dtype=np.float32, mode='r',
                             shape=(N, num_channels, size, size))
    raise ValueError(
        f"Cannot infer spatial size from {path}. "
        f"File has {total_floats} floats with {num_channels} channel(s). "
        "Expected one of: 64, 128, 224, 256, 336."
    )


def make_split(data, val_split, seed):
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(data))
    val_n = int(len(data) * val_split)
    return data[idx[val_n:]], data[idx[:val_n]]


def load_pretrained_single_channel(model, checkpoint_path):
    """
    Load RGB pretrained MAE weights into a 1-channel model.
    The patch embedding conv weight (embed_dim, 3, P, P) is averaged
    across the 3 input channels → (embed_dim, 1, P, P).
    All other weights load as-is; mismatched keys stay randomly initialised.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model']

    rgb_weight  = state_dict['patch_embed.proj.weight']   # (D, 3, P, P)
    gray_weight = rgb_weight.mean(dim=1, keepdim=True)    # (D, 1, P, P)
    state_dict['patch_embed.proj.weight'] = gray_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights loaded.")
    print(f"  Missing keys   : {msg.missing_keys}")
    print(f"  Unexpected keys: {msg.unexpected_keys}")
    return model


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.text(len(train_losses) - 1, train_losses[-1],
             f"{train_losses[-1]:.4f}", color='tab:blue',
             ha='right', va='bottom', fontsize=9)
    plt.text(len(val_losses) - 1, val_losses[-1],
             f"{val_losses[-1]:.4f}", color='tab:orange',
             ha='right', va='bottom', fontsize=9)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────
def main(args):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_channels = 1
    print(f"Device : {device}")

    # ── load data ─────────────────────────────────────────────────────────
    print(f"Loading : {args.input}")
    craters = load_memmap(args.input, num_channels)
    print(f"  Total : {len(craters)}")

    # track positions in ORIGINAL .dat coordinates throughout
    orig_index = np.arange(len(craters))          # full-file indices

    if args.num_samples is not None:
        n   = min(args.num_samples, len(craters))
        sub = np.random.default_rng(seed).choice(len(craters), size=n, replace=False)
        craters    = craters[sub]
        orig_index = orig_index[sub]              # subset positions in .dat coords
        print(f"  Subset : {len(craters)}")

    print(f"  Range  : [{craters.min():.3f}, {craters.max():.3f}]")

    # ── train / val split (mirrors make_split, but also splits orig_index) ──
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(craters))
    val_n = int(len(craters) * args.val_split)
    val_perm, train_perm = perm[:val_n], perm[val_n:]

    train_data, val_data = craters[train_perm], craters[val_perm]
    train_orig = orig_index[train_perm]           # in .dat coords
    val_orig   = orig_index[val_perm]

    # "the rest": everything in the full file not in the chosen num_samples
    test_orig  = np.setdiff1d(np.arange(len(load_memmap(args.input, num_channels))),
                            orig_index)

    print(f"  Train / Val / Test : {len(train_orig)} / {len(val_orig)} / {len(test_orig)}")

    # ── persist split indices (original .dat coordinates) ───────────────────
    out_dir = os.path.dirname(args.model_output)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_idx.npy"), train_orig)
    np.save(os.path.join(out_dir, "val_idx.npy"),   val_orig)
    np.save(os.path.join(out_dir, "test_idx.npy"),  test_orig)
    print(f"  Split indices saved → {out_dir}")

    # ── datasets & loaders ────────────────────────────────────────────────
    # augment=True for training so H/V flips are applied on-the-fly.
    # If training on the clean branch, also uncomment the rotation block
    # inside CraterDataset.__getitem__.
    train_dataset = CraterDataset(train_data, augment=False)
    val_dataset   = CraterDataset(val_data,   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────
    if args.autoencoder_model == 'cae':
        model     = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
        criterion = nn.MSELoss()

    elif args.autoencoder_model == 'mae':
        model = MaskedAutoencoderViT(        # plain, bottleneck-free (current default for new runs)
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
        model = model.to(device)
        if args.pretrained_weights and os.path.exists(args.pretrained_weights):
            print(f"Loading pretrained weights : {args.pretrained_weights}")
            model = load_pretrained_single_channel(model, args.pretrained_weights)
        else:
            print("No pretrained weights — training from scratch.")
        model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # ── training loop ─────────────────────────────────────────────────────
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for (imgs,) in train_loader:
            imgs = imgs.to(device)
            if args.autoencoder_model == 'cae':
                loss = criterion(model(imgs), imgs)
            else:
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)

        train_loss = running / len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for (imgs,) in val_loader:
                imgs = imgs.to(device)
                if args.autoencoder_model == 'cae':
                    loss = criterion(model(imgs), imgs)
                else:
                    loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)
                running += loss.item() * imgs.size(0)

        val_loss = running / len(val_dataset)
        val_losses.append(val_loss)

        scheduler.step()
        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

    # ── save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"Model saved → {args.model_output}")

    plot_losses(train_losses, val_losses, args.loss_plot)
    print(f"Loss plot  → {args.loss_plot}")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder_model',  choices=['cae', 'mae'], default='mae')
    parser.add_argument('--input',              required=True,
                        help="Path to .dat memmap (clean or augmented branch)")
    parser.add_argument('--model_output',       required=True)
    parser.add_argument('--loss_plot',          required=True)
    parser.add_argument('--pretrained_weights', type=str,   default=None)
    parser.add_argument('--epochs',             type=int,   default=50)
    parser.add_argument('--batch_size',         type=int,   default=32)
    parser.add_argument('--latent_dim',         type=int,   default=6)
    parser.add_argument('--lr',                 type=float, default=1e-3)
    parser.add_argument('--weight_decay',       type=float, default=1e-5)
    parser.add_argument('--min_lr',             type=float, default=1e-8)
    parser.add_argument('--val_split',          type=float, default=0.2)
    parser.add_argument('--num_samples',        type=int,   default=None)
    parser.add_argument('--mask_ratio',         type=float, default=0.75)
    args = parser.parse_args()
    main(args)