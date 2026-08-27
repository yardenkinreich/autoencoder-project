import os
import argparse
import datetime
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
import random
import sys
from torch.utils.tensorboard import SummaryWriter

from src.train.prototypical_loss import LabeledEpisodePool, prototypical_loss

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


def is_distributed() -> bool:
    """True when launched via torchrun (WORLD_SIZE env var set, >1 process)."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    """
    Single-node multi-GPU only. Launch with:
        torchrun --standalone --nproc_per_node=<N> src/train/train.py ...
    Plain `python src/train/train.py ...` still works unchanged (world_size=1).
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def adjust_learning_rate(optimizer, epoch, base_lr, min_lr, warmup_epochs, epochs):
    """Linear warmup then half-cycle cosine decay - matches the official MAE
    recipe's src/models/mae/util/lr_sched.py exactly (per-epoch here rather
    than per-iteration, since this trainer steps once per epoch)."""
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(warmup_epochs, 1)
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1. + math.cos(math.pi * (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


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


def _write_run_manifest(args):
    """Standardized run metadata, written alongside the checkpoint so the
    dashboard (src/dashboard/model_meta.py) can read structured params
    instead of regex-parsing directory names / Snakefile.snapshot. No-ops if
    --run_dir wasn't passed (e.g. train.py invoked by hand outside the
    Snakefile) - this is purely additive, never required for training itself
    to succeed."""
    if not getattr(args, "run_dir", None):
        return
    diameter_range = ([args.min_diameter, args.max_diameter]
                      if args.min_diameter is not None and args.max_diameter is not None
                      else None)
    manifest = {
        "schema_version": 1,
        "family": args.autoencoder_model,
        "source": "finetune" if args.pretrained_weights else "scratch",
        "pretrained_weights": args.pretrained_weights,
        "epochs": args.epochs,
        "mask_ratio": args.mask_ratio if args.autoencoder_model == "mae" else None,
        "latent_dim": args.latent_dim if args.autoencoder_model == "cae" else None,
        "num_samples": args.num_samples,
        "diameter_range": diameter_range,
        "data_tag": args.data_tag,
        # train.py only ever trains mae/cae, both always at the 0.5x-diameter
        # crop radius (see preprocess_2.py's default --clean_offset) - unlike
        # dashboard/model_meta.py's parsers, which have to infer this from
        # family since they read older runs that predate this field, a
        # freshly-written manifest can just state it directly.
        "fov": 0.5,
        # Same schema/fields train_dino.py's manifest writer uses - see that
        # file's own comment for why this is recorded (lets the dashboard
        # distinguish a proto-loss run from a plain one without re-parsing
        # config/CLI args).
        "proto_loss_enabled": bool(args.proto_enabled),
        "proto_loss_weight": args.proto_loss_weight if args.proto_enabled else None,
        "proto_n_support_per_class": args.proto_n_support_per_class if args.proto_enabled else None,
        "proto_split_csv": args.proto_split_csv if args.proto_enabled else None,
        "param_source": "run_manifest",
        "checkpoint_path": args.model_output,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "written_by": "train.py",
    }
    os.makedirs(args.run_dir, exist_ok=True)
    json.dump(manifest, open(os.path.join(args.run_dir, "run_manifest.json"), "w"),
              indent=2, default=float)
    print(f"Run manifest → {args.run_dir}/run_manifest.json")


# ── main ──────────────────────────────────────────────────────────────────
def main(args):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    distributed = is_distributed()
    if distributed:
        local_rank, rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)

    num_channels = 1
    if is_main:
        print(f"Device : {device}  |  world_size={world_size}")

    # ── load data ─────────────────────────────────────────────────────────
    # Deterministic and identical on every rank (same seed, no rank-dependent
    # branching) so all processes agree on the same train/val/test split.
    if is_main:
        print(f"Loading : {args.input}")
    craters = load_memmap(args.input, num_channels)
    if is_main:
        print(f"  Total : {len(craters)}")

    # track positions in ORIGINAL .dat coordinates throughout
    orig_index = np.arange(len(craters))          # full-file indices

    if args.num_samples is not None:
        n   = min(args.num_samples, len(craters))
        sub = np.random.default_rng(seed).choice(len(craters), size=n, replace=False)
        craters    = craters[sub]
        orig_index = orig_index[sub]              # subset positions in .dat coords
        if is_main:
            print(f"  Subset : {len(craters)}")

    if is_main:
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

    out_dir = os.path.dirname(args.model_output)
    writer = None
    if is_main:
        print(f"  Train / Val / Test : {len(train_orig)} / {len(val_orig)} / {len(test_orig)}")

        # ── persist split indices (original .dat coordinates) ───────────────
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "train_idx.npy"), train_orig)
        np.save(os.path.join(out_dir, "val_idx.npy"),   val_orig)
        np.save(os.path.join(out_dir, "test_idx.npy"),  test_orig)
        print(f"  Split indices saved → {out_dir}")

        # ── live loss monitoring ─────────────────────────────────────────────
        # tensorboard --logdir <out_dir>/tensorboard  (while training runs)
        tb_dir = os.path.join(out_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"  TensorBoard  → tensorboard --logdir {tb_dir}")

    if distributed:
        dist.barrier()   # make sure out_dir exists before any rank needs it

    # ── datasets & loaders ────────────────────────────────────────────────
    # augment=True for training so H/V flips are applied on-the-fly.
    # If training on the clean branch, also uncomment the rotation block
    # inside CraterDataset.__getitem__.
    train_dataset = CraterDataset(train_data, augment=False)
    val_dataset   = CraterDataset(val_data,   augment=False)

    # Each rank sees a different shard of train_data; validation runs on
    # rank 0 only, over the full val_dataset (simpler and exact — no
    # cross-rank loss averaging needed for the number that gets logged).
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=4, pin_memory=True)
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
            if is_main:
                print(f"Loading pretrained weights : {args.pretrained_weights}")
            model = load_pretrained_single_channel(model, args.pretrained_weights)
        elif is_main:
            print("No pretrained weights — training from scratch.")
        model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    do_proto = args.proto_enabled and args.autoencoder_model == 'mae'
    proto_pool = proto_rng = None
    if args.proto_enabled and args.autoencoder_model != 'mae':
        raise ValueError("--proto_enabled only supports --autoencoder_model mae")
    if do_proto:
        if is_main:
            print(f"Proto loss enabled: weight={args.proto_loss_weight}, "
                 f"n_support_per_class={args.proto_n_support_per_class}, "
                 f"dat_path={args.proto_dat_path}")
        # crop_size=128/in_chans=1: MAE's own native resolution/channel
        # convention (img_size=128 above) - unlike DINO's proto pool, which
        # resizes to 224 and replicates to 3 channels to match a checkpoint
        # it didn't train from scratch, MAE's encoder here IS the thing
        # being trained, so the pool should just match its actual input
        # shape directly, no reframing needed.
        proto_pool = LabeledEpisodePool(
            dat_path=args.proto_dat_path, metadata_csv=args.proto_metadata_csv,
            split_csv=args.proto_split_csv, class_names=args.proto_class_names.split(","),
            crop_size=128, in_chans=1, device=str(device),
        )
        proto_rng = np.random.RandomState(args.proto_seed)

    def embed_for_proto(x):
        # forward_encoder(..., mask_ratio=0.0)[:, 0, :] - full image (no
        # masking) through the encoder, CLS token out - matches
        # mae_bottleneck.encode_for_clustering()'s own convention exactly
        # (this project's established eval-time MAE embedding), so proto
        # loss shapes the SAME embedding space eval will later score, not a
        # differently-defined one. Unlike encode_for_clustering, no
        # torch.no_grad() here - this needs to backprop into the encoder.
        m = model.module if distributed else model
        latent, _, _ = m.forward_encoder(x, mask_ratio=0.0)
        return latent[:, 0, :]

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
        # betas match the official MAE pretraining recipe (main_pretrain.py),
        # not AdamW's default (0.9, 0.999) - beta2=0.95 reacts faster to a
        # sudden gradient-magnitude spike instead of smoothing it away.
    )
    # AMP: also gives free protection against a bad/exploding step - GradScaler
    # detects an inf/nan-producing gradient, skips that optimizer step, and
    # backs off the scale, instead of applying it. This is what the official
    # recipe actually relies on for pretraining-time stability (it does NOT
    # use explicit gradient clipping during pretraining - only fine-tuning).
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if distributed else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main:
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    def save_checkpoint(epoch):
        ckpt_path = os.path.join(out_dir, "checkpoint_last.pth")
        torch.save({
            "model": (model.module if distributed else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path} (epoch {epoch+1})")

    def save_best_model(epoch, val_loss):
        # standard practice: track the model at its best validation loss
        # separately from the last-epoch weights, since the final epoch
        # isn't necessarily the best one (and, as we've seen, could even be
        # a corrupted step) - weights only, not a resumable checkpoint.
        best_path = os.path.join(out_dir, "model_best.pth")
        torch.save((model.module if distributed else model).state_dict(), best_path)
        print(f"Best model saved → {best_path} (epoch {epoch+1}, val_loss {val_loss:.4f})")

    # ── training loop ─────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)   # different shuffle per epoch, per rank

        lr = adjust_learning_rate(optimizer, epoch, args.lr, args.min_lr,
                                  args.warmup_epochs, args.epochs)

        model.train()
        running, n_seen = 0.0, 0
        proto_running, proto_steps = 0.0, 0
        for (imgs,) in train_loader:
            imgs = imgs.to(device)

            with torch.cuda.amp.autocast():
                if args.autoencoder_model == 'cae':
                    loss = criterion(model(imgs), imgs)
                else:
                    loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)

                proto_loss_value = None
                if do_proto:
                    sx, sy, qx, qy = proto_pool.sample(args.proto_n_support_per_class, proto_rng)
                    proto_loss_t = prototypical_loss(embed_for_proto, sx, sy, qx, qy, proto_pool.n_classes)
                    # one combined loss, one backward() call below - same
                    # single-accumulator pattern as ssl_meta_arch.py's
                    # forward_backward(), required for GradScaler to see one
                    # coherent loss to scale rather than two separate calls.
                    loss = loss + args.proto_loss_weight * proto_loss_t
                    proto_loss_value = proto_loss_t.item()

            loss_value = loss.item()
            # matches engine_pretrain.py's finite-loss check - halt immediately
            # on a bad step instead of silently continuing on a corrupted model
            bad = torch.tensor(0.0 if math.isfinite(loss_value) else 1.0, device=device)
            if distributed:
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item() > 0:
                # do NOT save a checkpoint here - the model is already
                # corrupted at this point, saving now would overwrite the
                # last good periodic checkpoint with garbage. Recovery is
                # via --resume from checkpoint_last.pth (last good save).
                if is_main:
                    print(f"Loss is {loss_value}, stopping training.")
                if distributed:
                    dist.destroy_process_group()
                sys.exit(1)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss_value * imgs.size(0)
            n_seen  += imgs.size(0)
            if proto_loss_value is not None:
                # plain per-STEP average, not weighted by imgs.size(0) - that's
                # the reconstruction batch's size, unrelated to proto_loss's
                # own per-episode query count (already averaged internally by
                # cross_entropy's default reduction), so weighting by it here
                # wouldn't reflect proto_loss's actual sample size anyway.
                proto_running += proto_loss_value
                proto_steps += 1

            if is_main:
                writer.add_scalar("Loss/train_step", loss_value, global_step)
                if proto_loss_value is not None:
                    writer.add_scalar("Loss/proto_train_step", proto_loss_value, global_step)
            global_step += 1

        if distributed:
            # each rank only trained on its own shard — reduce to the true
            # global average before logging/reporting
            stats = torch.tensor([running, float(n_seen)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss = (stats[0] / stats[1]).item()
        else:
            train_loss = running / n_seen
        train_losses.append(train_loss)
        proto_train_loss = (proto_running / proto_steps) if proto_steps else None

        # validation runs on rank 0 only, over the full val set
        if is_main:
            model.eval()
            running = 0.0
            with torch.no_grad(), torch.cuda.amp.autocast():
                for (imgs,) in val_loader:
                    imgs = imgs.to(device)
                    if args.autoencoder_model == 'cae':
                        loss = criterion(model(imgs), imgs)
                    else:
                        loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)
                    running += loss.item() * imgs.size(0)
            val_loss = running / len(val_dataset)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_model(epoch, val_loss)

        if is_main:
            writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            writer.add_scalar("Loss/val_epoch",   val_loss,   epoch)
            writer.add_scalar("LR",               lr,         epoch)
            proto_suffix = ""
            if proto_train_loss is not None:
                writer.add_scalar("Loss/proto_train_epoch", proto_train_loss, epoch)
                proto_suffix = f" | Proto {proto_train_loss:.4f}"
            print(f"Epoch {epoch+1:03d}/{args.epochs} | "
                  f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                  f"LR {lr:.2e}{proto_suffix}")
            # matches the official recipe's cadence (main_pretrain.py: every
            # 20 epochs + the final one) - so a bad step doesn't cost all
            # progress since the beginning, only since the last checkpoint
            if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
                save_checkpoint(epoch)

    # ── save ──────────────────────────────────────────────────────────────
    if is_main:
        writer.close()
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        state_dict = model.module.state_dict() if distributed else model.state_dict()
        torch.save(state_dict, args.model_output)
        print(f"Model saved → {args.model_output}")

        _write_run_manifest(args)

        plot_losses(train_losses, val_losses, args.loss_plot)
        print(f"Loss plot  → {args.loss_plot}")

    if distributed:
        dist.destroy_process_group()


# ── CLI ───────────────────────────────────────────────────────────────────
# Single GPU (unchanged, e.g. pin to a specific free GPU on a shared machine):
#     CUDA_VISIBLE_DEVICES=7 python src/train/train.py --input ... --model_output ...
#
# Multi-GPU, single node — batch_size below is PER GPU:
#     torchrun --standalone --nproc_per_node=4 src/train/train.py --input ... --model_output ...
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder_model',  choices=['cae', 'mae'], default='mae')
    parser.add_argument('--input',              required=True,
                        help="Path to .dat memmap (clean or augmented branch)")
    parser.add_argument('--model_output',       required=True)
    parser.add_argument('--loss_plot',          required=True)
    parser.add_argument('--pretrained_weights', type=str,   default=None)
    parser.add_argument('--resume',             type=str,   default=None,
                        help="Path to a checkpoint_last.pth to resume from "
                             "(model/optimizer/scaler/epoch)")
    parser.add_argument('--checkpoint_every',   type=int,   default=20,
                        help="Save a resumable checkpoint every N epochs "
                             "(matches the official MAE recipe's cadence)")
    parser.add_argument('--warmup_epochs',      type=int,   default=40,
                        help="Linear LR warmup epochs before cosine decay "
                             "(official MAE default for an 400-epoch "
                             "schedule; scale proportionally for other "
                             "--epochs, e.g. ~10% of --epochs)")
    parser.add_argument('--epochs',             type=int,   default=50)
    parser.add_argument('--batch_size',         type=int,   default=32)
    parser.add_argument('--latent_dim',         type=int,   default=6)
    parser.add_argument('--lr',                 type=float, default=1e-3)
    parser.add_argument('--weight_decay',       type=float, default=1e-5)
    parser.add_argument('--min_lr',             type=float, default=1e-8)
    parser.add_argument('--val_split',          type=float, default=0.2)
    parser.add_argument('--num_samples',        type=int,   default=None)
    parser.add_argument('--mask_ratio',         type=float, default=0.75)
    parser.add_argument('--run_dir',            type=str,   default=None,
                        help="Run directory to write run_manifest.json into "
                             "(standardized metadata for the dashboard) - "
                             "skipped if not given")
    parser.add_argument('--min_diameter',       type=float, default=None,
                        help="For run_manifest.json only - the crater "
                             "diameter range (km) this run's --input was "
                             "filtered to, not used for training itself")
    parser.add_argument('--max_diameter',       type=float, default=None)
    parser.add_argument('--data_tag',           type=str,   default=None,
                        help="For run_manifest.json only - short label for "
                             "--input's data source")
    # Prototypical-loss auxiliary supervision (src/train/prototypical_loss.py) -
    # same mechanism already wired into DINO's finetune loop
    # (dinov2/train/ssl_meta_arch.py / configs/dino_craters_finetune.yaml),
    # ported here architecture-agnostically per that module's own design
    # (embed_fn is the only backbone-specific piece). MAE only for now (CAE
    # not wired up) - only meaningful with --autoencoder_model mae.
    parser.add_argument('--proto_enabled', action='store_true',
                        help="Add a prototypical-loss term (Snell et al. 2017) on top "
                             "of the reconstruction loss, drawing episodes from "
                             "--proto_dat_path's train_pool split. MAE only.")
    parser.add_argument('--proto_loss_weight', type=float, default=0.05,
                        help="Multiplier on the raw proto_loss before it's added to "
                             "the reconstruction loss - NOT the same default as DINO's "
                             "config (0.2): MAE's own reconstruction loss sits on a much "
                             "smaller scale (~1-3) than DINO's combined total (~18-20), "
                             "so the same 0.2 weight gave proto_loss ~7-12% of the total "
                             "in a smoke test (src/train/prototypical_loss.py) vs DINO's "
                             "~1-2% - 0.05 was chosen to land in that same gentle-nudge "
                             "ballpark instead of dominating early training.")
    parser.add_argument('--proto_n_support_per_class', type=int, default=5)
    parser.add_argument('--proto_dat_path', type=str,
                        default="data/processed_wac_100m_new/sigma/100/new_test_set/craters.dat",
                        help="Narrow (0.5x-offset) new_test_set variant - MAE always "
                             "trains at that FOV (see fov=0.5 in _write_run_manifest), "
                             "unlike DINO's proto config which uses the _wide variant.")
    parser.add_argument('--proto_metadata_csv', type=str,
                        default="data/processed_wac_100m_new/sigma/100/new_test_set/metadata.csv")
    parser.add_argument('--proto_split_csv', type=str,
                        default="configs/new_test_set_split.csv")
    parser.add_argument('--proto_class_names', type=str,
                        default="v-fr,r-fr,r-dr,v-dr",
                        help="Comma-separated, ordinal order (index 0 = freshest)")
    parser.add_argument('--proto_seed', type=int, default=0)
    args = parser.parse_args()
    main(args)