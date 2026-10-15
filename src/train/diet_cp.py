"""
diet_cp.py — a from-scratch reimplementation of DIET-CP (Rambhatla et al.,
2025, arXiv:2509.06990)'s continued-pretraining recipe: an OPTIONAL,
standalone alternative to DINO's own self-distillation finetuning
(train_dino.py / configs/dino_craters_finetune.yaml). Nothing in DINOv2
itself is touched - this script only imports the plain ViT backbone class
(vision_transformer.vit_small) and drives its own training loop around it,
exactly like DIET-CP's own approach: reuse the backbone, replace everything
that normally sits around it (teacher, EMA, DINO head, multi-crop) with a
single linear classifier and a single-view classification loss.

WHY: DINO's finetune collapsed onto raw brightness as its dominant learned
feature (near-perfect PC1 correlation - see docs/dino_mae_proto_finetune_
status.md) and a stronger/rebalanced augmentation recipe (dino_craters_
augmentation.py's v2) did NOT fix it. DIET-CP's objective is structurally
different in a way that can't be satisfied by a single crop-invariant
shortcut the way DINO's can - see below.

WHAT WE ADD/CHANGE VS. THE DINO FINETUNE (concrete, side by side)
─────────────────────────────────────────────────────────────────
DINO finetune has: a student AND a teacher (EMA-averaged) copy of the
backbone, a DINO head (thousands of learnable prototypes) on each, two
augmented views of the same crater pushed through student/teacher
separately, and a loss that pulls their outputs into agreement.

DIET-CP has: ONE copy of the backbone (no teacher, no EMA, no momentum
encoder), ONE new linear layer W (embed_dim -> n_images) - this is the only
new learnable component, and it is thrown away after training, never used
downstream - and ONE augmented view per step, no paired view at all.

THE LOSS: ordinary multi-class cross-entropy, but the "classes" are just
"which specific training image is this" (datum-index / instance
classification), not a degradation category. Crater #47 is permanently
class-index 47. One step: augment one crater once -> backbone -> CLS
embedding -> W -> n_images logits -> cross-entropy against the true index.

WHY THIS CAN'T COLLAPSE THE SAME WAY: DINO's "do these two crops look
similar" task is fully solvable by matching mean brightness alone - that's
the whole failure. DIET-CP's "which ONE of 321 specific craters is this"
task cannot be solved that way - many craters share similar brightness, so
brightness alone only gets a small fraction of a 321-way guess right. The
gradient has no choice but to recruit other embedding dimensions (shape,
texture, rim structure) to do better than that.

STAGED UNFREEZING (the paper's own schedule, kept as-is): backbone fully
FROZEN for the first ~5% of epochs (train only W, so the randomly
initialized classifier has something sane to push against before its
gradients start reshaping the backbone at all), then unfreeze ONLY the
last N transformer blocks (default 2) for the rest - most of the pretrained
network stays architecturally incapable of moving, a form of the
"constrain how much it's allowed to change" idea discussed earlier.

HYPERPARAMETERS: the paper's own values (lr=1e-4, weight_decay=0.05, 10%
linear warmup + cosine decay, 150 epochs, label_smoothing=0.3, batch_size
32), used as-is - the point of this experiment is "does the paper's own
recipe work here", not a fresh hyperparameter search.

DATA: a random sample (default 1,000 - the top of the paper's own tested
500-1,000 range) drawn straight from the SAME unlabeled pool DINO's own
self-distillation finetune already trains on (dino_wide/craters_wide.dat,
50,000 craters) - NOT new_test_set_split.csv's labeled "train_pool" half.
DIET-CP never looks at true_label at all (the datum index is just a
crater's position in the sampled pool, nothing about its degradation
state), so there is no leakage risk from using any unlabeled crater,
labeled-set or not - restricting to the labeled train_pool split (an
earlier version of this script did that, for a shakier "reuse the same
craters as the prototypical-loss experiment" reason) only cost us: it's
below the paper's tested range (321 < 500), and it left the full labeled
new_test_set (642 craters) needlessly off-limits, when it could instead
stay entirely untouched by ANY training and available whole for
evaluation - unlike a proto-loss checkpoint, a DIET-CP checkpoint can be
evaluated against the FULL new_set, not just the leak-safe
new_set_final_test half.

OUTPUT: a plain backbone state_dict (no "teacher"-key wrapping) - loads
through dino_backbone.load_dino_backbone() exactly like the stock
checkpoint itself does (load_pretrained_weights() already falls back to
using the raw dict when no "teacher" key is present), so this checkpoint
is evaluable through the exact same eval.evaluate pipeline as every other
DINO checkpoint in this project, no special-casing needed. The CLS-token
readout used downstream (clustering/kNN/eval) is unchanged from every
other DINO checkpoint - only how the last 2 blocks got trained differs.
"""
from __future__ import annotations
import argparse
import datetime
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from src.eval.holdout import load_holdout_dat
from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.data.cell_dino.transforms import RandomBrightness, RandomContrast
from src.models.dinov2.dinov2.data.transforms import GaussianBlur
from src.run_layout import canonical_run_dir


def load_random_pool(dat_path: str, n_images: int, crop_size: int, in_chans: int,
                     seed: int, device: str) -> torch.Tensor:
    """Randomly samples n_images craters from the full unlabeled pool (no
    label file involved at all - see module docstring for why this, not
    the labeled new_test_set_split.csv's train_pool, is the right source)."""
    data = load_holdout_dat(dat_path)
    rng = np.random.RandomState(seed)
    row_idx = rng.choice(len(data), size=min(n_images, len(data)), replace=False)
    row_idx.sort()  # sequential memmap reads, not scattered - cheaper I/O

    imgs = torch.from_numpy(np.asarray(data[row_idx])).float()  # (N, 1, 128, 128), already [0,1]
    imgs = F.interpolate(imgs, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    if in_chans > 1 and imgs.shape[1] == 1:
        imgs = imgs.repeat(1, in_chans, 1, 1)
    print(f"load_random_pool: {len(row_idx)} craters sampled from {dat_path} (of {len(data)} total)")
    return imgs.to(device)


def build_augmentation(crop_size: int):
    """Single-view augmentation (NOT multi-crop - there's no paired view to
    align here, just one augmented crop per step), now matching the paper's
    own Appendix B recipe as closely as this project's grayscale data
    allows:
        v2.RGB
        RandomResizedCrop(224, antialias=True)
        RandomHorizontalFlip()
        RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.3)
        RandomGrayscale(p=0.2)
        RandomApply([GaussianBlur((3, 3), (1.0, 2.0))], p=0.2)

    RandomResizedCrop's scale already defaults to (0.08, 1.0) - matches
    without needing to state it explicitly. v2.RGB and RandomGrayscale are
    dropped, not translated: our tensors are grayscale duplicated across 3
    identical channels (see load_random_pool), so there is no real
    per-channel color for either op to act on - RandomGrayscale would be a
    literal no-op here, not a faithful substitute for anything. ColorJitter's
    saturation/hue terms are equally inert on a zero-saturation pixel, which
    is why this project's own RandomBrightness/RandomContrast (already used
    in dino_craters_augmentation.py for this exact reason) stands in for
    ColorJitter's brightness/contrast terms - same substitution as before.

    What actually changes vs. the previous version: the paper GATES its
    jitter behind RandomApply(p=0.3) - only 30% of views get it at all,
    vs. this project's prior always-applied (p=1.0), milder (std=0.2)
    version - and its blur is stronger (sigma 1.0-2.0 vs. this project's
    prior radius 0.1-1.0). Both are corrected below to match the paper's
    actual magnitude and probability, not just its named components.

    GaussianBlur's own p ARGUMENT IS INVERTED in this environment: its
    source (dinov2/data/transforms.py) does `keep_p = 1 - p; super().
    __init__(transforms=[t], p=keep_p)`, compensating for a torchvision
    RandomApply behavior where p apparently meant "probability of
    returning the ORIGINAL image" - but measured directly against the
    torchvision actually installed here, plain RandomApply(p=X) applies
    with probability X (standard, non-inverted semantics; verified
    empirically: RandomApply(p=0.2) fires ~20.3% of 20k trials). That
    makes GaussianBlur's own compensation a double negative here:
    GaussianBlur(p=X) actually blurs with probability (1-X), confirmed
    by direct measurement (p=1.0 -> blurred 0/3000 times; p=0.2 ->
    blurred ~80.1% of 3000 times). NOT touching dinov2/data/transforms.py
    itself (project convention - DINOv2 stays unmodified); compensating
    at the call site instead, so the argument passed below is (1 -
    intended probability), with the intended value stated explicitly."""
    jitter = transforms.Compose([RandomBrightness(p=0.231), RandomContrast(p=0.231)])
    intended_blur_p = 0.2
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size,
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([jitter], p=0.3),
        GaussianBlur(p=1.0 - intended_blur_p, radius_min=1.0, radius_max=2.0),
    ])


def cosine_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    """10% linear warmup then cosine decay to 0 - the paper's own schedule
    shape, reimplemented per-step (DIET-CP's own schedule is defined that
    way, unlike train.py's adjust_learning_rate(), which steps per-epoch)."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * peak_lr * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained_weights", default="data/raw/pretrained/dinov2_vits14_pretrain.pth")
    ap.add_argument("--dat_path", default="data/processed_wac_100m_new/sigma/100/dino_wide/craters_wide.dat",
                    help="Unlabeled pool to sample from - the SAME file DINO's own "
                         "self-distillation finetune trains on, not a labeled set.")
    ap.add_argument("--n_images", type=int, default=1000,
                    help="How many craters to randomly sample from --dat_path - the "
                         "top of DIET-CP's own tested 500-1000 range.")
    ap.add_argument("--output_dir", default=None,
                    help="Defaults to the canonical logs/dino/diet-cp/... path "
                         "(run_layout.canonical_run_dir(), source='diet-cp' - a new "
                         "bucket alongside scratch/finetune/stock_pretrained, so this "
                         "training METHOD is distinguishable from a DINO hyperparameter "
                         "choice at the directory level, not just in run_manifest.json). "
                         "Override only for a one-off manual run.")
    ap.add_argument("--crop_size", type=int, default=224)
    ap.add_argument("--in_chans", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--unfreeze_last_n_blocks", type=int, default=2)
    ap.add_argument("--freeze_epochs_frac", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.10)
    ap.add_argument("--label_smoothing", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # source="diet-cp": a NEW bucket alongside canonical_run_dir()'s existing
    # scratch/finetune/stock_pretrained conventions (see run_layout.py) - this
    # is a different training METHOD, not a DINO hyperparameter choice, so it
    # gets its own place in the path (logs/dino/diet-cp/...), not folded into
    # "finetune" alongside the self-distillation runs it's being compared
    # against. structure/frozen both come out "none"/"na" the same way they
    # already do for "scratch" - canonical_run_dir()'s structure segment (the
    # warm-start checkpoint slug) is only populated for source=="finetune";
    # the actual --pretrained_weights path is still recorded in
    # run_manifest.json either way, just not reflected in the directory name.
    output_dir = args.output_dir or canonical_run_dir({
        "family": "dino", "source": "diet-cp",
        "pretrained_weights": args.pretrained_weights,
        "epochs": args.epochs, "num_samples": args.n_images,
        "diameter_range": None, "data_tag": "diet_cp_random_sample",
        "freeze_until": None,
    })
    if os.path.exists(os.path.join(output_dir, "run_manifest.json")):
        raise FileExistsError(
            f"{output_dir} already has a completed run (run_manifest.json exists) - "
            f"pass a different --seed/--n_images/--epochs, or --output_dir explicitly, "
            f"rather than overwrite it.")
    print(f"output_dir: {output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pool = load_random_pool(args.dat_path, args.n_images, args.crop_size,
                            args.in_chans, args.seed, device)
    n_images = pool.shape[0]

    # Same architecture as the stock checkpoint (patch14/in_chans3, flat
    # block_chunks=0 matching its own key naming - see dino_backbone.py's
    # STOCK_DINO_VIT_KWARGS) - a fair comparison against the self-
    # distillation finetune needs the SAME warm start, so any difference in
    # outcome is attributable to the training objective, not a different
    # initialization.
    #
    # img_size=crop_size (224), NOT the checkpoint's native 518 - we always
    # train on crop_size crops, and the SAVED checkpoint needs its pos_embed
    # in that same 224-native shape too, or it won't load through
    # dino_backbone.load_dino_backbone()'s architecture auto-detection later
    # (which infers img_size from patch_size via a fixed, project-wide
    # mapping - see _DINO_IMG_SIZE_BY_PATCH_SIZE - not from inspecting the
    # checkpoint's own pos_embed shape). Reuses ssl_meta_arch.py's own
    # _resize_pretrained_pos_embed() rather than a second implementation of
    # the same bicubic interpolation - this is calling an existing helper,
    # not modifying DINOv2 itself.
    from src.models.dinov2.dinov2.train.ssl_meta_arch import _resize_pretrained_pos_embed
    model = vit_small(img_size=args.crop_size, patch_size=14, in_chans=args.in_chans,
                      block_chunks=0, init_values=1.0)
    state_dict = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
    _resize_pretrained_pos_embed(state_dict, model.pos_embed)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # Freeze everything to start (Phase A - only W trains until the
    # unfreeze point below).
    for p in model.parameters():
        p.requires_grad = False
    # list[-0:] is the WHOLE list (== list[0:], since -0 == 0), not zero
    # elements - guard explicitly, or --unfreeze_last_n_blocks 0 (fully
    # frozen backbone) would silently unfreeze everything instead.
    unfreeze_blocks = (list(model.blocks)[-args.unfreeze_last_n_blocks:]
                       if args.unfreeze_last_n_blocks > 0 else [])

    W = nn.Linear(model.embed_dim, n_images).to(device)

    # Single optimizer built up front over the UNION of "always trainable"
    # (W) and "trainable from the unfreeze point on" (last N blocks) -
    # simpler than reconstructing the optimizer mid-run. During Phase A the
    # blocks' requires_grad=False means they never accumulate a gradient,
    # so optimizer.step() is a no-op for them regardless of being in the
    # param list; flipping requires_grad=True at the phase boundary is all
    # that's needed to start updating them from that point on.
    block_params = [p for blk in unfreeze_blocks for p in blk.parameters()]
    optimizer = torch.optim.AdamW(
        [{"params": W.parameters()}, {"params": block_params}],
        lr=args.lr, weight_decay=args.weight_decay)

    augment = build_augmentation(args.crop_size)
    steps_per_epoch = max(n_images // args.batch_size, 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_frac * total_steps)
    freeze_epochs = max(1, round(args.freeze_epochs_frac * args.epochs))

    rng = np.random.RandomState(args.seed)
    os.makedirs(output_dir, exist_ok=True)
    metrics_f = open(os.path.join(output_dir, "training_metrics.json"), "w")

    # Same tensorboard --logdir <output_dir>/tensorboard convention
    # train.py/train_dino.py both already use, so this run shows up the
    # same way in the dashboard/manual inspection - missed in the first
    # version of this script, added here to match.
    tb_dir = os.path.join(output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard  → tensorboard --logdir {tb_dir}")

    global_step = 0
    for epoch in range(args.epochs):
        if epoch == freeze_epochs:
            print(f"Epoch {epoch}: unfreezing last {args.unfreeze_last_n_blocks} blocks")
            for p in block_params:
                p.requires_grad = True

        perm = rng.permutation(n_images)
        epoch_loss, epoch_correct, epoch_seen = 0.0, 0, 0
        for i in range(0, n_images - n_images % args.batch_size, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch_targets = torch.as_tensor(idx, dtype=torch.long, device=device)
            # single augmented view per image per step, sampled fresh every
            # step (not cached) - matches "one augmented crop, no paired
            # view" from the module docstring above.
            batch = torch.stack([augment(pool[j]) for j in idx])

            lr = cosine_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            model.train()
            out = model(batch, is_training=True)
            cls_token = out["x_norm_clstoken"]
            logits = W(cls_token)
            loss = F.cross_entropy(logits, batch_targets, label_smoothing=args.label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            epoch_seen += len(idx)
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", lr, global_step)
            global_step += 1

        avg_loss = epoch_loss / epoch_seen
        acc = epoch_correct / epoch_seen
        print(f"Epoch {epoch+1:03d}/{args.epochs} | loss {avg_loss:.4f} | "
             f"datum_acc {acc:.4f} | lr {lr:.2e}")
        metrics_f.write(json.dumps({"epoch": epoch, "loss": avg_loss,
                                    "datum_accuracy": acc, "lr": lr}) + "\n")
        metrics_f.flush()
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("DatumAccuracy/train_epoch", acc, epoch)

    metrics_f.close()

    # Plain state_dict, no "teacher" wrapping - see module docstring for why
    # this loads through load_dino_backbone() unmodified.
    ckpt_dir = os.path.join(output_dir, "eval", "final")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "teacher_checkpoint.pth"))
    print(f"Checkpoint saved → {ckpt_dir}/teacher_checkpoint.pth")

    manifest = {
        "schema_version": 1,
        "family": "dino",
        "source": "diet-cp",
        "pretrained_weights": args.pretrained_weights,
        "epochs": args.epochs,
        "iterations": total_steps,
        "mask_ratio": None, "latent_dim": None, "num_samples": n_images,
        "diameter_range": None, "data_tag": "diet_cp_random_sample",
        "fov": 1.0,
        "proto_loss_enabled": False, "proto_loss_weight": None,
        "proto_n_support_per_class": None, "proto_split_csv": None,
        "augmentation_recipe": None,
        # The field that actually distinguishes this checkpoint from every
        # other DINO finetune in this project - a different training
        # METHOD entirely, not a different DINO hyperparameter.
        "training_method": "diet_cp",
        "diet_cp_unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "diet_cp_freeze_epochs": freeze_epochs,
        "param_source": "run_manifest",
        "checkpoint_path": os.path.join(ckpt_dir, "teacher_checkpoint.pth"),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "written_by": "diet_cp.py",
    }
    json.dump(manifest, open(os.path.join(output_dir, "run_manifest.json"), "w"),
              indent=2, default=float)
    print(f"Run manifest → {output_dir}/run_manifest.json")


if __name__ == "__main__":
    main()
