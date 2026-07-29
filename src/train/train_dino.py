"""
train_dino.py
──────────────
Thin CLI wrapper for DINOv2 self-supervised pretraining on crater crops.

Loads configs/dino_craters.yaml, applies a few common overrides as normal
CLI flags (matching the style of src/train/train.py), then drives DINOv2's
own validated training orchestration (SSLMetaArch + do_train(), in
src/models/dinov2/dinov2/train/train.py) directly — same thing its own
entry point does, just without requiring the --config-file/key=value CLI
convention for the common case.

For anything not exposed as a flag here, edit configs/dino_craters.yaml
directly, or pass extra Yacs-style overrides via --opts key=value key=value.

Unlike train.py, this has no CAE/MAE-style plain-python single-process loop
to fall back on for multi-GPU: DINOv2's own do_train()/SSLMetaArch always
goes through FSDP wrapping (even at world_size=1), so this always requires
torch.distributed to be initialized — run via torchrun, same as train.py's
multi-GPU path, even for a single GPU:
    torchrun --standalone --nproc_per_node=1 src/train/train_dino.py --input ... --output_dir ...
"""

import argparse

import torch

from src.models.dinov2.dinov2.utils.config import setup
from src.models.dinov2.dinov2.train.ssl_meta_arch import SSLMetaArch
from src.models.dinov2.dinov2.train.train import do_train


def main(args):
    opts = list(args.opts)
    opts.append(f"train.dataset_path=Craters:root={args.input}")
    if args.epochs is not None:
        opts.append(f"optim.epochs={args.epochs}")
    if args.batch_size is not None:
        opts.append(f"train.batch_size_per_gpu={args.batch_size}")

    setup_args = argparse.Namespace(
        config_file=args.config_file,
        output_dir=args.output_dir,
        opts=opts,
        seed=0,
    )
    cfg = setup(setup_args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 self-supervised pretraining on crater crops")
    parser.add_argument("--input",       required=True,
                        help="Path to the DINO-specific craters_*.dat memmap "
                             "(wide-FOV, single-branch — see preprocess_2.py --clean_offset)")
    parser.add_argument("--output_dir",  required=True,
                        help="Where checkpoints, tensorboard/, and training_metrics.json go")
    parser.add_argument("--config_file", default="configs/dino_craters.yaml")
    parser.add_argument("--epochs",      type=int, default=None,
                        help="Override optim.epochs from the config")
    parser.add_argument("--batch_size",  type=int, default=None,
                        help="Override train.batch_size_per_gpu from the config")
    parser.add_argument("--no_resume",   action="store_true",
                        help="Don't resume from a checkpoint in output_dir even if one exists")
    parser.add_argument("--opts", nargs="*", default=[],
                        help="Extra Yacs-style overrides, e.g. dino.head_n_prototypes=2048")
    args = parser.parse_args()
    main(args)
