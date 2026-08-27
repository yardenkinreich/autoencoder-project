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
import datetime
import json
import os

import torch

from src.models.dinov2.dinov2.utils.config import setup
from src.models.dinov2.dinov2.train.ssl_meta_arch import SSLMetaArch
from src.models.dinov2.dinov2.train.train import do_train, do_test


def _write_run_manifest(cfg, args):
    """Standardized run metadata, written alongside the checkpoint so the
    dashboard (src/dashboard/model_meta.py) can read structured params
    directly instead of re-parsing config.yaml. Purely additive - config.yaml
    (written earlier by DINOv2's own write_config()) stays in place as-is;
    this just normalizes the same information into the schema MAE/CAE runs
    also write, so all three architectures look the same to the dashboard."""
    pretrained = (cfg.student.pretrained_weights or cfg.MODEL.WEIGHTS) or None
    # Same cfg.get("proto", None) convention as ssl_meta_arch.py's own read of
    # this section - absent entirely for any config without a "proto:"
    # section (e.g. dino_craters.yaml), so this stays None/False for those
    # runs instead of raising. Recorded here so the dashboard/run history can
    # distinguish a prototypical-loss finetune from a plain one without
    # re-parsing config.yaml.
    proto_cfg = cfg.get("proto", None)
    proto_enabled = bool(proto_cfg) and proto_cfg.get("enabled", False)
    manifest = {
        "schema_version": 1,
        "family": "dino",
        "source": "finetune" if pretrained else "scratch",
        "pretrained_weights": pretrained,
        "epochs": cfg.optim.epochs,
        # DINOv2 trains in ITERATIONS, not epoch-bounded passes over the
        # dataset (the dataset is sampled infinitely) - "epochs" here is a
        # config-defined scaling label (OFFICIAL_EPOCH_LENGTH iterations per
        # "epoch"), not the literal MAE/CAE sense of one full dataset pass.
        # Recording the actual iteration count alongside it so a run showing
        # "epochs: 2" isn't misread as comparable training volume to an
        # MAE/CAE run at epochs: 2 - see configs/dino_craters.yaml's comment.
        "iterations": cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH,
        "mask_ratio": None,
        "latent_dim": None,
        "num_samples": None,
        "diameter_range": ([args.min_diameter, args.max_diameter]
                           if args.min_diameter is not None and args.max_diameter is not None
                           else None),
        "data_tag": args.data_tag,
        # train_dino.py always trains at the wider ~1.0x-diameter crop
        # radius (see Snakefile's DINO_CLEAN_OFFSET / preprocess_2.py's
        # --clean_offset) - unlike dashboard/model_meta.py's parsers, which
        # have to infer this from family since they read older runs that
        # predate this field, a freshly-written manifest can just state it
        # directly, whether this is the finetune config or the from-scratch
        # one (both always use the same wide-FOV data convention).
        "fov": 1.0,
        "proto_loss_enabled": proto_enabled,
        "proto_loss_weight": proto_cfg.get("loss_weight") if proto_enabled else None,
        "proto_n_support_per_class": proto_cfg.get("n_support_per_class") if proto_enabled else None,
        "proto_split_csv": proto_cfg.get("split_csv") if proto_enabled else None,
        "param_source": "run_manifest",
        "checkpoint_path": os.path.join(args.output_dir, "eval", "final", "teacher_checkpoint.pth"),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "written_by": "train_dino.py",
    }
    json.dump(manifest, open(os.path.join(args.output_dir, "run_manifest.json"), "w"),
              indent=2, default=float)
    print(f"Run manifest → {args.output_dir}/run_manifest.json")


_DINO_LOSS_COMPONENT_COLORS = {
    "dino_local_crops_loss": "tab:orange", "dino_global_crops_loss": "tab:green",
    "koleo_loss": "tab:red", "ibot_loss": "tab:purple", "proto_loss": "tab:brown",
}


def plot_dino_loss(output_dir: str) -> dict:
    """Loss curves from training_metrics.json (JSONL, one line per ~10
    iterations, DINOv2's own logger - see do_train()).

    Writes TWO kinds of plot, for two different dashboard uses:
      - loss_curve.png: total_loss ONLY, at output_dir's top level (same
        name/location as before, so every existing caller - logs_registry.py,
        1_Compare_Runs.py - keeps working unmodified). This is deliberately
        just the one line: overlaying every run's full component set when
        comparing runs side by side used to be actively misleading (a proto
        run has a proto_loss line a plain finetune never will, so the old
        combined plot wasn't even comparing the same thing run to run) - the
        total is the one number every DINO run shares.
      - loss_components/{key}.png: one plot per loss TERM that actually
        appears in this run's own log (dino_local_crops_loss,
        dino_global_crops_loss, koleo_loss, ibot_loss, and proto_loss only
        for a proto.enabled run - see ssl_meta_arch.py) - for the single-run
        deep dive, where seeing each term's own trend (very different scales -
        koleo hovers near 0, dino_local_crops_loss is ~7) matters more than a
        cross-run-comparable single number.

    Returns {"total": path_or_None, "components": {key: path}} - total is
    None if training_metrics.json isn't there (e.g. training crashed before
    the first log line) or is empty; components is {} in the same case."""
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    if not os.path.exists(metrics_path):
        return {"total": None, "components": {}}
    rows = [json.loads(line) for line in open(metrics_path) if line.strip()]
    if not rows:
        return {"total": None, "components": {}}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [r["iteration"] for r in rows]
    total = [r["total_loss"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(iters, total, label="total_loss", color="tab:blue", linewidth=2)
    plt.text(iters[-1], total[-1], f"{total[-1]:.4f}", color="tab:blue",
             ha="right", va="bottom", fontsize=9)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("DINO Training Loss — total")
    plt.tight_layout()
    total_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(total_path)
    plt.close()
    print(f"Loss plot (total)      → {total_path}")

    component_paths = {}
    components_dir = os.path.join(output_dir, "loss_components")
    # any(...) over every row, not just rows[0]: MetricLogger's meters dict
    # (dinov2/logging/helpers.py) can lag a key's first appearance by a few
    # dump cycles - confirmed directly on a real proto run's own
    # training_metrics.json, proto_loss is absent from rows 0-1 but present
    # from row 63/126 onward, so a rows[0]-only check silently dropped its
    # entire plot even though the vast majority of rows have it.
    for key, color in _DINO_LOSS_COMPONENT_COLORS.items():
        if not any(key in r for r in rows):
            continue
        os.makedirs(components_dir, exist_ok=True)
        values = [r.get(key, float("nan")) for r in rows]
        plt.figure(figsize=(8, 5))
        plt.plot(iters, values, color=color, linewidth=2)
        plt.text(iters[-1], values[-1], f"{values[-1]:.4f}", color=color,
                 ha="right", va="bottom", fontsize=9)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"DINO Training Loss — {key}")
        plt.tight_layout()
        comp_path = os.path.join(components_dir, f"{key}.png")
        plt.savefig(comp_path)
        plt.close()
        component_paths[key] = comp_path
    if component_paths:
        print(f"Loss plots (components) → {components_dir}/ ({', '.join(component_paths)})")

    return {"total": total_path, "components": component_paths}


def main(args):
    opts = list(args.opts)
    opts.append(f"train.dataset_path=Craters:root={args.input}")
    if args.epochs is not None:
        opts.append(f"optim.epochs={args.epochs}")
    if args.batch_size is not None:
        opts.append(f"train.batch_size_per_gpu={args.batch_size}")
    if args.pretrained_weights is not None:
        opts.append(f"student.pretrained_weights={args.pretrained_weights}")

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

    # do_train() only saves via FSDPCheckpointer, whose LOCAL_STATE_DICT
    # format is for resuming training on the same sharding topology, not
    # portable loading elsewhere (see src/models/dino_backbone.py). Always
    # save one clean, inference-ready checkpoint at the end, using the same
    # method DINOv2's own do_test() uses periodically during training.
    do_test(cfg, model, "final")

    plot_dino_loss(args.output_dir)
    _write_run_manifest(cfg, args)


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
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Warm-start the student backbone from this checkpoint (e.g. "
                             "data/raw/pretrained/dinov2_vits14_pretrain.pth) - overrides "
                             "student.pretrained_weights from the config file. Architecture "
                             "(patch_size/in_chans/arch) must match the checkpoint or "
                             "load_state_dict will raise on shape mismatch.")
    parser.add_argument("--opts", nargs="*", default=[],
                        help="Extra Yacs-style overrides, e.g. dino.head_n_prototypes=2048")
    parser.add_argument("--min_diameter", type=float, default=None,
                        help="For run_manifest.json only - the crater diameter "
                             "range (km) this run's --input was filtered to")
    parser.add_argument("--max_diameter", type=float, default=None)
    parser.add_argument("--data_tag",     type=str,   default=None,
                        help="For run_manifest.json only - short label for "
                             "--input's data source")
    args = parser.parse_args()
    main(args)
