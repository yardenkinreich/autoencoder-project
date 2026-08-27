"""
logs_registry.py — discover trained checkpoints directly under logs/, so the
dashboard shows every trained model (whether or not it has ever been run
through eval.evaluate), not just the ones with an eval_results/ entry.

Two checkpoint conventions (see Snakefile):
    MAE/CAE: {RUN_DIR}/models/autoencoder.pth
    DINO:    {RUN_DIR}/eval/final/teacher_checkpoint.pth
"""
from __future__ import annotations
import os


def _architecture_from_path(path: str) -> str | None:
    parts = path.replace(os.sep, "/").split("/")
    if "logs" not in parts:
        return None
    arch = parts[parts.index("logs") + 1] if parts.index("logs") + 1 < len(parts) else None
    return arch if arch in ("mae", "cae", "dino") else None


def discover_checkpoints(logs_dir: str = "logs") -> list[dict]:
    """Every trained checkpoint found under logs/, evaluated or not.
    Returns dicts with: checkpoint, run_dir (the training run's directory -
    parent of models/ or of eval/final/), architecture."""
    found = []
    if not os.path.isdir(logs_dir):
        return found
    for dirpath, _dirnames, filenames in os.walk(logs_dir):
        if "autoencoder.pth" in filenames and os.path.basename(dirpath) == "models":
            run_dir = os.path.dirname(dirpath)
            ckpt = os.path.join(dirpath, "autoencoder.pth")
            found.append({
                "checkpoint": ckpt, "run_dir": run_dir,
                "architecture": _architecture_from_path(ckpt),  # None for pre-convention dirs
            })
        if "teacher_checkpoint.pth" in filenames and dirpath.replace(os.sep, "/").endswith("eval/final"):
            run_dir = os.path.dirname(os.path.dirname(dirpath))
            ckpt = os.path.join(dirpath, "teacher_checkpoint.pth")
            found.append({"checkpoint": ckpt, "run_dir": run_dir, "architecture": "dino"})
    return found


def training_artifacts(run_dir: str) -> dict:
    """Training-only artifacts available without ever running eval.evaluate:
    the loss curve and any plots left by the old, pre-eval-harness Snakefile
    clustering rules (encode_latents/display_clusters), which wrote bare
    PNGs into a sibling results/ dir - present for some older runs, not the
    current convention.

    Loss curve path differs by architecture's own directory convention:
    MAE/CAE write it under models/ (see train.py); DINO has no models/
    subdir, so train_dino.py's plot_dino_loss() writes it flat at run_dir's
    top level instead (see train_dino.py, training_metrics.json).

    loss_curve_png is ALWAYS total-loss-only, for the cross-run Compare Runs
    view (a proto run's loss components aren't comparable to a plain
    finetune's - see plot_dino_loss()'s docstring); loss_component_pngs
    (DINO only, {} for MAE/CAE and for any DINO run's whose training
    predates the loss_components/ split) is each individual term's own plot,
    for the single-run deep dive only."""
    loss_curve = os.path.join(run_dir, "models", "loss_curve.png")
    if not os.path.exists(loss_curve):
        loss_curve = os.path.join(run_dir, "loss_curve.png")
    components_dir = os.path.join(run_dir, "loss_components")
    loss_component_pngs = {}
    if os.path.isdir(components_dir):
        loss_component_pngs = {
            os.path.splitext(f)[0]: os.path.join(components_dir, f)
            for f in sorted(os.listdir(components_dir)) if f.lower().endswith(".png")
        }
    legacy_dir = os.path.join(run_dir, "results")
    legacy_images = []
    if os.path.isdir(legacy_dir):
        legacy_images = sorted(
            os.path.join(legacy_dir, f) for f in os.listdir(legacy_dir)
            if f.lower().endswith(".png")
        )
    return {
        "loss_curve_png": loss_curve if os.path.exists(loss_curve) else None,
        "loss_component_pngs": loss_component_pngs,
        "legacy_result_images": legacy_images,
    }
