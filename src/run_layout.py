"""
run_layout.py — canonical on-disk location for a training run and for the
eval results of any checkpoint (trained or stock), shared by src/eval/
evaluate.py (which needs to know where to write new eval output) and
src/dashboard/migrate_logs.py (which needs the same computation to
reorganize existing runs). Kept as a standalone module, not under eval/ or
dashboard/, so neither package has to depend on the other for this.

    logs/{family}/{source}/{structure}/{frozen}/{data_source}/
         {crater_range}/{num_samples}/{other_metric}/{epochs}/
             eval_results/{eval_run_id}/          <- NOT "eval/" - DINOv2
                                                      itself already uses
                                                      that name inside a
                                                      DINO run's own
                                                      directory for its
                                                      internal checkpoint
                                                      snapshots (eval/final/,
                                                      eval/training_1999/)
"""
from __future__ import annotations
import os
import re

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_RANK_CKPT_RE = re.compile(r"^model_\d+\.rank_\d+\.pth$")


def slug(s: str) -> str:
    """Filesystem-safe token from an arbitrary string: last two path
    components (enough to be distinctive without the full path), extension
    stripped, unsafe chars collapsed to '_'."""
    s = s.rstrip("/")
    parts = [p for p in s.replace(os.sep, "/").split("/") if p]
    tail = "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else s)
    tail = re.sub(r"\.pth$", "", tail)
    return _SAFE_RE.sub("_", tail).strip("_") or "unknown"


def canonical_run_dir(manifest: dict, run_datetime: str | None = None) -> str:
    """The canonical logs/... directory a training run (or a synthesized
    stand-in for a stock/external checkpoint) belongs at, computed purely
    from its manifest fields - the same function used to physically
    reorganize logs/ (migrate_logs.py) and to decide where a new eval run's
    output nests (evaluate.py), so the two always agree.

    run_datetime: optional final path segment (e.g. "20260115-134702") -
    every hyperparameter field above it can be shared by more than one run
    (two runs can genuinely have identical family/source/epochs/etc, only
    differing in when they were trained), so this is what actually makes a
    run's own directory unique. Only migrate_logs.py passes this
    (derived from the run's own checkpoint mtime, since a real training
    run's directory always contains one); omitted by evaluate.py's
    stock-checkpoint stand-in path (that directory must stay the same
    stable location across every eval of the same stock checkpoint, not
    grow a new one per eval) and by the Snakefile's own live RUN_DIR
    computation (unchanged - out of scope for this retroactive reorg)."""
    family = manifest.get("family") or "legacy"
    source = manifest.get("source") or "unknown"

    structure = ("none" if source != "finetune" or not manifest.get("pretrained_weights")
                else slug(manifest["pretrained_weights"]))

    if source != "finetune":
        frozen = "na"
    else:
        fu = manifest.get("freeze_until")
        frozen = f"freeze{fu:g}" if fu is not None else "unknown"

    data_source = manifest.get("data_tag") or "untagged"

    dr = manifest.get("diameter_range")
    crater_range = f"d{dr[0]:g}-{dr[1]:g}" if dr else "drange_unknown"

    n = manifest.get("num_samples")
    num_samples = f"n{n}" if n is not None else "nunknown"

    if family == "mae":
        mr = manifest.get("mask_ratio")
        other_metric = f"mask{mr:g}" if mr is not None else "maskunknown"
    elif family == "cae":
        ld = manifest.get("latent_dim")
        other_metric = f"lat{ld}" if ld is not None else "latunknown"
    else:
        other_metric = "na"

    ep = manifest.get("epochs")
    epochs = f"ep{ep}" if ep is not None else "epunknown"

    base = os.path.join("logs", family, source, structure, frozen,
                        data_source, crater_range, num_samples, other_metric, epochs)
    return os.path.join(base, run_datetime) if run_datetime else base


def find_training_run_dir(checkpoint_path: str) -> str | None:
    """The directory that identifies a checkpoint's training run, if it has
    one at all - None for a checkpoint that isn't under logs/ (a stock/
    external checkpoint, which was never trained by this pipeline and has no
    run directory of its own)."""
    normalized = checkpoint_path.replace(os.sep, "/")
    if not (normalized.startswith("logs/") or "/logs/" in normalized):
        return None

    parent = os.path.dirname(checkpoint_path)
    parent_name = os.path.basename(parent)
    grandparent = os.path.dirname(parent)

    if parent_name == "models":
        return grandparent  # MAE/CAE: {run_dir}/models/autoencoder.pth

    parts = parent.replace(os.sep, "/").split("/")
    if "eval" in parts:
        # DINO: {run_dir}/eval/<anything>/teacher_checkpoint.pth - DINOv2's
        # OWN checkpoint convention (final or periodic snapshots), distinct
        # from this module's eval_results/ for evaluation output
        eval_idx = parts.index("eval")
        return "/".join(parts[:eval_idx])

    if _RANK_CKPT_RE.match(os.path.basename(checkpoint_path)):
        return parent  # DINO's own per-rank resume checkpoints, unwrapped

    return parent  # orphan: checkpoint sits directly in its run's directory


def default_eval_out(checkpoint: str, autoencoder_model: str, run_id: str) -> str:
    """{training_run_dir}/eval_results/<run_id> for a logs/-resident
    checkpoint; a synthesized, schema-consistent stand-in directory under
    logs/ for anything else (stock/external checkpoints). Shared by
    eval/evaluate.py (new eval runs) and dashboard/migrate_eval_runs.py
    (relocating pre-existing ones), so both always agree on where a given
    checkpoint's eval output belongs."""
    from dashboard import model_meta as MM  # deferred: only needed for the stock-checkpoint
                                             # branch, and avoids a module-load-time dependency
                                             # from this neutral module back onto dashboard/
    run_dir = find_training_run_dir(checkpoint)
    if run_dir is None:
        run_dir = canonical_run_dir(MM.extract_model_params(checkpoint, autoencoder_model))
    return os.path.join(run_dir, "eval_results", run_id)
