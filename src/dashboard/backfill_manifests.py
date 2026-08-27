"""
backfill_manifests.py — one-time (idempotent, safe to re-run) generator of
run_manifest.json for every existing training run under logs/, using the
best signal already available (config.yaml / Snakefile.snapshot / dirname
regex, via model_meta.extract_model_params()) plus hand-confirmed
corrections from legacy_run_overrides.py.

Purely additive: only ever writes a new run_manifest.json where one doesn't
already exist. Nothing is moved or deleted - that's migrate_logs.py's job,
and it only runs after this has given every run a manifest to plan from.

Usage: PYTHONPATH=src python -m dashboard.backfill_manifests [--logs-dir logs]
"""
from __future__ import annotations
import argparse
import datetime
import json
import os

from . import model_meta as MM
from . import logs_registry as LR
from . import legacy_run_overrides as OVERRIDES

_RANK_CKPT_RE = __import__("re").compile(r"^model_\d+\.rank_\d+\.pth$")


def _run_dir_for_checkpoint(ckpt_path: str) -> str:
    """Map a checkpoint file to the directory that identifies its training
    run - broader than logs_registry.discover_checkpoints()'s convention so
    orphan checkpoints and non-`final` DINO checkpoints aren't skipped."""
    parent = os.path.dirname(ckpt_path)
    parent_name = os.path.basename(parent)
    grandparent = os.path.dirname(parent)

    if parent_name == "models":
        return grandparent  # MAE/CAE: {run_dir}/models/autoencoder.pth

    parts = parent.replace(os.sep, "/").split("/")
    if "eval" in parts:
        # DINO: {run_dir}/eval/<anything>/teacher_checkpoint.pth (final or
        # periodic, e.g. eval/training_1999/teacher_checkpoint.pth)
        eval_idx = parts.index("eval")
        return "/".join(parts[:eval_idx])

    if _RANK_CKPT_RE.match(os.path.basename(ckpt_path)):
        return parent  # DINO's own per-rank resume checkpoints, unwrapped

    return parent  # orphan: checkpoint sits directly in its run's directory


def iter_run_dirs(logs_dir: str = "logs") -> dict[str, str]:
    """{run_dir: representative_checkpoint_path} for every training run
    found under logs_dir, how ever its checkpoint(s) are laid out."""
    by_run_dir: dict[str, list[str]] = {}
    for dirpath, _dirnames, filenames in os.walk(logs_dir):
        # tests/edge_cases/* holds checkpoints from a run's own edge-case
        # test suite (TEST_DIR in the Snakefile), not independent runs -
        # without this, each one gets misread as its own separate run.
        if f"{os.sep}tests{os.sep}" in f"{dirpath}{os.sep}":
            continue
        for fname in filenames:
            if not fname.endswith(".pth"):
                continue
            ckpt = os.path.join(dirpath, fname)
            run_dir = _run_dir_for_checkpoint(ckpt)
            by_run_dir.setdefault(run_dir, []).append(ckpt)

    representative = {}
    for run_dir, ckpts in by_run_dir.items():
        preferred = [c for c in ckpts if os.path.basename(c) in
                    ("autoencoder.pth", "teacher_checkpoint.pth")]
        representative[run_dir] = sorted(preferred or ckpts)[0]
    return representative


def _family_hint_for(run_dir: str) -> str | None:
    """Substring match on the "/"-normalized-to-"_" form, not just a
    startswith prefix check on the live path - a run this hint applies to
    may already have been physically migrated once (e.g. re-running backfill
    after a manifest correction), in which case its *current* path no longer
    starts with the original prefix at all, but the original identifying
    fragment survives inside the disambiguation suffix migrate_logs.py
    appends (e.g. "...__cae_facebook_vit-mae-large_0_50_20"). Without this,
    the hint silently stops applying the moment the run moves once, and its
    own (confirmed wrong) Snakefile.snapshot value wins again."""
    normalized = run_dir.replace(os.sep, "/").replace("/", "_")
    for prefix, hint in OVERRIDES.FAMILY_HINTS:
        # strip a leading logs/ - it won't appear inside a post-move
        # disambiguation suffix, only the distinctive tail does
        clean_prefix = prefix.replace(os.sep, "/").strip("/")
        clean_prefix = clean_prefix.removeprefix("logs/")
        normalized_prefix = clean_prefix.replace("/", "_")
        if normalized_prefix in normalized:
            return hint
    return None


def build_manifest(run_dir: str, checkpoint: str) -> dict:
    # None (not defaulted to "mae") when the path itself doesn't say -
    # extract_model_params()/parse_run_dirname() both handle that correctly
    # by leaving family unknown rather than guessing.
    architecture = LR._architecture_from_path(checkpoint)
    family_hint = _family_hint_for(run_dir)
    record = MM.extract_model_params(checkpoint, architecture, family_hint=family_hint)
    record.update(OVERRIDES.FIELD_OVERRIDES.get(run_dir, {}))
    record.update({
        "schema_version": 1,
        "checkpoint_path": checkpoint,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "written_by": "backfill_manifests.py",
    })
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written without writing anything")
    args = ap.parse_args()

    run_dirs = iter_run_dirs(args.logs_dir)
    written, skipped = 0, 0
    for run_dir, checkpoint in sorted(run_dirs.items()):
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        if os.path.exists(manifest_path):
            skipped += 1
            continue
        manifest = build_manifest(run_dir, checkpoint)
        print(f"{'[dry-run] ' if args.dry_run else ''}{run_dir}: "
              f"family={manifest['family']} source={manifest['source']} "
              f"param_source={manifest['param_source']}")
        if not args.dry_run:
            json.dump(manifest, open(manifest_path, "w"), indent=2, default=float)
        written += 1

    print(f"\n{written} manifest(s) {'would be ' if args.dry_run else ''}written, "
         f"{skipped} run(s) already had one.")


if __name__ == "__main__":
    main()
