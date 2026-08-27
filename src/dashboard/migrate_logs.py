"""
migrate_logs.py — physically reorganize logs/ into one consistent tree,
grouped by training metadata (from run_manifest.json, written by Phase A/B):

    logs/{family}/{source}/{structure}/{frozen}/{data_source}/
         {crater_range}/{num_samples}/{other_metric}/{epochs}/{run_datetime}/

run_datetime (the run's own checkpoint mtime) is the actual leaf - every
field above it can be shared by more than one run (two runs can genuinely
be identical in every recorded hyperparameter, only differing in when they
were trained), so it's what makes each run's own directory unique.

Dry-run by default - prints the full old->new mapping and a byte/
param_source summary, touches nothing. Pass --apply to actually move
anything, only after reviewing the dry-run output.

Usage:
    PYTHONPATH=src python -m dashboard.migrate_logs               # dry-run
    PYTHONPATH=src python -m dashboard.migrate_logs --apply        # execute
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import re
import shutil

from run_layout import canonical_run_dir, slug as _slug

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# kept as a local alias - this module's own logic below calls target_path()
# throughout; the actual computation now lives in run_layout.py so
# evaluate.py can compute the same canonical location for new eval output.
target_path = canonical_run_dir


def find_manifests(logs_dir: str = "logs") -> list[tuple[str, dict]]:
    """[(run_dir, manifest), ...] for every run_manifest.json under logs_dir."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(logs_dir):
        if "run_manifest.json" in filenames:
            manifest = json.load(open(os.path.join(dirpath, "run_manifest.json")))
            found.append((dirpath, manifest))
    return found


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def _checkpoint_mtime(run_dir: str) -> float | None:
    """Earliest autoencoder.pth/teacher_checkpoint.pth mtime found anywhere
    under run_dir - a real, distinguishing training-completion signal.
    run_manifest.json's own mtime isn't usable for this (often identical
    across a whole batch of runs backfilled/migrated together in one pass,
    which is exactly the case this needs to disambiguate)."""
    earliest = None
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        for f in filenames:
            if f in ("autoencoder.pth", "teacher_checkpoint.pth"):
                mtime = os.path.getmtime(os.path.join(dirpath, f))
                if earliest is None or mtime < earliest:
                    earliest = mtime
    return earliest


def _run_datetime_for(run_dir: str) -> str:
    """The datetime segment every migrated run's target path ends in - the
    run's own checkpoint mtime (a real fact about when it finished
    training), formatted short and readable. Falls back to a sanitized full
    old-path only if no checkpoint file can be found at all (shouldn't
    happen for a real trained run) - still unique, just uglier."""
    mtime = _checkpoint_mtime(run_dir)
    if mtime is not None:
        return datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d-%H%M%S")
    return _SAFE_RE.sub("_", run_dir.removeprefix("logs/").rstrip("/")).strip("_")


def _disambiguate(rows: list[dict]) -> None:
    """Every run's target already ends in its own checkpoint-mtime datetime
    (see build_mapping), so two DIFFERENT runs colliding here means their
    checkpoints happen to share the exact same to-the-second mtime (rare,
    e.g. a batch of runs saved in the same training script's loop) or one
    of them had no checkpoint file to derive a datetime from at all - append
    the old path too, just for whichever entries still collide after that.
    Mutates rows in place. Two passes: first against each other (in-batch),
    then against anything already sitting on disk from a prior migration
    pass."""
    by_dest: dict[str, list[dict]] = {}
    for r in rows:
        by_dest.setdefault(r["new_path"], []).append(r)
    for dest, group in by_dest.items():
        if len(group) < 2:
            continue
        for r in group:
            suffix = _SAFE_RE.sub("_", r["old_path"].removeprefix("logs/").rstrip("/")).strip("_")
            r["new_path"] = f"{dest}__{suffix}"
            r["disambiguated_from"] = dest

    for r in rows:
        if "disambiguated_from" not in r and os.path.exists(r["new_path"]):
            dest = r["new_path"]
            suffix = _SAFE_RE.sub("_", r["old_path"].removeprefix("logs/").rstrip("/")).strip("_")
            r["new_path"] = f"{dest}__{suffix}"
            r["disambiguated_from"] = dest


def build_mapping(logs_dir: str = "logs") -> list[dict]:
    """Rows for runs that actually need to move. Every target ends in the
    run's own checkpoint-mtime datetime (uniform across all runs, not just
    colliding ones - two runs can genuinely share every hyperparameter field
    above it), so collisions should be rare; _disambiguate() is still the
    fallback for the runs where they happen anyway.

    Safe to run repeatedly against a partially-migrated tree: a run whose
    *current* path already equals its freshly-recomputed target (datetime
    segment included - checkpoint mtime doesn't change, so this is stable
    across reruns), or that target plus a "__" disambiguation suffix, is
    recognized as already correctly placed and excluded - both from moving
    and from the disambiguation collision pool below.

    Skips any manifest with family=None entirely (confirmed 2026-08-23: the
    one real case is a pre-Snakefile-convention exploratory directory
    bundling two different checkpoints - autoencoder_cae.pth AND
    autoencoder_mae.pth - together, so it doesn't represent a single run and
    doesn't fit the one-checkpoint-per-directory model this whole migration
    assumes. Left exactly where it is rather than guessed at."""
    all_rows = []
    for run_dir, manifest in find_manifests(logs_dir):
        if manifest.get("family") is None:
            continue
        base_target = target_path(manifest, run_datetime=_run_datetime_for(run_dir))
        already_placed = (run_dir == base_target or run_dir.startswith(base_target + "__"))
        all_rows.append({
            "old_path": run_dir,
            "new_path": run_dir if already_placed else base_target,
            "already_placed": already_placed,
            "family": manifest.get("family") or "legacy",
            "param_source": manifest.get("param_source"),
            "bytes": _dir_size(run_dir),
        })
    pending = [r for r in all_rows if not r["already_placed"]]
    _disambiguate(pending)
    return pending


def _print_report(rows: list[dict]) -> list[str]:
    """Prints the dry-run report; returns collision descriptions (empty if none)."""
    print(f"{'OLD PATH':<75} NEW PATH")
    for r in sorted(rows, key=lambda r: r["new_path"]):
        print(f"{r['old_path']:<75} -> {r['new_path']}")

    disambiguated = [r for r in rows if "disambiguated_from" in r]
    if disambiguated:
        print(f"\n--- {len(disambiguated)} run(s) disambiguated (identical metadata, "
             f"original dir name appended so nothing was merged/lost) ---")
        by_original: dict[str, list[str]] = {}
        for r in disambiguated:
            by_original.setdefault(r["disambiguated_from"], []).append(r["old_path"])
        for dest, sources in by_original.items():
            print(f"  {dest}/ split into:")
            for s in sources:
                print(f"    {s}")

    print("\n--- by param_source ---")
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["param_source"]] = by_source.get(r["param_source"], 0) + 1
    for src, n in sorted(by_source.items()):
        print(f"  {src:20s} {n}")

    print("\n--- by family (bytes) ---")
    by_family: dict[str, int] = {}
    for r in rows:
        by_family[r["family"]] = by_family.get(r["family"], 0) + r["bytes"]
    for fam, b in sorted(by_family.items()):
        print(f"  {fam:10s} {b / 1e9:.2f} GB")

    dest_counts: dict[str, list[str]] = {}
    for r in rows:
        dest_counts.setdefault(r["new_path"], []).append(r["old_path"])
    collisions = []
    for dest, sources in dest_counts.items():
        if len(sources) > 1:
            collisions.append(f"{dest} <- {sources}")
        elif os.path.exists(dest):
            collisions.append(f"{dest} already exists on disk (source: {sources[0]})")
    return collisions


def apply_mapping(rows: list[dict], logs_dir: str = "logs") -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    audit_path = os.path.join(logs_dir, f".migration_log_{ts}.json")
    audit = []
    for r in rows:
        os.makedirs(os.path.dirname(r["new_path"]), exist_ok=True)
        shutil.move(r["old_path"], r["new_path"])
        audit.append({**r, "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
        print(f"moved: {r['old_path']} -> {r['new_path']}")
    json.dump(audit, open(audit_path, "w"), indent=2)
    print(f"\n{len(rows)} run(s) moved. Audit log: {audit_path}")
    _rewrite_references(rows, logs_dir)


def _rewrite_references(rows: list[dict], logs_dir: str = "logs") -> None:
    """Update any exact-matching checkpoint path in logs/eval_history.csv
    and every eval_results/*/summary.json after a move. Backs up touched
    files to .bak first. Eval results now nest INSIDE their checkpoint's own
    training run directory, so moving that directory already carries its
    eval_results/ along for free (shutil.move() moves the whole subtree) -
    what still needs fixing here is the "checkpoint" field *recorded inside*
    those already-moved summary.json files (and any eval_history.csv row),
    since the checkpoint's own path changed."""
    import glob
    import pandas as pd

    path_map = {}
    for r in rows:
        old_ckpt_prefix, new_ckpt_prefix = r["old_path"], r["new_path"]
        path_map[old_ckpt_prefix] = new_ckpt_prefix

    def _remap(checkpoint: str) -> str | None:
        for old_prefix, new_prefix in path_map.items():
            if checkpoint == old_prefix or checkpoint.startswith(old_prefix + os.sep):
                return new_prefix + checkpoint[len(old_prefix):]
        return None

    history_path = os.path.join(logs_dir, "eval_history.csv")
    n_history = 0
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        changed = False
        for i, row in df.iterrows():
            remapped = _remap(str(row["checkpoint"]))
            if remapped:
                df.at[i, "checkpoint"] = remapped
                changed = True
                n_history += 1
        if changed:
            shutil.copy(history_path, history_path + ".bak")
            df.to_csv(history_path, index=False)

    n_summaries = 0
    for summary_path in glob.glob(os.path.join(logs_dir, "**", "summary.json"), recursive=True):
        summary = json.load(open(summary_path))
        remapped = _remap(summary.get("checkpoint", ""))
        if remapped:
            shutil.copy(summary_path, summary_path + ".bak")
            summary["checkpoint"] = remapped
            json.dump(summary, open(summary_path, "w"), indent=2, default=float)
            n_summaries += 1

    print(f"Reference rewrite: {n_history} eval_history.csv row(s), "
         f"{n_summaries} summary.json file(s) updated.")


def _find_nested_run_dirs(rows: list[dict]) -> list[str]:
    """Detect a run_dir that is itself inside another run_dir (e.g. an
    orphan checkpoint's directory containing another orphan checkpoint
    further down its own tree - happened for real: logs/mae/
    processed_wac_100m_new both had its own autoencoder.pth AND a nested
    autoencoder.pth under processed_wac_100m_new/sigma/100). shutil.move()
    moves a whole subtree at once, so moving the outer one silently carries
    the inner one away with it, leaving the inner "run" both double-counted
    in the mapping and missing from its old location when its own turn
    comes up - this must be caught before any move happens, not mid-move."""
    paths = sorted(r["old_path"].rstrip("/") for r in rows)
    problems = []
    for i, p in enumerate(paths):
        for other in paths[i + 1:]:
            if other == p:
                continue
            if other.startswith(p + "/"):
                problems.append(f"{other} is nested inside {p}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--apply", action="store_true",
                    help="actually move files (default: dry-run report only)")
    args = ap.parse_args()

    if not find_manifests(args.logs_dir):
        print("No run_manifest.json files found - run backfill_manifests.py first.")
        return

    rows = build_mapping(args.logs_dir)
    if not rows:
        print("Every run is already at its canonical location - nothing to do.")
        return

    nested = _find_nested_run_dirs(rows)
    if nested:
        print("!!! NESTED RUN DIRECTORIES - aborting, nothing moved. Resolve manually "
             "(these can't be moved independently without one carrying the other along):")
        for n in nested:
            print(f"  {n}")
        return

    collisions = _print_report(rows)
    if collisions:
        print("\n!!! COLLISIONS - aborting, nothing moved:")
        for c in collisions:
            print(f"  {c}")
        return

    if args.apply:
        apply_mapping(rows, args.logs_dir)
    else:
        print(f"\n{len(rows)} run(s) would move. Dry-run only - pass --apply to execute.")


if __name__ == "__main__":
    main()
