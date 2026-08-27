"""
migrate_eval_runs.py — one-time move of eval run output from the old flat
runs/eval/<run_id>/ layout into its new home nested inside the evaluated
checkpoint's own directory under logs/ (or a synthesized stand-in for
stock/external checkpoints) - see run_layout.py / evaluate.py's
default_eval_out(). Also relocates runs/eval_history.csv to
logs/eval_history.csv.

Dry-run by default - prints the full old->new mapping, touches nothing.
Pass --apply to actually move anything.

Usage:
    PYTHONPATH=src python -m dashboard.migrate_eval_runs               # dry-run
    PYTHONPATH=src python -m dashboard.migrate_eval_runs --apply        # execute
"""
from __future__ import annotations
import argparse
import json
import os
import shutil

from run_layout import default_eval_out


def find_eval_runs(old_eval_dir: str = "runs/eval") -> list[dict]:
    """[{old_path, checkpoint, autoencoder_model, run_id}, ...] for every
    eval run under the old flat runs/eval/ layout."""
    found = []
    if not os.path.isdir(old_eval_dir):
        return found
    for name in sorted(os.listdir(old_eval_dir)):
        d = os.path.join(old_eval_dir, name)
        summary_path = os.path.join(d, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        summary = json.load(open(summary_path))
        found.append({
            "old_path": d,
            "checkpoint": summary["checkpoint"],
            "autoencoder_model": summary["autoencoder_model"],
            "run_id": summary.get("run_id", name),
        })
    return found


def build_mapping(old_eval_dir: str = "runs/eval") -> list[dict]:
    rows = []
    for run in find_eval_runs(old_eval_dir):
        new_path = default_eval_out(run["checkpoint"], run["autoencoder_model"], run["run_id"])
        rows.append({**run, "new_path": new_path})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-eval-dir", default="runs/eval")
    ap.add_argument("--old-history", default="runs/eval_history.csv")
    ap.add_argument("--new-history", default="logs/eval_history.csv")
    ap.add_argument("--apply", action="store_true",
                    help="actually move files (default: dry-run report only)")
    args = ap.parse_args()

    rows = build_mapping(args.old_eval_dir)
    if not rows:
        print(f"No eval runs found under {args.old_eval_dir}.")
    else:
        print(f"{'OLD PATH':<45} NEW PATH")
        for r in rows:
            print(f"{r['old_path']:<45} -> {r['new_path']}")

        dest_counts: dict[str, list[str]] = {}
        for r in rows:
            dest_counts.setdefault(r["new_path"], []).append(r["old_path"])
        collisions = [f"{d} <- {s}" for d, s in dest_counts.items() if len(s) > 1
                     or os.path.exists(d)]
        if collisions:
            print("\n!!! COLLISIONS - aborting, nothing moved:")
            for c in collisions:
                print(f"  {c}")
            return

    history_exists = os.path.exists(args.old_history)
    print(f"\nhistory: {args.old_history} -> {args.new_history}"
         f"{'' if history_exists else '  (source does not exist, skipping)'}")

    if not args.apply:
        print(f"\n{len(rows)} eval run(s) would move. Dry-run only - pass --apply to execute.")
        return

    for r in rows:
        os.makedirs(os.path.dirname(r["new_path"]), exist_ok=True)
        shutil.move(r["old_path"], r["new_path"])
        print(f"moved: {r['old_path']} -> {r['new_path']}")

    if history_exists:
        os.makedirs(os.path.dirname(args.new_history) or ".", exist_ok=True)
        if os.path.exists(args.new_history):
            import pandas as pd
            merged = pd.concat([pd.read_csv(args.new_history), pd.read_csv(args.old_history)],
                               ignore_index=True)
            merged.to_csv(args.new_history, index=False)
            os.remove(args.old_history)
            print(f"merged {args.old_history} into existing {args.new_history}")
        else:
            shutil.move(args.old_history, args.new_history)
            print(f"moved: {args.old_history} -> {args.new_history}")

    print(f"\n{len(rows)} eval run(s) moved.")


if __name__ == "__main__":
    main()
