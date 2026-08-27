"""
history.py — persistent run history so eval scores from different model
runs/architectures can be compared, not just viewed one at a time.

Every `evaluate.py` invocation appends one row to a CSV (default
logs/eval_history.csv) with the headline metrics from every labeled set it
was run against, plus the holdout set's numbers. One row per run, one file
to diff/sort/filter across your whole project.
"""
from __future__ import annotations
import os
import hashlib
import datetime
import pandas as pd


def _legacy_run_id(checkpoint: str) -> str:
    """
    Short, comparable label for a checkpoint path, e.g.
    logs/mae/wac100m_sigma100_noband/d3-10_ep50_mask0.75_n25000/models/autoencoder.pth
    -> mae/wac100m_sigma100_noband/d3-10_ep50_mask0.75_n25000
    Falls back to the raw path if it doesn't match the usual logs/ layout
    (e.g. a checkpoint outside logs/, or an unfamiliar suffix). Only used
    as a fallback when a run has no generated run_id (pre-generate_run_id
    runs) - see generate_run_id() for current run identity.
    """
    p = checkpoint.replace("logs/", "", 1) if checkpoint.startswith("logs/") else checkpoint
    for suffix in ("/models/", "/eval/"):
        if suffix in p:
            return p.split(suffix)[0]
    return p


def generate_run_id(checkpoint: str, autoencoder_model: str,
                    when: datetime.datetime | None = None) -> str:
    """
    Stable, filesystem-safe, sortable run identity: architecture + start
    time + a short hash of the checkpoint path. The timestamp keeps repeat
    evals of the same checkpoint distinct; the hash ties the id back to the
    checkpoint even if the output directory gets renamed later.
    """
    when = when or datetime.datetime.now()
    digest = hashlib.sha1(checkpoint.encode()).hexdigest()[:8]
    return f"{autoencoder_model}_{when:%Y%m%d-%H%M%S}_{digest}"


def flatten_results(all_results: dict) -> dict:
    """One flat dict per run, ready for a CSV row. Every labeled set's
    headline metrics get a `{set_name}_` prefix so the schema stays
    extensible as sets are added/removed across runs."""
    row = {
        "timestamp": all_results.get("run_timestamp")
                     or datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": all_results.get("run_id")
                  or _legacy_run_id(all_results["checkpoint"]),
        "checkpoint": all_results["checkpoint"],
        "autoencoder_model": all_results["autoencoder_model"],
        "tag": all_results.get("tag"),
    }

    for name, r in all_results.get("labeled_sets", {}).items():
        row[f"{name}_n_samples"] = r["n_samples"]
        row[f"{name}_accuracy"] = r["accuracy"]["point"]
        row[f"{name}_macro_f1"] = r["macro_f1"]["point"]
        row[f"{name}_qwk"] = r["quadratic_weighted_kappa"]["point"]
        row[f"{name}_ordinal_mae"] = r["ordinal_mae"]["point"]
        row[f"{name}_map"] = r["map"]["point"] if r.get("map") else None
        row[f"{name}_ece"] = r["ece"]
        row[f"{name}_ari"] = r["agreement"]["ari"]
        row[f"{name}_nmi"] = r["agreement"]["nmi"]
        row[f"{name}_v_measure"] = r["agreement"]["v_measure"]
        row[f"{name}_purity"] = r["alignment"]["purity"]
        row[f"{name}_boundary_uncertainty"] = r["boundary_uncertainty"]["boundary_uncertainty"]

    holdout = all_results.get("holdout")
    if holdout:
        rl = holdout.get("reconstruction_loss")
        if rl:
            row["holdout_recon_mean_loss"] = rl["mean_loss"]
            row["holdout_recon_std_loss"] = rl["std_loss"]
        q = holdout["cluster_quality"]
        row["holdout_silhouette"] = q["silhouette"]
        row["holdout_davies_bouldin"] = q["davies_bouldin"]
        row["holdout_calinski_harabasz"] = q["calinski_harabasz"]

    return row


def append_to_history(all_results: dict, history_path: str = "logs/eval_history.csv") -> None:
    row = flatten_results(all_results)
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(history_path):
        # union of columns across all runs so far — a run without e.g. a
        # "new_set" doesn't crash appending onto a history that has one
        existing = pd.read_csv(history_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(history_path, index=False)
    print(f"Appended to {history_path} ({len(df)} runs total)")


def best_runs(metric: str, history_path: str = "logs/eval_history.csv",
             top_n: int = 10, minimize: bool = False) -> pd.DataFrame:
    """
    Top N runs by a metric column (e.g. "julie_accuracy", "holdout_silhouette",
    "holdout_recon_mean_loss"). minimize=True for metrics where lower is
    better (loss, davies_bouldin, ece).

    Deduped to one row per checkpoint (the most recent eval of it) before
    ranking - the same convention 1_Compare_Runs.py's own table already uses
    (sort_values("timestamp").drop_duplicates("checkpoint", keep="last")).
    A checkpoint evaluated repeatedly (e.g. the stock DINO checkpoint, run
    7 times across this project so far) would otherwise compete against
    ITSELF for top_n slots with its own older, possibly noisier or
    since-fixed scores - burying or crowding out its current, most
    trustworthy evaluation instead of ranking it once, on its latest number.
    """
    df = pd.read_csv(history_path)
    if metric not in df.columns:
        raise ValueError(f"'{metric}' not in history columns: {sorted(df.columns)}")
    df = df.dropna(subset=[metric])
    df = df.sort_values("timestamp").drop_duplicates("checkpoint", keep="last")
    return df.sort_values(metric, ascending=minimize).head(top_n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Show the best runs so far by a metric")
    ap.add_argument("--metric", required=True)
    ap.add_argument("--history", default="logs/eval_history.csv")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--minimize", action="store_true",
                    help="lower is better for this metric (loss, davies_bouldin, ece)")
    ap.add_argument("--columns", nargs="*", default=None,
                    help="extra columns to show besides run_id/timestamp/metric")
    args = ap.parse_args()

    top = best_runs(args.metric, args.history, args.top, args.minimize)
    cols = ["run_id", "timestamp", args.metric] + (args.columns or [])
    cols = [c for c in cols if c in top.columns]
    print(top[cols].to_string(index=False))
