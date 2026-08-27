"""
data.py — all dashboard data access. File-based only (eval_history.csv +
logs/**/eval_results/*/summary.json + configs/eval_suite.yaml) - no new
persistence layer, since those files already contain everything the
dashboard needs.
"""
from __future__ import annotations
import os
import json
import pandas as pd
import yaml
import streamlit as st

from . import model_meta as MM
from . import logs_registry as LR

HISTORY_PATH = "logs/eval_history.csv"
LOGS_DIR = "logs"
EVAL_SUITE_CFG = "configs/eval_suite.yaml"


@st.cache_data
def load_history(path: str = HISTORY_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def list_run_dirs(logs_dir: str = LOGS_DIR) -> list[str]:
    """Every eval run's output directory - nested arbitrarily deep as
    {training_run_dir}/eval_results/{eval_run_id}/ (or under a synthesized
    stand-in directory for stock/external checkpoints - see run_layout.py),
    so this needs a full tree walk rather than a flat listing."""
    if not os.path.isdir(logs_dir):
        return []
    dirs = []
    for dirpath, _dirnames, filenames in os.walk(logs_dir):
        if "summary.json" in filenames:
            dirs.append(dirpath)
    return sorted(dirs)


@st.cache_data
def load_run_summary(run_dir: str) -> dict:
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    # pre-generate_run_id runs have no stored run_id - fall back to the
    # directory name so every run on disk is still browsable.
    summary.setdefault("run_id", os.path.basename(run_dir.rstrip(os.sep)))
    return summary


def join_history_to_dirs(history_df: pd.DataFrame, run_dirs: list[str]) -> pd.DataFrame:
    """Match history rows to their eval output directory, preferring run_id
    but falling back to checkpoint path - runs from before generate_run_id()
    existed have a history run_id that was derived from the checkpoint (or
    is otherwise not the stored summary.json run_id, e.g. "src" for a path
    containing "/models/"), so a run_id-only join silently drops them even
    though their directory is perfectly findable by checkpoint."""
    records = [{"run_dir": d, "run_id": load_run_summary(d)["run_id"],
               "checkpoint": load_run_summary(d).get("checkpoint")} for d in run_dirs]
    dir_df = pd.DataFrame(records, columns=["run_dir", "run_id", "checkpoint"])
    if history_df.empty:
        return history_df.assign(run_dir=pd.Series(dtype=object))

    by_id = dir_df.drop_duplicates("run_id", keep="last").set_index("run_id")["run_dir"]
    by_checkpoint = (dir_df.dropna(subset=["checkpoint"])
                     .drop_duplicates("checkpoint", keep="last").set_index("checkpoint")["run_dir"])

    def _resolve(row) -> str | None:
        if row["run_id"] in by_id.index:
            return by_id[row["run_id"]]
        if row["checkpoint"] in by_checkpoint.index:
            return by_checkpoint[row["checkpoint"]]
        return None

    out = history_df.copy()
    out["run_dir"] = out.apply(_resolve, axis=1)
    return out


def load_labeled_set_artifacts(run_dir: str, set_name: str) -> dict:
    """Bundle one labeled set's already-written metrics/tables/plots for a
    run. Nothing is recomputed - this only reads what evaluate.py wrote."""
    base = os.path.join(run_dir, "labeled", set_name)

    def _csv(name):
        p = os.path.join(base, name)
        return pd.read_csv(p, index_col=0) if os.path.exists(p) else None

    def _png(name):
        p = os.path.join(base, name)
        return p if os.path.exists(p) else None

    metrics_path = os.path.join(base, "metrics.json")

    # Unsupervised over-clustering exploration (evaluate.py's
    # evaluate_labeled_set(), see visualize.py's plot_unsupervised_cluster_
    # dots/_images) - k in (10, 50) x technique in (pca, umap) x (dots,
    # images), any subset of which may be missing (an older eval run that
    # predates this, or a set too small for a given k - see evaluate.py's
    # `if target_k >= len(df): continue`). Keyed by (k, technique, kind) so
    # the dashboard can group/order them without parsing filenames.
    unsupervised_cluster_pngs = {}
    for k in (10, 50):
        for technique in ("pca", "umap"):
            for kind, suffix in [("dots", ""), ("images", "_images")]:
                p = _png(f"unsupervised_k{k}_{technique}{suffix}.png")
                if p:
                    unsupervised_cluster_pngs[(k, technique, kind)] = p

    return {
        "metrics": json.load(open(metrics_path)) if os.path.exists(metrics_path) else None,
        "confusion": _csv("confusion.csv"),
        "per_class": _csv("per_class.csv"),
        "confound_correlation": _csv("confound_correlation.csv"),
        "geo_correlation": _csv("geo_correlation.csv"),
        "reliability": _csv("reliability.csv"),
        "confusion_matrix_png": _png("confusion_matrix.png"),
        "clusters_by_sample_png": _png("clusters_by_sample.png"),
        "latent_separation_png": _png("latent_separation.png"),
        "latent_separation_umap_png": _png("latent_separation_umap.png"),
        "agreement_examples_png": _png("agreement_examples.png"),
        "unsupervised_cluster_pngs": unsupervised_cluster_pngs,
    }


# Suffix (everything after "{set_name}_") -> human label, shared by every
# labeled set plus "holdout" - see eval/history.py's flatten_results() for
# the authoritative list of suffixes a column can have.
METRIC_LABELS = {
    "n_samples": "N samples",
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "qwk": "Quadratic Weighted Kappa",
    "ordinal_mae": "Ordinal MAE",
    "map": "mAP",
    "ece": "ECE (calibration error)",
    "ari": "ARI (cluster agreement)",
    "nmi": "NMI (cluster agreement)",
    "v_measure": "V-measure (cluster agreement)",
    "purity": "Purity",
    "boundary_uncertainty": "Boundary uncertainty",
    "recon_mean_loss": "Reconstruction loss (mean)",
    "recon_std_loss": "Reconstruction loss (std)",
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin",
    "calinski_harabasz": "Calinski-Harabasz",
}

# Suffixes where a lower value is the better one - everything else in
# METRIC_LABELS is "higher is better".
LOWER_IS_BETTER_SUFFIXES = {
    "ece", "ordinal_mae", "boundary_uncertainty",
    "recon_mean_loss", "recon_std_loss", "davies_bouldin",
}


def metric_is_lower_better(col: str) -> bool:
    return any(col.endswith(f"_{suffix}") for suffix in LOWER_IS_BETTER_SUFFIXES)


def metric_label_map(groups: dict[str, list[str]]) -> dict[str, str]:
    """Column name -> human label, e.g. 'julie_accuracy' -> 'julie · Accuracy'.
    Keeps the column name as the underlying value everywhere (sorting,
    filtering, chart indexing) - this only relabels how it's displayed."""
    labels = {}
    for set_name, cols in groups.items():
        for col in cols:
            suffix = col[len(set_name) + 1:]
            labels[col] = f"{set_name} · {METRIC_LABELS.get(suffix, suffix)}"
    return labels


def metric_groups(history_df: pd.DataFrame, eval_suite_cfg: str = EVAL_SUITE_CFG) -> dict[str, list[str]]:
    """Split history columns by {set_name}_/holdout_ prefix, driven by
    eval_suite.yaml's labeled_sets keys rather than hardcoded - so a second
    labeled set shows up automatically once it's wired into the config."""
    set_names = []
    if os.path.exists(eval_suite_cfg):
        cfg = yaml.safe_load(open(eval_suite_cfg)) or {}
        set_names = list((cfg.get("labeled_sets") or {}).keys())

    groups: dict[str, list[str]] = {}
    for name in set_names:
        cols = [c for c in history_df.columns if c.startswith(f"{name}_")]
        if cols:
            groups[name] = cols
    holdout_cols = [c for c in history_df.columns if c.startswith("holdout_")]
    if holdout_cols:
        groups["holdout"] = holdout_cols
    return groups


@st.cache_data
def _cached_model_params(checkpoint: str, autoencoder_model: str) -> dict:
    return MM.extract_model_params(checkpoint, autoencoder_model)


@st.cache_data
def _discover_checkpoints_cached(logs_dir: str = "logs") -> list[dict]:
    return LR.discover_checkpoints(logs_dir)


def training_artifacts(training_run_dir: str) -> dict:
    return LR.training_artifacts(training_run_dir)


def build_registry() -> pd.DataFrame:
    """Every checkpoint the dashboard knows about, one row per checkpoint:
    every trained model found under logs/ (evaluated or not) plus every
    checkpoint referenced in eval_history.csv that isn't under logs/ (e.g.
    stock pretrained checkpoints). This is what powers Run Deep Dive's
    picker - eval_history.csv/the eval_results/ dirs alone under-report
    what's actually been trained."""
    logs_entries = _discover_checkpoints_cached()
    logs_df = pd.DataFrame(logs_entries, columns=["checkpoint", "run_dir", "architecture"])

    history_df = load_history()
    run_dirs = list_run_dirs()
    joined = join_history_to_dirs(history_df, run_dirs)

    eval_by_checkpoint: dict[str, dict] = {}
    if "checkpoint" in joined.columns:
        for _, row in joined.iterrows():
            eval_by_checkpoint[row["checkpoint"]] = {
                "run_id": row["run_id"], "eval_run_dir": row.get("run_dir"),
                # eval_history.csv's own autoencoder_model - the only source
                # of architecture for a checkpoint outside logs/ (e.g. a
                # stock pretrained one), since discover_checkpoints() never
                # finds it and _architecture_from_path() only recognizes
                # the logs/{family}/... convention.
                "autoencoder_model": row.get("autoencoder_model"),
            }

    rows = []
    seen = set()
    for _, r in logs_df.iterrows():
        ckpt = r["checkpoint"]
        seen.add(ckpt)
        ev = eval_by_checkpoint.get(ckpt)
        rows.append({
            "checkpoint": ckpt,
            "training_run_dir": r["run_dir"],
            "architecture": r["architecture"],
            "has_eval": ev is not None,
            "eval_run_dir": ev["eval_run_dir"] if ev else None,
            "run_id": ev["run_id"] if ev else None,
        })
    for ckpt, ev in eval_by_checkpoint.items():
        if ckpt in seen:
            continue
        rows.append({
            "checkpoint": ckpt, "training_run_dir": None, "architecture": ev.get("autoencoder_model"),
            "has_eval": True, "eval_run_dir": ev["eval_run_dir"], "run_id": ev["run_id"],
        })
    return pd.DataFrame(rows, columns=["checkpoint", "training_run_dir", "architecture",
                                       "has_eval", "eval_run_dir", "run_id"])


def enrich_history_with_params(history_df: pd.DataFrame) -> pd.DataFrame:
    """Join extracted training-parameter fields (family, source, epochs,
    mask_ratio, ...) onto the history DataFrame in memory - eval_history.csv
    itself stays untouched (a lightweight scalar-only index)."""
    if history_df.empty:
        return history_df
    records = [
        _cached_model_params(row.checkpoint, row.autoencoder_model)
        for row in history_df.itertuples()
    ]
    params_df = pd.DataFrame(records).add_prefix("param_")
    return pd.concat([history_df.reset_index(drop=True), params_df], axis=1)


def enrich_registry_with_params(registry_df: pd.DataFrame) -> pd.DataFrame:
    """Same as enrich_history_with_params, but for build_registry()'s output,
    which names its architecture column "architecture" rather than
    "autoencoder_model" (it covers untrained-eval checkpoints too, so it
    can't reuse the history schema outright)."""
    if registry_df.empty:
        return registry_df
    records = [
        _cached_model_params(row.checkpoint, row.architecture)
        for row in registry_df.itertuples()
    ]
    params_df = pd.DataFrame(records).add_prefix("param_")
    return pd.concat([registry_df.reset_index(drop=True), params_df], axis=1)
