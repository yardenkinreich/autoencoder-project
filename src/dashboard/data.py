"""
data.py — all dashboard data access. File-based only (eval_history.csv +
logs/**/eval_results/*/summary.json + configs/eval_suite.yaml) - no new
persistence layer, since those files already contain everything the
dashboard needs.
"""
from __future__ import annotations
import os
import re
import json
from datetime import datetime
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

    # Attention-map diagnostic (evaluate.py, DINO-family checkpoints only -
    # see eval/attention_maps.py) - one file per sampled crater, named
    # {crater_id}_class{true_label}.png. Listed (not a fixed key set like
    # the unsupervised-cluster ones above) since which craters got sampled
    # varies run to run.
    attention_map_dir = os.path.join(base, "attention_maps")
    attention_map_pngs = []
    if os.path.isdir(attention_map_dir):
        attention_map_pngs = sorted(
            os.path.join(attention_map_dir, f) for f in os.listdir(attention_map_dir)
            if f.lower().endswith(".png")
        )

    return {
        "metrics": json.load(open(metrics_path)) if os.path.exists(metrics_path) else None,
        "confusion": _csv("confusion.csv"),
        # supervised probes on the same set, 5-fold CV (eval/head.py):
        # raw_cv = k-means, lda_head_cv = linear probe, knn_cv = k-NN
        "head_comparison": _csv("head_comparison.csv"),
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
        "attention_map_pngs": attention_map_pngs,
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


def _GEO_ARTIFACT_FILES(k: int):
    """(key, filename) for everything eval/geographic_map.py and
    eval/cluster_profile.py write per k. One list, used by both discovery
    functions, so the two cannot drift."""
    return (
        ("clusters_csv", f"geographic_clusters_k{k}.csv"),
        ("gpkg", f"geographic_clusters_k{k}.gpkg"),
        ("qml", f"geographic_clusters_k{k}.qml"),
        ("profile_csv", f"cluster_profile_k{k}.csv"),
        ("colors_json", f"cluster_colors_k{k}.json"),
        ("ordinal_json", f"cluster_ordinal_k{k}.json"),
        ("mean_craters_png", f"cluster_mean_craters_k{k}.png"),
        ("feature_profile_png", f"cluster_feature_profile_k{k}.png"),
        # src/eval/pool_projection.py: the whole pool in PCA / UMAP, coloured
        # by THESE k-means clusters (same colours as the map)
        ("pca_kmeans_png", f"pca_kmeans_k{k}.png"),
        ("umap_kmeans_png", f"umap_kmeans_k{k}.png"),
        # Per-band clustering (src/eval/per_band_clustering.py): cluster INSIDE
        # each acquisition band, then match clusters across bands, so band
        # identity cannot be a feature. Two maps so the before/after is one
        # glance, plus its own GeoPackage for QGIS.
        ("per_band_json", f"per_band_clustering_k{k}.json"),
        ("per_band_map_global", f"per_band_map_global_k{k}.png"),
        ("per_band_map_matched", f"per_band_map_bandmatched_k{k}.png"),
        ("per_band_gpkg", f"per_band_clusters_bandmatched_k{k}.gpkg"),
        ("per_band_qml", f"per_band_clusters_bandmatched_k{k}.qml"),
        ("per_band_csv", f"per_band_clusters_k{k}.csv"),
        ("per_band_matches_png", f"per_band_matches_k{k}.png"),
        ("per_band_examples_png", f"per_band_examples_k{k}.png"),
        ("per_band_maps_byband_png", f"per_band_maps_byband_k{k}.png"),
        ("per_band_map_west", f"per_band_map_west_k{k}.png"),
        ("per_band_map_mid", f"per_band_map_mid_k{k}.png"),
        ("per_band_map_east", f"per_band_map_east_k{k}.png"),
        ("per_band_profile_west", f"per_band_profile_west_k{k}.png"),
        ("per_band_profile_mid", f"per_band_profile_mid_k{k}.png"),
        ("per_band_profile_east", f"per_band_profile_east_k{k}.png"),
        ("per_band_latent_png", f"per_band_latent_k{k}.png"),
        ("per_band_ownbasis_png", f"per_band_ownbasis_k{k}.png"),
        # one standalone own-basis PCA per band, so each band's tab can show
        # the PCA of the very clusters displayed above it
        ("per_band_ownbasis_west", f"per_band_ownbasis_west_k{k}.png"),
        ("per_band_ownbasis_mid", f"per_band_ownbasis_mid_k{k}.png"),
        ("per_band_ownbasis_east", f"per_band_ownbasis_east_k{k}.png"),
        ("per_band_gpkg_global", f"per_band_clusters_global_k{k}.gpkg"),
        ("per_band_qml", f"per_band_clusters_bandmatched_k{k}.qml"),
        ("per_band_gpkg_west", f"per_band_clusters_west_k{k}.gpkg"),
        ("per_band_gpkg_mid", f"per_band_clusters_mid_k{k}.gpkg"),
        ("per_band_gpkg_east", f"per_band_clusters_east_k{k}.gpkg"),
    )


def geographic_map_artifacts(eval_run_dir: str | None) -> dict:
    """Global geographic cluster-map artifacts (eval/geographic_map.py) for
    a checkpoint - a PNG plus its per-cluster summary table, if that script
    has been run for this checkpoint.

    Hung off the checkpoint's RUN dir rather than a specific eval run: the
    map is a property of the checkpoint (embed a big unlabeled pool, cluster
    it, plot it geographically), not of one eval.evaluate invocation over the
    labeled sets, so tying it to a single eval_results/<run_id>/ folder would
    make it vanish the next time that checkpoint is re-evaluated. eval_run_dir
    is always {run_dir}/eval_results/{run_id} (see run_layout.default_eval_out),
    so the run dir is two levels up - and that works for a stock/external
    checkpoint too, whose run dir is the synthesized stand-in run_layout
    builds for it rather than a real training directory."""
    if not eval_run_dir:
        return {"png": None, "summary": None, "run_dir": None}
    run_dir = os.path.dirname(os.path.dirname(eval_run_dir.rstrip(os.sep)))
    d = os.path.join(run_dir, "geographic_map")
    if not os.path.isdir(d):
        return {"by_k": {}, "run_dir": run_dir}
    # geographic_map.py writes one PNG (+ summary CSV) per requested k, named
    # geographic_map_k{k}.png / geographic_clusters_k{k}_summary.csv - keyed
    # by k here so the page can offer a k selector rather than guessing which
    # single one to show.
    by_k: dict[int, dict] = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("geographic_map_k") and f.endswith(".png")):
            continue
        stem = f[len("geographic_map_k"):-len(".png")]
        # "_panels" is the supplementary per-cluster panel figure for large k,
        # not a k of its own - attach it to its k rather than parsing as one.
        panels = stem.endswith("_panels")
        if panels:
            stem = stem[:-len("_panels")]
        try:
            k = int(stem)
        except ValueError:
            continue
        entry = by_k.setdefault(k, {"png": None, "panels_png": None,
                                    "summary": None, "clusters_csv": None,
                                    "gpkg": None, "qml": None,
                                    "profile_csv": None, "colors_json": None,
                                    "ordinal_json": None,
                                    "mean_craters_png": None,
                                    "feature_profile_png": None})
        if panels:
            entry["panels_png"] = os.path.join(d, f)
            continue
        entry["png"] = os.path.join(d, f)

        summary_path = os.path.join(d, f"geographic_clusters_k{k}_summary.csv")
        if os.path.exists(summary_path):
            try:
                entry["summary"] = pd.read_csv(summary_path)
            except Exception:
                entry["summary"] = None
        # Per-crater exports, offered as downloads: the GeoJSON is for QGIS
        # (points styled by the categorical `cluster` field), the CSV is the
        # same data flat. Paths only - never read into memory here, since a
        # 100k-crater GeoJSON is tens of MB and the dashboard caches
        # aggressively.
        # TWO GeoJSON variants, and the distinction matters: `projected` is in
        # the mosaic's own CRS and overlays it with no reprojection, which is
        # the only one that lands correctly (the WAC GeoTIFFs declare
        # DATUM["unknown"], so no GIS can transform lon/lat onto them).
        # `geographic` is lon/lat on D_Moon_2000, for use with other lunar
        # layers. Each ships zipped with its own .prj.
        # GeoPackage only: one file with the CRS embedded, already in the
        # mosaic's projection. The .qml beside it makes QGIS apply the same
        # per-cluster colours the figures use.
        for key, name in _GEO_ARTIFACT_FILES(k):
            p = os.path.join(d, name)
            if os.path.exists(p):
                entry[key] = p
    return {"by_k": by_k, "run_dir": run_dir}


def data_comparison_artifacts(eval_run_dir: str | None) -> dict:
    """THE SAME checkpoint evaluated on many different data presentations -
    every (source mosaic x high-pass filter x normalization) combination -
    plus the 100k-crater geographic maps for the leading ones.

    WHY THIS IS A SECTION AND NOT 30 REGISTRY ROWS
    ----------------------------------------------
    build_registry() is one row per CHECKPOINT: it keys on the checkpoint
    file path and the last eval wins. These 30 evaluations all share one
    frozen checkpoint and differ only in the data fed to it, so as registry
    rows they would collapse into a single entry showing whichever ran last.
    They are a property OF a checkpoint - "how does this model respond to
    each data presentation" - which is exactly what a Deep Dive section is
    for. Same reasoning, and the same layout, as geographic_map_artifacts().

    Layout, under {run_dir}/data_comparison/:
        eval/<catalog>/summary.json                     the metric suite
        eval/<catalog>/labeled/<catalog>__<norm>/*.png  per-cell figures
        geomap/<catalog>__<norm>/geographic_map_k*.png  100k-crater maps

    Returns paths and small tables only - never a figure or a 100k-row
    GeoJSON, which are read on demand."""
    if not eval_run_dir:
        return {"cells": {}, "geomap": {}, "run_dir": None}
    run_dir = os.path.dirname(os.path.dirname(eval_run_dir.rstrip(os.sep)))
    root = os.path.join(run_dir, "data_comparison")
    if not os.path.isdir(root):
        return {"cells": {}, "geomap": {}, "run_dir": run_dir}

    # ── the metric cells, one per (catalog, norm) ────────────────────────────
    cells: dict[tuple, dict] = {}
    eval_root = os.path.join(root, "eval")
    if os.path.isdir(eval_root):
        for cat in sorted(os.listdir(eval_root)):
            sj = os.path.join(eval_root, cat, "summary.json")
            if not os.path.exists(sj):
                continue
            try:
                summary = json.load(open(sj))
            except Exception:
                continue
            for set_name, s in (summary.get("labeled_sets") or {}).items():
                # set names are "<catalog>__<norm>"; fall back to the whole
                # name as the norm if that convention is ever broken, rather
                # than dropping the cell.
                _, _, norm = set_name.partition("__")
                figs_dir = os.path.join(eval_root, cat, "labeled", set_name)
                figs = {}
                if os.path.isdir(figs_dir):
                    for f in sorted(os.listdir(figs_dir)):
                        if f.endswith(".png"):
                            figs[f[:-4]] = os.path.join(figs_dir, f)
                cells[(cat, norm or set_name)] = {
                    "set_name": set_name, "summary": s,
                    "figures": figs, "dir": figs_dir,
                }

    # ── the 100k geographic maps ─────────────────────────────────────────────
    geomap: dict[str, dict] = {}
    geo_root = os.path.join(root, "geomap")
    if os.path.isdir(geo_root):
        for variant in sorted(os.listdir(geo_root)):
            d = os.path.join(geo_root, variant)
            if not os.path.isdir(d):
                continue
            by_k: dict[int, dict] = {}
            for f in sorted(os.listdir(d)):
                if not (f.startswith("geographic_map_k") and f.endswith(".png")):
                    continue
                stem = f[len("geographic_map_k"):-len(".png")]
                panels = stem.endswith("_panels")
                if panels:
                    stem = stem[:-len("_panels")]
                try:
                    k = int(stem)
                except ValueError:
                    continue
                e = by_k.setdefault(k, {"png": None, "panels_png": None,
                                        "summary": None, "clusters_csv": None,
                                        "gpkg": None, "qml": None,
                                        "profile_csv": None, "colors_json": None,
                                        "ordinal_json": None,
                                        "mean_craters_png": None,
                                        "feature_profile_png": None})
                if panels:
                    e["panels_png"] = os.path.join(d, f)
                    continue
                e["png"] = os.path.join(d, f)
                sp = os.path.join(d, f"geographic_clusters_k{k}_summary.csv")
                if os.path.exists(sp):
                    try:
                        e["summary"] = pd.read_csv(sp)
                    except Exception:
                        e["summary"] = None
                for key, name in _GEO_ARTIFACT_FILES(k):
                    p = os.path.join(d, name)
                    if os.path.exists(p):
                        e[key] = p
            if by_k:
                geomap[variant] = {"by_k": by_k, "dir": d}

    return {"cells": cells, "geomap": geomap, "run_dir": run_dir, "root": root}


# How a legacy preprocess_2-derived map differs from a catalog-derived one, in
# the one way that matters when they sit side by side.
LEGACY_GEOMAP_LABEL = "legacy (preprocess_2 — seam band EXCLUDED)"


def all_geographic_variants(eval_run_dir: str | None) -> dict:
    """Every geographic map available for a checkpoint, as ONE dict keyed by
    variant name, merging two sources that used to be shown in two places:

      {run_dir}/geographic_map/                  the original, from the
                                                 preprocess_2 pipeline
      {run_dir}/data_comparison/geomap/<variant> catalog-derived, one per
                                                 (catalog x normalization)

    They were separate sections, which meant the per-cluster profile rendered
    TWICE for a checkpoint that had both, and the legacy map - built from a
    pipeline that cut out a 90-degree longitude band - was shown alongside
    full-longitude maps with nothing marking the difference. Merging them puts
    the choice in one selector where the labels can say which is which.

    The legacy entry is deliberately labelled rather than dropped: for
    checkpoints evaluated before the catalogs existed it is the only map
    there is."""
    out: dict[str, dict] = {}
    legacy = geographic_map_artifacts(eval_run_dir)
    if legacy.get("by_k"):
        out[LEGACY_GEOMAP_LABEL] = {"by_k": legacy["by_k"],
                                    "dir": os.path.join(legacy["run_dir"] or "",
                                                        "geographic_map"),
                                    "legacy": True}
    dc = data_comparison_artifacts(eval_run_dir)
    for name, g in (dc.get("geomap") or {}).items():
        out[name] = {**g, "legacy": False}
    return out


def data_comparison_table(cells: dict) -> pd.DataFrame:
    """Flatten data_comparison_artifacts()['cells'] into one row per
    (catalog, norm), with the quality metrics and the confound columns side by
    side.

    They are never separated, because they must be read together: across 67
    previously recorded evaluations, corr(lon_eta, silhouette_true) = +0.248
    (p = 0.043), i.e. part of the apparent class separation rides on the
    geographic artifact. A table showing only the quality half can be read to
    support the opposite conclusion from the same data."""
    def pt(block, key):
        v = (block or {}).get(key)
        return v.get("point") if isinstance(v, dict) else v

    rows = []
    for (cat, norm), c in sorted(cells.items()):
        s = c["summary"]
        cc = s.get("confound_correlation") or {}
        gc = s.get("geo_correlation") or {}
        sep = s.get("separation") or {}
        ag = s.get("agreement") or {}

        def eta(name):
            return (cc.get(name) or {}).get("eta_sq")
        rows.append({
            "catalog": cat, "norm": norm,
            "silhouette_true": sep.get("silhouette_true"),
            "map": pt(s, "map"), "ari": ag.get("ari"), "nmi": ag.get("nmi"),
            "accuracy": pt(s, "accuracy"),
            "qwk": pt(s, "quadratic_weighted_kappa"),
            "purity": sep.get("purity"),
            "lon_eta": eta("lon"), "lat_eta": eta("lat"),
            "brightness_eta": eta("brightness"), "sobel_eta": eta("sobel_mean"),
            "diam_eta": eta("diam_km"), "frac_sat_eta": eta("frac_saturated"),
            "pc1_vs_lon": (gc.get("pc1_vs_lon") or {}).get("r"),
            "pc2_vs_lat": (gc.get("pc2_vs_lat") or {}).get("r"),
            "n_samples": s.get("n_samples"),
        })
    return pd.DataFrame(rows)


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
        # Explicit sort before the overwrite loop below, rather than relying
        # on eval_history.csv's rows already being in append (= chronological)
        # order - true today (append_to_history() only ever appends) and
        # after every hand-edit made to the CSV so far (tag renames etc. never
        # reordered rows), but an explicit sort makes "last one standing wins
        # -> latest eval per checkpoint" correct by construction instead of by
        # convention, matching 1_Compare_Runs.py's own explicit
        # sort_values("timestamp").drop_duplicates(...) for the same purpose.
        if "timestamp" in joined.columns:
            joined = joined.sort_values("timestamp")
        # Within that chronological order, sort STANDARD evals after
        # non-standard ones so the "last one standing wins" loop below
        # lands on a standard-schema run whenever the checkpoint has one,
        # regardless of which happened to run most recently. A
        # non-standard row (eval_kind set - e.g. supervised_probe.py's,
        # which scores only new_set_final_test and writes no alignment/
        # geometry/confound sections) otherwise becomes this checkpoint's
        # entire dashboard view purely by being newest, blanking every
        # other labeled set's columns - which is what hid the off-the-
        # shelf DINOv2 entry. Non-standard runs stay in eval_history.csv
        # and stay browsable by their own run_id; they just no longer
        # define the checkpoint's headline row. Stable sort, so relative
        # order (and therefore latest-wins) is preserved within each group.
        if "eval_kind" in joined.columns:
            joined = joined.assign(
                _is_standard=joined["eval_kind"].isna()
            ).sort_values("_is_standard", kind="stable").drop(columns="_is_standard")
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


# ── what a variant name actually MEANS ──────────────────────────────────────
# Variants are keyed "<catalog>__<norm>", e.g. "morph_sigma100_div500__global_wide".
# That string is unreadable to anyone who did not build it, and it is the label
# on every figure, selector and download in the geographic sections - so a
# reader has no way to know which mosaic, which filter or which normalization
# they are looking at.
#
# The provenance is READ FROM DISK rather than hardcoded here: build_catalog
# already writes catalog_manifest.json (mosaic, filter, crater selection) and
# fit()/norm_stats writes norm_stats.json (the actual constants used), so a
# lookup table in the dashboard would be a second source of truth that silently
# goes stale the moment a catalog is rebuilt with different arguments.

_CATALOG_ROOT = "data/catalogs"

# The project's primary data source (2026-10: Kaguya TC). Its variants are
# listed FIRST - and are the default selection - in every data-comparison
# selector, table and gallery; WAC variants follow as the comparison baseline.
PRIMARY_SOURCE_PREFIX = "kaguya"


def is_primary_variant(name) -> bool:
    """True for a variant/catalog of the primary source. Accepts a
    "<catalog>__<norm>" string or a (catalog, norm) tuple."""
    cat = name[0] if isinstance(name, tuple) else str(name).split("__")[0]
    return cat.startswith(PRIMARY_SOURCE_PREFIX)


def primary_first(items, key=None):
    """Stable re-order: primary-source items first, everything else after,
    each group keeping its incoming order."""
    key = key or (lambda x: x)
    items = list(items)
    return [x for x in items if is_primary_variant(key(x))] + \
           [x for x in items if not is_primary_variant(key(x))]

# Only the parts that are genuinely not recoverable from the files: what the
# two source mosaics ARE. Everything else below comes from the manifests.
_SOURCE_BLURB = {
    "morph": ("WAC_GLOBAL morphology mosaic",
              "LROC WAC, incidence 55-75 deg - shadows deliberately retained so "
              "relief is visible. Assembled from three observation windows, "
              "which is the origin of the acquisition-band seams."),
    "emp": ("WAC_EMP_643 empirically normalized mosaic",
            "Boyd et al. 2012, photometrically corrected to a fixed geometry "
            "(i=30 deg, e=0 deg, g=30 deg), which is why it shows ~40% less "
            "band structure than the morphology mosaic."),
    "kaguya": ("Kaguya TC Ortho Map Seamless (JAXA TCO_MAPs02)",
               "SELENE Terrain Camera, 8.42 m/px (~12x finer than WAC), photometrically "
               "normalized to i=30 e=0 g=30 by JAXA. Crops are cut at NATIVE resolution "
               "on a local azimuthal-equidistant grid, registered per tile to the "
               "Robbins frame; 60S-60N, very-low-sun block (~47-60S 143-210E) excluded "
               "by the clipped-shadow cutoff. See src/data/kaguya_tco.py."),
}


def _decode_filter(desc: str) -> str:
    """Turn build_catalog's filter_desc into something readable."""
    if not desc or desc == "none":
        return "none - raw mosaic values"
    out = []
    m = re.search(r"gaussian_highpass_sigma(\d+)", desc)
    if m:
        out.append(f"subtractive Gaussian high-pass, sigma={m.group(1)} px "
                   f"(removes the additive albedo offset)")
    m = re.search(r"divisive_sigma(\d+)", desc)
    if m:
        out.append(f"then divisive local-contrast normalization, sigma={m.group(1)} px "
                   f"(removes multiplicative gain)")
    m = re.search(r"^highpass_subtract_sigma_max(\d+)km_(\d+)D", desc)
    if m:
        out.append(f"subtractive high-pass (WAC sigma100 recipe): reflectance minus its sunlit "
                   f"Gaussian low-pass, sigma = max({m.group(1)} km, {m.group(2)} x crater "
                   f"diameter) - removes the albedo offset, keeps absolute shading size")
    m = re.search(r"^(log)?ratio_sigma_max(\d+)km_(\d+)D", desc)
    if m:
        out.append(f"{'log-' if m.group(1) else ''}ratio albedo removal: reflectance divided "
                   f"by its sunlit Gaussian low-pass, sigma = max({m.group(2)} km, "
                   f"{m.group(3)} x crater diameter) on the ground (removes the "
                   f"multiplicative maria/highland albedo, keeps shading linear)")
    return "; ".join(out) or desc


def _describe_norm(norm: str, stats: dict) -> tuple[str, str]:
    """(what it does, the constants actually used) for one normalization."""
    d = (stats or {}).get(norm) or {}
    if norm == "none":
        return "values passed through unchanged", ""
    if norm in ("global", "global_wide"):
        return (f"map [{d.get('lo_pct')}, {d.get('hi_pct')}] percentile to [0,1], "
                f"CLIPPING outside it",
                f"lo={d.get('lo'):.4g}, hi={d.get('hi'):.4g}"
                if d.get("lo") is not None else "")
    if norm == "zscore":
        return ("subtract mean, divide by std - no clipping",
                f"center={d.get('center'):.4g}, scale={d.get('scale'):.4g}"
                if d.get("center") is not None else "")
    if norm == "robust":
        return ("subtract median, divide by 1.4826*MAD - no clipping, and "
                "outliers keep their magnitude instead of being flattened",
                f"center={d.get('center'):.4g}, scale={d.get('scale'):.4g}"
                if d.get("center") is not None else "")
    if norm == "asinh":
        return ("asinh compression (Lupton et al. 1999) - linear near zero, "
                "logarithmic in the tails, so bright ejecta stays ordered "
                "instead of saturating",
                f"center={d.get('center'):.4g}, scale={d.get('scale'):.4g}, "
                f"softening={d.get('softening')}"
                if d.get("center") is not None else "")
    return norm, ""


def variant_provenance(variant: str, catalog_root: str = _CATALOG_ROOT) -> dict:
    """Decode "<catalog>__<norm>" into where the data came from and what was
    done to it. Returns {} for the legacy map, which has no catalog behind it.

    Every field is read from the catalog's own manifests, so this cannot
    disagree with the data that was actually used."""
    if not variant or "__" not in variant:
        return {}
    catalog, norm = variant.split("__", 1)
    cdir = os.path.join(catalog_root, catalog)
    out: dict = {"catalog": catalog, "norm": norm, "catalog_dir": cdir,
                 "exists": os.path.isdir(cdir)}

    fam = catalog.split("_")[0]
    if fam in _SOURCE_BLURB:
        out["source_name"], out["source_note"] = _SOURCE_BLURB[fam]

    man = os.path.join(cdir, "catalog_manifest.json")
    if os.path.exists(man):
        try:
            m = json.load(open(man))
        except Exception:
            m = {}
        out["mosaic"] = m.get("map_file")
        out["filter_raw"] = m.get("filter")
        out["filter"] = _decode_filter(m.get("filter", ""))
        out["n_craters"] = m.get("n_craters")
        out["diameter_km"] = m.get("diameter_km")
        out["lat_bounds"] = m.get("lat_bounds")
        out["longitude"] = m.get("longitude_exclusion")
        out["crater_selection"] = m.get("crater_selection")
        out["excluded"] = m.get("excluded_crater_ids")
        out["built"] = datetime.fromtimestamp(os.path.getmtime(man)).strftime(
            "%Y-%m-%d %H:%M")

    ns = os.path.join(cdir, "norm_stats.json")
    if os.path.exists(ns):
        try:
            stats = json.load(open(ns))
        except Exception:
            stats = {}
        out["norm_does"], out["norm_constants"] = _describe_norm(norm, stats)
        # Percent of pixels this norm actually clips, from the percentile
        # window it was fitted with - the number that separates `global`
        # (label-correlated damage) from `global_wide` (negligible).
        d = (stats or {}).get(norm) or {}
        if d.get("clip") and d.get("lo_pct") is not None:
            out["clipped_pct"] = float(d["lo_pct"]) + (100.0 - float(d["hi_pct"]))
        raw = stats.get("raw") or {}
        if raw:
            out["raw_range"] = (raw.get("min"), raw.get("max"))
            out["raw_median"] = raw.get("median")
            out["raw_std"] = raw.get("std")

    why = os.path.join(cdir, "loader.json")
    if os.path.exists(why):
        try:
            out["why"] = (json.load(open(why)) or {}).get("_why")
        except Exception:
            pass
    return out


# ── pool labelling methods (NOT KMeans) ─────────────────────────────────────
# src/eval/pool_labeling.py writes one directory per labelling method,
# label_<tag>/, with the SAME filenames inside each. Discovery therefore globs
# for the directories rather than knowing any method's name - registering a new
# method in pool_labeling._METHODS makes it appear in the app with no change
# here, which is the whole point of that layout.

_POOL_LABEL_FILES = {
    "map_png": "map.png",
    "pca_png": "pca.png",
    "pca_band_png": "pca_band.png",
    "pca_images_png": "pca_images.png",
    "umap_png": "umap.png",
    "examples_png": "examples.png",
    "eval_json": "eval.json",
    "clusters_csv": "clusters.csv",
    "profile_csv": "profile.csv",
    "gpkg": "clusters.gpkg",
    "qml": "clusters.qml",
    "zip": "qgis.zip",
}

# Human text per method. A method missing from here still shows up, labelled by
# its directory name - the registry is for prose, not for gating discovery.
POOL_LABEL_BLURB = {
    "knn": ("k-NN, 4 classes",
            "Majority vote among the 5 nearest human-labelled craters. The "
            "straight propagation of the 642 labels, no correction applied."),
    "knn_balanced": ("k-NN, 4 classes, prior-corrected",
                     "The training class prior is divided out of the vote "
                     "(k=25, distance-weighted) so that rare classes are not "
                     "voted away. Trades overall accuracy for balanced "
                     "accuracy - read the balanced figures, not `accuracy`."),
    "knn_binary": ("k-NN, fresh vs degraded",
                   "The 4 classes collapsed to the one boundary this embedding "
                   "actually resolves: v-fr+r-fr against r-dr+v-dr. The "
                   "4-class version confuses r-dr with v-dr on half of r-dr."),
}


def pool_label_variants(geomap_variant_dir: str | None) -> dict:
    """{method_tag: {artifact_key: path}} for one geomap variant directory.

    Empty when the pool has only ever been KMeans-clustered, which is the
    normal state for older runs - the section simply does not render."""
    out: dict[str, dict] = {}
    if not geomap_variant_dir or not os.path.isdir(geomap_variant_dir):
        return out
    for name in sorted(os.listdir(geomap_variant_dir)):
        if not name.startswith("label_"):
            continue
        d = os.path.join(geomap_variant_dir, name)
        if not os.path.isdir(d):
            continue
        entry = {k: os.path.join(d, f) for k, f in _POOL_LABEL_FILES.items()
                 if os.path.exists(os.path.join(d, f))}
        if not entry:
            continue
        entry["dir"] = d
        entry["tag"] = name[len("label_"):]
        ej = entry.get("eval_json")
        if ej:
            try:
                entry["eval"] = json.load(open(ej))
            except Exception:
                entry["eval"] = {}
        out[entry["tag"]] = entry
    return out
