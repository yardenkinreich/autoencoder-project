"""
profile_data_sources.py — precompute distribution profiles for every crop
set in configs/data_sources.yaml (per-crater brightness/sobel-texture
features, diameter/lat/lon from metadata where available, cross-source
significance tests, per-source geo/diameter correlations), so the Data
Sources dashboard page only ever reads pre-computed artifacts - same "no
live recomputation" convention the rest of this dashboard follows.

Reuses metrics.py's existing per-crater feature extraction
(image_intensity_features) and cross-group significance test
(confound_correlation, here grouping by data SOURCE instead of a model's
cluster assignment) rather than reimplementing either.

Usage:
    PYTHONPATH=src python -m eval.profile_data_sources
    PYTHONPATH=src python -m eval.profile_data_sources --sample-size 3000
"""
from __future__ import annotations
import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, ks_2samp, levene, rankdata, norm
from statsmodels.stats.multitest import multipletests

from eval.metrics import image_intensity_features, confound_correlation
from eval.holdout import load_holdout_dat

# Single accent color (dataviz skill's categorical slot 1) - a box plot
# doesn't need one hue per source at all, since x-position + labels already
# carry identity; that also sidesteps the skill's "≤3 slots validate for
# all-pairs/overlapping series" cap, which 6 sources would otherwise hit.
_BOX_COLOR = "#2a78d6"

DEFAULT_CATALOG = "configs/data_sources.yaml"
DEFAULT_OUT = "logs/data_profiles"
DEFAULT_SAMPLE = 2000
FEATURE_COLS = ["brightness", "sobel_mean", "frac_saturated", "diam_km"]


def _pixel_features(img_0_1: np.ndarray) -> dict:
    """Same three computations image_intensity_features() does per-file,
    applied to an in-memory [0,1] array instead - for sources with no
    per-crater display crop file to read (see _dat_features below)."""
    import cv2
    img = img_0_1.astype(np.float32) * 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return {"brightness": float(img.mean()),
           "sobel_mean": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
           "frac_saturated": float((img >= 254.0).mean())}


def _dat_features(dat_path: str, row_idx: np.ndarray) -> pd.DataFrame:
    """Pixel features for specific row positions of a craters.dat memmap -
    for a source whose crater_crops/ was never materialized
    (--save_raw_crops wasn't used at preprocessing time), so there's no
    per-crater file for image_intensity_features() to read. Row order is
    assumed to match its sibling metadata.csv's row order (the same
    assumption pipeline.labels_from_memmap_csv()/holdout.py rely on)."""
    data = load_holdout_dat(dat_path)
    return pd.DataFrame([_pixel_features(np.asarray(data[i, 0])) for i in row_idx])


def _source_resolution(entry: dict) -> dict:
    """Channels/height/width of one real crop from this source - not every
    source shares the project's current 1-channel/128x128 convention (some
    are older-convention crops, e.g. 224x224 or 3-channel), which is a real
    confound for comparing brightness/sobel magnitude across sources on top
    of the raw-vs-sigma-filtered pipeline difference already flagged."""
    if "dat_path" in entry:
        data = load_holdout_dat(entry["dat_path"])
        c, h, w = data.shape[1], data.shape[2], data.shape[3]
    elif "crops_dir" in entry:
        f = next((f for f in sorted(os.listdir(entry["crops_dir"])) if f.endswith(".npy")), None)
        if f is None:
            return {"channels": None, "height": None, "width": None}
        arr = np.load(os.path.join(entry["crops_dir"], f))
        c, h, w = arr.shape
    else:
        from PIL import Image
        f = next((f for f in sorted(os.listdir(entry["imgs_dir"])) if f.endswith(".png")), None)
        img = Image.open(os.path.join(entry["imgs_dir"], f))
        c, h, w = (3 if img.mode == "RGB" else 1), img.height, img.width
    return {"channels": int(c), "height": int(h), "width": int(w)}


def _sample_source(entry: dict, sample_size: int, seed: int = 0) -> pd.DataFrame:
    """crater_id, filename, [lat, lon, diam_km] for up to sample_size
    randomly sampled craters from this source. metadata_csv-backed sources
    (id,lat,lon,x,y,diam columns) get geo/diameter fields for free; an
    imgs_dir-only source (Julie's hand-provided PNGs) has none - pixel
    features only, same limitation robbins_lookup_features() already
    documents for Julie's set elsewhere in this codebase."""
    if "metadata_csv" in entry:
        meta = pd.read_csv(entry["metadata_csv"])
        if len(meta) > sample_size:
            meta = meta.sample(sample_size, random_state=seed).sort_index()
        row_idx = meta.index.to_numpy()
        df = pd.DataFrame({
            "crater_id": meta["id"].astype(str).to_numpy(),
            "lat": meta["lat"].to_numpy(),
            "lon": meta["lon"].to_numpy(),
            "diam_km": meta["diam"].to_numpy() / 1000.0,
        })
        if "dat_path" in entry:
            feats = _dat_features(entry["dat_path"], row_idx)
            return pd.concat([df.reset_index(drop=True), feats], axis=1)
        df["filename"] = df["crater_id"] + ".npy"
        feats = image_intensity_features(df, entry["crops_dir"])
        return df.merge(feats, on="crater_id", how="inner")

    rng = np.random.RandomState(seed)
    files = sorted(f for f in os.listdir(entry["imgs_dir"]) if f.endswith(".png"))
    if len(files) > sample_size:
        files = list(rng.choice(files, sample_size, replace=False))
    df = pd.DataFrame({
        "crater_id": [f.rsplit(".", 1)[0] for f in files],
        "filename": files,
    })
    feats = image_intensity_features(df, entry["imgs_dir"])
    return df.merge(feats, on="crater_id", how="inner")


def _geo_correlation(df: pd.DataFrame) -> dict:
    """Spearman correlation of each pixel feature against lat/lon/diameter,
    within one source - same non-parametric convention as
    metrics.latent_geo_correlation(), just against raw image features
    instead of latent PCA components."""
    out = {}
    for feat in ["brightness", "sobel_mean", "frac_saturated"]:
        for geo in ["lat", "lon", "diam_km"]:
            sub = df[[feat, geo]].dropna()
            # a constant feature (e.g. frac_saturated=0 for every crater in
            # a source that never saturates at all) makes rank correlation
            # undefined, not just "no correlation" - report that distinctly
            # rather than let scipy emit a silent NaN dressed up as a value.
            if len(sub) < 3 or sub[feat].nunique() < 2:
                out[f"{feat}_vs_{geo}"] = {"r": None, "p_value": None, "n": len(sub)}
                continue
            r, p = spearmanr(sub[feat], sub[geo])
            out[f"{feat}_vs_{geo}"] = {"r": float(r), "p_value": float(p), "n": int(len(sub))}
    return out


def _pairwise_ks_levene(groups: dict[str, np.ndarray]) -> dict:
    """Two-sample KS + Levene for every pair of sources.

    Kruskal-Wallis (the omnibus test in confound_correlation) mainly detects
    a location/rank shift - it can under-flag two distributions that share
    a similar median but differ in shape or spread (exactly what this
    project already found: Julie's set has roughly half training's std with
    a similar mean). KS tests the whole distribution shape; Levene tests
    variance equality specifically - together they catch what Kruskal-
    Wallis alone can miss."""
    out = {}
    for a, b in combinations(groups.keys(), 2):
        va, vb = groups[a], groups[b]
        key = f"{a}__vs__{b}"
        if len(va) < 2 or len(vb) < 2:
            out[key] = {"ks_stat": None, "ks_p": None, "levene_stat": None, "levene_p": None}
            continue
        ks_stat, ks_p = ks_2samp(va, vb)
        # Levene's F is undefined (0/0) when both groups have zero variance
        # (e.g. a feature that's constant across the board in both) -
        # report that distinctly rather than a silent NaN.
        if len(np.unique(va)) < 2 and len(np.unique(vb)) < 2:
            out[key] = {"ks_stat": float(ks_stat), "ks_p": float(ks_p),
                       "levene_stat": None, "levene_p": None}
            continue
        lev_stat, lev_p = levene(va, vb)
        out[key] = {"ks_stat": float(ks_stat), "ks_p": float(ks_p),
                    "levene_stat": float(lev_stat), "levene_p": float(lev_p)}
    return out


def _dunn_test(groups: dict[str, np.ndarray]) -> dict:
    """Dunn's post-hoc test (tie-corrected pairwise rank-sum z-test) with
    Holm step-down correction across all pairs - the standard follow-up to
    a significant Kruskal-Wallis, which only says "some group differs
    somewhere", not which pair. Implemented directly against the textbook
    formula rather than pulling in scikit-posthocs, an extra dependency
    this project doesn't otherwise need."""
    names = list(groups.keys())
    sizes = [len(groups[n]) for n in names]
    all_vals = np.concatenate([groups[n] for n in names])
    N = len(all_vals)
    ranks = rankdata(all_vals)

    _, counts = np.unique(all_vals, return_counts=True)
    tie_correction = (counts ** 3 - counts).sum() / (12 * (N - 1)) if N > 1 else 0.0

    offsets = np.cumsum([0] + sizes)
    mean_rank = {n: ranks[offsets[i]:offsets[i + 1]].mean() for i, n in enumerate(names)}
    size = dict(zip(names, sizes))

    pairs = list(combinations(names, 2))
    zs, raw_p = [], []
    for a, b in pairs:
        se_sq = (N * (N + 1) / 12 - tie_correction) * (1 / size[a] + 1 / size[b])
        z = (mean_rank[a] - mean_rank[b]) / np.sqrt(se_sq) if se_sq > 0 else 0.0
        zs.append(z)
        raw_p.append(2 * (1 - norm.cdf(abs(z))))

    adj_p = multipletests(raw_p, method="holm")[1] if raw_p else []
    return {f"{a}__vs__{b}": {"z": float(z), "p_value": float(p), "p_holm": float(ap)}
           for (a, b), z, p, ap in zip(pairs, zs, raw_p, adj_p)}


def profile_all(catalog_path: str = DEFAULT_CATALOG, sample_size: int = DEFAULT_SAMPLE) -> dict:
    cfg = yaml.safe_load(open(catalog_path))
    per_source = {}
    pooled_rows = []

    for entry in cfg["sources"]:
        name = entry["name"]
        print(f"profiling {name} ...")
        df = _sample_source(entry, sample_size)
        df["source"] = name
        n_sampled = len(df)

        def _feature_summary(col: str) -> dict:
            vals = df[col].dropna()
            n_valid = len(vals)
            pct = np.percentile(vals, [5, 25, 50, 75, 95]).tolist() if n_valid else [None] * 5
            return {
                "mean": float(vals.mean()) if n_valid else None,
                "std": float(vals.std()) if n_valid else None,
                "p5": pct[0], "p25": pct[1], "p50": pct[2], "p75": pct[3], "p95": pct[4],
                # load_rate < 1.0 means some crop files for this source
                # couldn't be read (see image_intensity_features()'s NaN
                # fallback) - a data-quality signal worth surfacing on its
                # own, not just silently dropped from the stats above.
                "load_rate": n_valid / n_sampled if n_sampled else None,
                "values": vals.round(4).tolist(),
            }

        per_source[name] = {
            "data_tag": entry["data_tag"], "fov": entry.get("fov"), "note": entry.get("note"),
            "n_sampled": n_sampled,
            "resolution": _source_resolution(entry),
            "features": {col: _feature_summary(col) for col in FEATURE_COLS if col in df.columns},
        }
        if "lat" in df.columns:
            per_source[name]["geo_correlation"] = _geo_correlation(df)
            per_source[name]["geo_extent"] = {
                "lat_min": float(df["lat"].min()), "lat_max": float(df["lat"].max()),
                "lon_min": float(df["lon"].min()), "lon_max": float(df["lon"].max()),
            }
        pooled_rows.append(df)

    # Cross-source significance test: does each feature differ across
    # sources? Reuses confound_correlation's Kruskal-Wallis machinery
    # verbatim - "cluster" there is "source" here, same test either way.
    # crater_id can collide across sources (the same physical crater
    # sampled from two different crop sets), so the join key confound_
    # correlation groups by must be source-qualified.
    pooled = pd.concat(pooled_rows, ignore_index=True)
    pooled["crater_id"] = pooled["source"] + "::" + pooled["crater_id"].astype(str)
    feature_cols = [c for c in FEATURE_COLS if c in pooled.columns]
    cross_source_test = confound_correlation(
        pooled[["crater_id", "source"]], pooled[["crater_id", *feature_cols]], group_col="source")

    # Pairwise follow-up per feature: KS + Levene (distribution shape/
    # variance, not just the omnibus rank-shift KW already tests) and
    # Dunn's post-hoc (which specific pairs actually differ).
    pairwise = {}
    for feat in feature_cols:
        groups = {name: pooled.loc[pooled["source"] == name, feat].dropna().to_numpy()
                  for name in per_source}
        groups = {n: v for n, v in groups.items() if len(v) > 0}
        pairwise[feat] = {
            "ks_levene": _pairwise_ks_levene(groups),
            "dunn": _dunn_test(groups),
        }

    return {"sources": per_source, "cross_source_test": cross_source_test,
           "pairwise_tests": pairwise, "sample_size": sample_size}


def plot_feature_distributions(per_source: dict, feature_cols: list[str], out_dir: str) -> dict[str, str]:
    """One box plot per feature, one box per source. A box plot needs no
    per-source hue at all - x-position + labels already carry identity -
    which sidesteps needing 6 distinct, CVD-safe colors for what would
    otherwise be 6 overlapping histograms (the dataviz skill's palette caps
    at 3 slots for that many mutually-overlapping series)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for feat in feature_cols:
        data, labels = [], []
        for name, s in per_source.items():
            vals = s["features"].get(feat, {}).get("values")
            if vals:
                data.append(vals)
                labels.append(name)
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(data)), 5))
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.55)
        for box in bp["boxes"]:
            box.set(facecolor=_BOX_COLOR, alpha=0.35, edgecolor=_BOX_COLOR, linewidth=1.5)
        for med in bp["medians"]:
            med.set(color="#0b0b0b", linewidth=2)
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} by data source")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e5e5e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{feat}_boxplot.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[feat] = out_path
    return paths


def plot_feature_histograms(per_source: dict, feature_cols: list[str], out_dir: str,
                            shared_y: bool = False) -> dict[str, str]:
    """Small multiples: one histogram subplot per source, all on a shared
    x-axis range for direct visual comparison - the actual distribution
    shape (multimodal? skewed? where's the saturation spike?), which a box
    plot's five-number summary can hide. One subplot per source sidesteps
    needing per-source hues entirely (same reasoning as the box plots
    above), which 6 overlapping histograms in one axes would need.

    shared_y: also share the y-axis across subplots, for reading relative
    heights directly instead of eyeballing each subplot's own scale.
    Sources here have very different sample sizes (Julie n=150 vs. 2000
    elsewhere) - a shared RAW COUNT axis would just make the small-N source
    look flat, not genuinely comparable, so this normalizes to density
    (area under each histogram = 1) instead. Independent y-axis (the
    default) keeps raw counts, each subplot auto-scaled to its own data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for feat in feature_cols:
        series = {name: s["features"][feat]["values"] for name, s in per_source.items()
                 if s["features"].get(feat, {}).get("values")}
        if not series:
            continue
        lo = min(min(v) for v in series.values())
        hi = max(max(v) for v in series.values())
        bins = np.linspace(lo, hi, 40) if hi > lo else 10
        ncols = min(3, len(series))
        nrows = -(-len(series) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                                 squeeze=False, sharey=shared_y)
        for i, (name, vals) in enumerate(series.items()):
            ax = axes[i // ncols][i % ncols]
            ax.hist(vals, bins=bins, density=shared_y, color=_BOX_COLOR, alpha=0.75,
                   edgecolor="white", linewidth=0.3)
            ax.set_title(name, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
        for j in range(len(series), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        y_label = "density (shared y-axis, comparable across sample sizes)" if shared_y else "count"
        axes[0][0].set_ylabel(y_label, fontsize=9)
        fig.suptitle(f"{feat} - distribution per source (shared x-axis"
                    f"{', shared y-axis' if shared_y else ''})")
        fig.tight_layout()
        suffix = "_hist_shared_y.png" if shared_y else "_hist.png"
        out_path = os.path.join(out_dir, f"{feat}{suffix}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[feat] = out_path
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE)
    args = ap.parse_args()

    result = profile_all(args.catalog, args.sample_size)
    plot_dir = os.path.join(args.out, "plots")
    result["plots"] = plot_feature_distributions(result["sources"], FEATURE_COLS, plot_dir)
    result["histograms"] = plot_feature_histograms(result["sources"], FEATURE_COLS, plot_dir)
    result["histograms_shared_y"] = plot_feature_histograms(
        result["sources"], FEATURE_COLS, plot_dir, shared_y=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "profile.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
