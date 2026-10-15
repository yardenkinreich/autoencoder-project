"""
processing_compare.py — what every combination of (source mosaic x filter x
normalization) actually does to this project's pixels, measured on the 642
expert-labeled craters and broken down BY DEGRADATION CLASS.

WHY BY CLASS, ALWAYS
--------------------
A preprocessing step is usually judged by whether it is lossy. That is the
wrong question here. The question is whether it is lossy *in proportion to the
label*, because a step that destroys 6% of the freshest craters' pixels and 0%
of the most degraded ones has not merely lost information - it has written the
answer into the input, and every downstream metric becomes partly a measurement
of the preprocessing. Two independent instances of exactly that were found in
this pipeline:

  1. The high-pass filter's own global 0.1/99.9 percentile clip
     (full_mosaic_processing._LEGACY_CLIP_PCT): pinned 6.04% of v-fr pixels and
     0.00% of v-dr pixels onto the clip bounds; 11.87% overall on EMP.
  2. The [0,1] normalization's 1st/99th percentile clip
     (src/data/norm_stats.py): eta^2 = 0.52 against the class label.

Neither raised anything. Both were found only by grouping a pixel-level
statistic by class. So that grouping is this module's whole reason to exist,
and every table it emits carries a per-class breakdown plus a Kruskal-Wallis
eta^2 saying how strongly the effect tracks the label.

WHAT IT WRITES
--------------
logs/data_profiles/processing_compare.json   - all statistics
logs/data_profiles/processing_compare/*.png  - the figures

Read by the dashboard's "Data Processing" page. Nothing is recomputed live,
matching the rest of the dashboard's convention.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data import norm_stats as NS

CLASS_NAMES = ["v-fr", "r-fr", "r-dr", "v-dr"]
CLASS_LONG = {"v-fr": "very fresh", "r-fr": "relatively fresh",
              "r-dr": "relatively degraded", "v-dr": "very degraded"}
OUT_JSON = "logs/data_profiles/processing_compare.json"
OUT_FIGS = "logs/data_profiles/processing_compare"

# The normalizations compared, in the order they appear in every table/figure.
# "global" first because it is the incumbent every historical run used.
NORM_ORDER = ["none", "global", "global_wide", "zscore", "robust", "asinh"]
NORM_LABEL = {
    "none":        "raw (no normalization)",
    "global":      "[0,1] clip @ p1/p99  (incumbent)",
    "global_wide": "[0,1] clip @ p0.01/p99.99",
    "zscore":      "z-score (mean 0, std 1), no clip",
    "robust":      "robust (median / 1.4826*MAD), no clip",
    "asinh":       "asinh(robust) - monotone tail compression, no clip",
}


# ── loading raw crops ────────────────────────────────────────────────────────

def _cache_path(variant: str) -> str:
    # v2: the cache now also stores crater ids. Selecting example craters by
    # POSITION only happened to align across variants, and would have silently
    # stopped aligning the moment one variant skipped a crop the others kept.
    return os.path.join(OUT_FIGS, f"_rawcrops_v2_{variant}.npz")


def load_raw_crops(variant: str, mosaic: str, labels_csv: str,
                   fov: float = 0.5, use_cache: bool = True):
    """The 642 labeled craters, cut RAW and at native size from `mosaic`.

    Returns (crops, classes, crater_ids). Cached, because cutting 642 windows
    out of a 17 GB GeoTIFF costs minutes and this module is re-run whenever a
    figure changes. Crops are ragged (native size varies with diameter and
    latitude), so the cache stores a concatenated float32 array plus shapes -
    the same layout as a catalog."""
    # A cache is trusted only if it is NEWER than the mosaic it was cut from.
    # That is the precise question - "were these pixels cut from the CURRENT
    # product?" - and it is what keeps re-runs cheap. Both failure modes are
    # real here: the sigma-100/200 mosaics were rebuilt unclipped after the
    # original caches were written, so blind reuse would compare new statistics
    # against old pixels; but blind recutting wastes ~3 minutes per variant
    # whose mosaic has not changed, which on 7 variants is ~20 minutes of pure
    # waste. An mtime comparison separates the two exactly.
    cache = _cache_path(variant)
    fresh = (os.path.exists(cache) and os.path.exists(mosaic)
             and os.path.getmtime(cache) > os.path.getmtime(mosaic))
    if use_cache and fresh:
        z = np.load(cache, allow_pickle=False)
        flat, shapes, cls, ids = z["flat"], z["shapes"], z["cls"], z["ids"]
        crops, o = [], 0
        for h, w in shapes:
            crops.append(flat[o:o + h * w].reshape(h, w)); o += h * w
        return crops, [CLASS_NAMES[i] for i in cls], [str(s) for s in ids]

    import pyproj
    import rasterio
    from src.helper_functions import crop_crater
    from src.data.preprocess_2 import get_craters_crs

    lab = pd.read_csv(labels_csv, dtype={"CRATER_ID": str})
    crops, cls, ids = [], [], []
    n_fail_exc = n_fail_geom = n_fail_nodata = 0
    with rasterio.open(mosaic) as ref:
        tr = pyproj.Transformer.from_crs(get_craters_crs(), ref.crs.to_string(),
                                         always_xy=True)
        for _, c in lab.iterrows():
            try:
                a = crop_crater(ref, c["lat"], c["lon"], c["diam_km"], fov, tr)
            except Exception:
                n_fail_exc += 1
                continue
            if a is None or a.ndim != 2 or a.size == 0:
                n_fail_geom += 1
                continue
            # Magnitude, not just isfinite. WAC_EMP_643's nodata is -3.4028e38,
            # which IS finite - an isfinite check passes it straight into the
            # statistics, where a single such pixel drags a mean to -inf. Same
            # guard and same constant as build_catalog.SENTINEL_MAG.
            if not np.isfinite(a).all() or np.abs(a).max() >= 1e30:
                n_fail_nodata += 1
                continue
            crops.append(np.ascontiguousarray(a, dtype=np.float32))
            cls.append(str(c["degree"]))
            ids.append(str(c["CRATER_ID"]))
    print(f"  {variant}: {len(crops)}/{len(lab)} craters usable "
          f"(dropped {n_fail_exc} crop_crater exception, {n_fail_geom} empty/non-2D, "
          f"{n_fail_nodata} nodata/non-finite)")
    os.makedirs(OUT_FIGS, exist_ok=True)
    np.savez(cache, flat=np.concatenate([c.reshape(-1) for c in crops]),
             shapes=np.array([c.shape for c in crops], dtype=np.int64),
             cls=np.array([CLASS_NAMES.index(k) for k in cls], dtype=np.int8),
             ids=np.array(ids, dtype="U32"))
    return crops, cls, ids


def seed_craters(labels_csv: str, n_per_class: int = 3, seed: int = 0,
                 restrict_to: set | None = None) -> list[str]:
    """A FIXED, stratified list of example crater ids reused by every figure in
    this module, so a reader can compare dataset-against-dataset and
    normalization-against-normalization on identical pixels.

    restrict_to: only consider these ids - pass the intersection of the ids
    every variant actually holds. Without it the pick comes from the labels
    file, which lists craters that fail to crop, and those rows silently vanish
    from the figures (measured: both r-dr examples disappeared, leaving that
    class unrepresented). Sorting before sampling keeps the choice reproducible
    regardless of set iteration order."""
    lab = pd.read_csv(labels_csv, dtype={"CRATER_ID": str})
    rng = np.random.default_rng(seed)
    out = []
    for c_name in CLASS_NAMES:
        ids = sorted(lab.loc[lab["degree"] == c_name, "CRATER_ID"].astype(str))
        if restrict_to is not None:
            ids = [i for i in ids if i in restrict_to]
        if ids:
            out += [str(i) for i in rng.choice(ids, size=min(n_per_class, len(ids)),
                                               replace=False)]
    return out


# ── statistics ───────────────────────────────────────────────────────────────

def _eta_sq_kruskal(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Kruskal-Wallis H with eta^2 = (H - k + 1)/(n - k), the same effect-size
    convention seam_check.py uses (0.01 small / 0.06 medium / 0.14 large), so
    numbers from the two analyses are directly comparable."""
    from scipy import stats
    groups = [g for g in groups if g.size > 1]
    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")
    H, p = stats.kruskal(*groups)
    n, k = sum(g.size for g in groups), len(groups)
    return float(H), float(p), float((H - k + 1) / (n - k)) if n > k else float("nan")


def variant_stats(crops, cls, name: str) -> dict:
    """Everything measurable about one (mosaic x filter) variant, before any
    normalization is chosen."""
    px = np.concatenate([c.reshape(-1) for c in crops])
    mn, mx = float(px.min()), float(px.max())
    q = np.percentile(px, [0.01, 0.1, 1, 25, 50, 75, 99, 99.9, 99.99])
    sub = px if px.size <= 20_000_000 else np.random.default_rng(0).choice(px, 20_000_000, False)
    mad = float(np.median(np.abs(sub - q[4])))
    mean, std = float(px.mean()), float(px.std())

    # Saturation AT THE STORED EXTREMES - the fingerprint of an upstream clip
    # baked into the product itself, independent of anything we do later.
    sat = {}
    for c_name in CLASS_NAMES:
        sel = [c for c, k in zip(crops, cls) if k == c_name]
        if not sel:
            continue
        g = np.concatenate([s.reshape(-1) for s in sel])
        sat[c_name] = {"n_craters": len(sel), "n_pixels": int(g.size),
                       "pct_at_extremes": float(100.0 * ((g == mn) | (g == mx)).mean()),
                       "mean": float(g.mean()), "std": float(g.std()),
                       "p1": float(np.percentile(g, 1)),
                       "p99": float(np.percentile(g, 99))}
    per_crater_sat = [float(100.0 * ((c == mn) | (c == mx)).mean()) for c in crops]
    H, p, eta = _eta_sq_kruskal([np.array([s for s, k in zip(per_crater_sat, cls)
                                           if k == c_name]) for c_name in CLASS_NAMES])
    return {
        "name": name,
        "n_craters": len(crops), "n_pixels": int(px.size),
        "native_size": {"min": int(min(min(c.shape) for c in crops)),
                        "max": int(max(max(c.shape) for c in crops)),
                        "median_h": float(np.median([c.shape[0] for c in crops])),
                        "median_w": float(np.median([c.shape[1] for c in crops]))},
        "raw": {"min": mn, "max": mx, "mean": mean, "std": std,
                "median": float(q[4]), "p0.01": float(q[0]), "p0.1": float(q[1]),
                "p1": float(q[2]), "p25": float(q[3]), "p75": float(q[5]),
                "p99": float(q[6]), "p99.9": float(q[7]), "p99.99": float(q[8]),
                "iqr": float(q[5] - q[3]), "mad": mad, "mad_std": 1.4826 * mad,
                # std/(1.4826*MAD) is 1.00 for Gaussian data and rises with tail
                # weight. It is the single number that says whether an affine
                # normalization suffices or a compression is warranted, because
                # it compares a tail-sensitive scale estimate against a
                # tail-immune one on the same footing.
                "tail_ratio_std_over_madstd": (std / (1.4826 * mad)) if mad > 0 else float("inf"),
                "skew": float(((px - mean) ** 3).mean() / std ** 3) if std else 0.0,
                "kurtosis_excess": (float(((px - mean) ** 4).mean() / std ** 4 - 3.0)
                                    if std else 0.0)},
        "stored_saturation": {
            "per_class": sat,
            "overall_pct": float(100.0 * ((px == mn) | (px == mx)).mean()),
            "vs_class": {"H": H, "p": p, "eta_sq": eta},
            "note": "share of pixels sitting on EXACTLY the stored min or max. A "
                    "high-pass residual is a difference of smooth fields and should "
                    "have no value spike at its extremes; a nonzero number here means "
                    "an upstream step clipped the product.",
        },
    }


def norm_stats_for_variant(crops, cls, vstats: dict) -> dict:
    """For each candidate normalization: what fraction of each class's pixels
    it destroys, and where the class ends up on the new scale."""
    r = vstats["raw"]
    # Constants fitted on THESE crops. For a real training run they come from
    # the parent catalog's 100k-crater sample instead (norm_stats.json); here
    # the labeled set is the population under study, so fitting on it keeps the
    # comparison self-contained and every variant treated identically.
    stats = {
        "global":      {"lo": r["p1"], "hi": r["p99"], "clip": True},
        "global_wide": {"lo": r["p0.01"], "hi": r["p99.99"], "clip": True},
        "zscore":      {"center": r["mean"], "scale": r["std"], "clip": False},
        "robust":      {"center": r["median"], "scale": r["mad_std"], "clip": False},
        "asinh":       {"center": r["median"], "scale": r["mad_std"], "clip": False,
                        "compress": "asinh", "softening": 1.0},
    }
    out = {}
    for norm in NORM_ORDER:
        per_class, per_crater_clip = {}, []
        d = stats.get(norm)
        for c_name in CLASS_NAMES:
            sel = [c for c, k in zip(crops, cls) if k == c_name]
            if not sel:
                continue
            g = np.concatenate([s.reshape(-1) for s in sel])
            if d is not None and d.get("clip"):
                lo, hi = d["lo"], d["hi"]
                below, above = (g < lo), (g > hi)
            else:
                below = above = np.zeros(g.shape, dtype=bool)
            v = NS.apply(g, norm, stats) if norm != "none" else g
            per_class[c_name] = {
                "n_craters": len(sel),
                # For the unbounded normalizations the cost is not clipping but
                # tail extent, so record how much mass lands far from centre.
                "pct_beyond_3": float(100.0 * (np.abs(v) > 3).mean()),
                "pct_beyond_5": float(100.0 * (np.abs(v) > 5).mean()),
                "pct_beyond_10": float(100.0 * (np.abs(v) > 10).mean()),
                "clip_low_pct": float(100.0 * below.mean()),
                "clip_high_pct": float(100.0 * above.mean()),
                "clip_pct": float(100.0 * (below | above).mean()),
                "mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max()),
            }
            if d is not None and d.get("clip"):
                for s in sel:
                    per_crater_clip.append((c_name, float(100.0 * (
                        (s < d["lo"]) | (s > d["hi"])).mean())))
        H, p, eta = (_eta_sq_kruskal(
            [np.array([c for k, c in per_crater_clip if k == n]) for n in CLASS_NAMES])
            if per_crater_clip else (0.0, 1.0, 0.0))
        allpx = np.concatenate([c.reshape(-1) for c in crops])
        v_all = NS.apply(allpx, norm, stats) if norm != "none" else allpx
        out[norm] = {
            "label": NORM_LABEL[norm], "constants": d,
            "per_class": per_class,
            "overall": {"mean": float(v_all.mean()), "std": float(v_all.std()),
                        "min": float(v_all.min()), "max": float(v_all.max()),
                        "pct_beyond_3": float(100.0 * (np.abs(v_all) > 3).mean()),
                        "pct_beyond_5": float(100.0 * (np.abs(v_all) > 5).mean()),
                        "pct_beyond_10": float(100.0 * (np.abs(v_all) > 10).mean())},
            "overall_clip_pct": float(np.mean([pc["clip_pct"] for pc in per_class.values()]))
            if per_class else 0.0,
            "clip_vs_class": {"H": H, "p": p, "eta_sq": eta},
        }
    return out


# ── figures ──────────────────────────────────────────────────────────────────

def _ramp():
    """Degradation classes are ORDINAL, so they get the sequential ordinal ramp
    from eval/visualize.py rather than a categorical palette - the colour
    ordering carries real meaning here."""
    from src.eval.visualize import _ORDINAL_RAMP
    n = len(CLASS_NAMES)
    return [_ORDINAL_RAMP[int(round(i * (len(_ORDINAL_RAMP) - 1) / (n - 1)))]
            for i in range(n)]


def fig_raw_distributions(all_stats: dict, out_path: str):
    """Per-class raw-value distributions, one small multiple per variant.

    Small multiples rather than one overlaid panel: the variants live on
    genuinely different absolute scales (raw reflectance ~[0,0.55] vs a
    high-pass residual ~[-0.05,0.09]), so a shared x-axis would compress three
    of them into a spike. Each panel therefore has its OWN x-axis and says so."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_stats.keys())
    cols = min(2, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.5 * rows), squeeze=False)
    colors = _ramp()
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        s = all_stats[name]
        hs = s.get("_hists")
        if not hs:
            ax.axis("off"); continue
        for j, c_name in enumerate(CLASS_NAMES):
            h = hs.get(c_name)
            if not h:
                continue
            edges, dens = np.array(h["edges"]), np.array(h["density"])
            ctr = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(ctr, dens, color=colors[j], lw=1.6,
                    label=f"{c_name} ({CLASS_LONG[c_name]})")
        r = s["raw"]
        ax.axvline(r["p1"], color="#444", ls=":", lw=1.0)
        ax.axvline(r["p99"], color="#444", ls=":", lw=1.0)
        ax.set_yscale("log")
        ax.set_title(f"{name}\nraw [{r['min']:.4g}, {r['max']:.4g}]  "
                     f"skew={r['skew']:.2f}  dotted = p1/p99 clip bounds", fontsize=9)
        ax.set_xlabel("raw pixel value (own scale per panel)", fontsize=8)
        ax.set_ylabel("density (log)", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    for i in range(len(names), rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle("Raw pixel distributions by degradation class — the p1/p99 clip bounds "
                 "sit inside the fresh classes' tails", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_clip_by_class(all_stats: dict, out_path: str):
    """The headline: clipped-pixel share per class, per normalization, per
    variant. Grouped bars, log y - the effect spans 0.00% to 30%, which a
    linear axis renders as four invisible bars and one tall one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(all_stats.keys())
    norms = [n for n in NORM_ORDER if n not in ("none",)]
    fig, axes = plt.subplots(1, len(variants), figsize=(3.6 * len(variants), 3.9),
                             squeeze=False, sharey=True)
    colors = _ramp()
    for vi, vname in enumerate(variants):
        ax = axes[0][vi]
        norm_blocks = all_stats[vname]["normalizations"]
        x = np.arange(len(norms))
        w = 0.8 / len(CLASS_NAMES)
        FLOOR = 1e-3
        for ci, c_name in enumerate(CLASS_NAMES):
            pos = x + ci * w - 0.4 + w / 2
            raw = [norm_blocks[n]["per_class"].get(c_name, {}).get("clip_pct", 0.0)
                   for n in norms]
            # An exact zero gets NO bar and an explicit "0" instead. Plotting it
            # at a floor would draw a visible stub indistinguishable from a real
            # small value - and "clips nothing" versus "clips 0.001%" is the
            # entire distinction this figure exists to make.
            drawn = [v if v > 0 else np.nan for v in raw]
            ax.bar(pos, drawn, width=w * 0.92, color=colors[ci],
                   label=c_name if vi == 0 else None)
            for xp, v in zip(pos, raw):
                if v == 0:
                    ax.text(xp, FLOOR * 1.35, "0", ha="center", va="bottom",
                            fontsize=6, color="#888888")
        ax.set_yscale("log")
        ax.set_ylim(bottom=FLOOR)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("global_wide", "wide") for n in norms],
                           fontsize=7, rotation=20, ha="right")
        ax.set_title(vname, fontsize=9)
        if vi == 0:
            ax.set_ylabel("% of pixels clipped (log)", fontsize=8)
            ax.legend(fontsize=7, frameon=False, title="class", title_fontsize=7)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Share of pixels destroyed by each normalization, by degradation class\n"
                 "z-score and robust clip nothing at all, so they have no bars",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _norm_constants(vstats: dict) -> dict:
    """The five normalizations' constants for one variant, fitted on that
    variant's own labeled-set pixels. Single definition, used by every figure
    so no two panels can be built from different constants."""
    r = vstats["raw"]
    return {"global": {"lo": r["p1"], "hi": r["p99"], "clip": True},
            "global_wide": {"lo": r["p0.01"], "hi": r["p99.99"], "clip": True},
            "zscore": {"center": r["mean"], "scale": r["std"], "clip": False},
            "robust": {"center": r["median"], "scale": r["mad_std"], "clip": False},
            "asinh": {"center": r["median"], "scale": r["mad_std"], "clip": False,
                      "compress": "asinh", "softening": 1.0}}


def fig_dataset_grid(by_variant: dict, crater_id: str, cls_of: str, out_path: str):
    """ONE crater, every dataset down the rows, every normalization across the
    columns - the panel that answers "is this difference the mosaic, the
    filter, or the normalization?" for a single fixed crater.

    by_variant: {variant_name: (crop_array, vstats)} for this crater.

    The per-variant grid shows one dataset against all normalizations; this is
    its transpose-partner, and between them every axis of the processing choice
    is visible on identical pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    variants = list(by_variant.keys())
    cols = NORM_ORDER + ["clipmask"]
    # 2.35 in per row, not 1.95: each panel carries a value-range label beneath
    # it, and at the tighter spacing row N's label overlapped row N+1's image.
    fig, axes = plt.subplots(len(variants), len(cols),
                             figsize=(1.75 * len(cols), 2.35 * len(variants)),
                             squeeze=False)
    mask_cmap = ListedColormap(["#00000000", "#d55e00"])
    for ri, vname in enumerate(variants):
        a, vstats = by_variant[vname]
        stats = _norm_constants(vstats)
        for cj, norm in enumerate(cols):
            ax = axes[ri][cj]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#dddddd")
            if a is None:
                ax.axis("off"); continue
            if norm == "clipmask":
                m = (a < stats["global"]["lo"]) | (a > stats["global"]["hi"])
                ax.imshow(NS.apply(a, "robust", stats), cmap="gray",
                          interpolation="nearest")
                ax.imshow(m.astype(int), cmap=mask_cmap, vmin=0, vmax=1,
                          interpolation="nearest")
                ax.set_xlabel(f"{100.0 * m.mean():.1f}%", fontsize=7,
                              color="#d55e00", labelpad=1)
                if ri == 0:
                    ax.set_title("pixels the p1/p99\nclip destroys", fontsize=7)
                continue
            v = NS.apply(a, norm, stats) if norm != "none" else a
            ax.imshow(v, cmap="gray", interpolation="nearest")
            ax.set_xlabel(f"[{v.min():.2f}, {v.max():.2f}]", fontsize=6.5, labelpad=1)
            if ri == 0:
                ax.set_title(norm, fontsize=7.5)
        axes[ri][0].set_ylabel(vname, fontsize=7.5)
    fig.suptitle(f"Crater {crater_id}  ({cls_of} — {CLASS_LONG.get(cls_of, '')})\n"
                 "rows = dataset, columns = normalization; identical pixels "
                 "throughout, each panel autoscaled with its range printed beneath",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_crop_grid(crops, cls, vstats: dict, out_path: str, ids=None,
                  pick_ids=None, n_per_class: int = 3, seed: int = 0):
    """One crater per row, every normalization across the columns, plus a final
    column showing WHICH pixels the incumbent clip destroys.

    The saturation mask is the point of this figure: it makes "30% of pixels"
    concrete by showing that the lost pixels are the rim and the shadow, i.e.
    the morphology, not a scattering of noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Select by CRATER ID, not by position. Row i of one variant's crop list is
    # not guaranteed to be the same crater as row i of another's - any variant
    # that skips a crop shifts everything after it - so a positional pick would
    # produce grids that look comparable and quietly are not.
    if pick_ids and ids is not None:
        pos = {cid: i for i, cid in enumerate(ids)}
        picks = [(cls[pos[c]], pos[c]) for c in pick_ids if c in pos]
    else:
        rng = np.random.default_rng(seed)
        picks = []
        for c_name in CLASS_NAMES:
            sel = [i for i, k in enumerate(cls) if k == c_name]
            if sel:
                picks += [(c_name, int(i)) for i in
                          rng.choice(sel, size=min(n_per_class, len(sel)), replace=False)]

    stats = _norm_constants(vstats)
    cols = NORM_ORDER + ["clipmask"]
    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(1.75 * len(cols), 1.85 * len(picks)), squeeze=False)
    mask_cmap = ListedColormap(["#00000000", "#d55e00"])
    for ri, (c_name, i) in enumerate(picks):
        a = crops[i]
        for cj, norm in enumerate(cols):
            ax = axes[ri][cj]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#dddddd")
            if norm == "clipmask":
                m = (a < stats["global"]["lo"]) | (a > stats["global"]["hi"])
                # grayscale crater underneath, destroyed pixels flagged on top
                ax.imshow(NS.apply(a, "robust", stats), cmap="gray",
                          interpolation="nearest")
                ax.imshow(m.astype(int), cmap=mask_cmap, vmin=0, vmax=1,
                          interpolation="nearest")
                if ri == 0:
                    ax.set_title("pixels the p1/p99\nclip destroys", fontsize=7)
                ax.set_xlabel(f"{100.0 * m.mean():.1f}%", fontsize=7,
                              color="#d55e00", labelpad=1)
                continue
            v = NS.apply(a, norm, stats) if norm != "none" else a
            # Per-panel autoscale is CORRECT here and everywhere else in this
            # figure would be wrong: the whole question is what each mapping
            # does to the VALUES, and the panels live on incomparable scales
            # (raw ~0.05, z-score ~±3). Fixing a shared vmin/vmax would render
            # four of the five columns as flat gray. The value range is printed
            # under each panel so nothing is hidden by the autoscale.
            ax.imshow(v, cmap="gray", interpolation="nearest")
            ax.set_xlabel(f"[{v.min():.2f}, {v.max():.2f}]", fontsize=6.5, labelpad=1)
            if ri == 0:
                ax.set_title(norm, fontsize=7.5)
        lbl = f"{c_name}"
        if ids is not None:
            lbl = f"{c_name}\n{ids[i]}"
        axes[ri][0].set_ylabel(lbl, fontsize=6.5)
    fig.suptitle(f"{vstats['name']} — the same craters under every normalization\n"
                 "each panel autoscaled (the scales are not comparable); "
                 "value range printed beneath", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── driver ───────────────────────────────────────────────────────────────────

DEFAULT_VARIANTS = [
    ("morph_raw", "data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif"),
    ("morph_sigma100", "data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif"),
    ("morph_sigma200", "data/raw/wac_mosaic_new_version/sigma/200/highpass_filtered_lunar_mosaic.tif"),
    ("emp_raw", "data/raw/wac_uniform/WAC_EMP_643_100m_180.tif"),
    ("emp_sigma100", "data/raw/wac_emp643/sigma/100/highpass_filtered_emp643.tif"),
    ("emp_sigma200", "data/raw/wac_emp643/sigma/200/highpass_filtered_emp643.tif"),
    # sigma-100 high-pass THEN divisive local-contrast normalization at
    # sigma=500 px. The high-pass is subtractive and removes the regional
    # albedo offset; this one is divisive and targets the multiplicative gain
    # difference that produces the tile seam - which subtraction cannot touch.
    ("morph_sigma100_div500",
     "data/raw/wac_mosaic_new_version/sigma/100_div500/highpass_filtered_lunar_mosaic.tif"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", nargs="*", default=None,
                    help="name=path pairs; defaults to the six catalog variants.")
    ap.add_argument("--labels-csv", default="configs/liorr_labels.csv")
    ap.add_argument("--fov", type=float, default=0.5)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--no-cache", action="store_true",
                    help="Kept for compatibility; caches are now validated by mtime "
                         "against their source mosaic on every run, so a stale one is "
                         "never reused and a current one is never needlessly recut. "
                         "Use --force-recut to rebuild every variant regardless.")
    ap.add_argument("--force-recut", action="store_true",
                    help="Rebuild every variant's crops from its mosaic, even when the "
                         "cache is provably current. The escape hatch for the one case "
                         "mtime cannot see: a mosaic rewritten with its timestamp "
                         "preserved.")
    ap.add_argument("--per-class", type=int, default=3,
                    help="Example craters per class. These same craters appear in "
                         "every per-variant grid AND get their own cross-dataset "
                         "grid, so raising this multiplies figures.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Selects WHICH example craters; fixed so figures are "
                         "reproducible across runs.")
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args()

    variants = ([tuple(v.split("=", 1)) for v in args.variant] if args.variant
                else DEFAULT_VARIANTS)
    os.makedirs(OUT_FIGS, exist_ok=True)

    # PASS 1: read every variant's pixels. Done before choosing example craters
    # because the labels file lists more craters than actually crop
    # successfully (770 rows, 642 usable - some fall outside a mosaic or come
    # back non-finite), so picking from the labels file alone names craters
    # that exist in no variant. Selecting from the INTERSECTION of what every
    # variant actually has is the only way a cross-dataset grid can be complete
    # in every row.
    loaded, skipped = {}, []
    for name, path in variants:
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found")
            skipped.append({"name": name, "path": path, "reason": "missing"})
            continue
        print(f"\n=== {name} ===", flush=True)
        crops, cls, ids = load_raw_crops(name, path, args.labels_csv, args.fov,
                                         use_cache=not args.force_recut)
        print(f"  {len(crops)} crops", flush=True)
        loaded[name] = (crops, cls, ids)

    if not loaded:
        raise SystemExit("no variants available - has the high-pass rebuild finished?")

    common = set.intersection(*(set(v[2]) for v in loaded.values()))
    print(f"\ncraters present in all {len(loaded)} variants: {len(common)}")
    ref_cls = {cid: k for _, (_, cls_, ids_) in loaded.items()
               for cid, k in zip(ids_, cls_)}
    pick_ids = seed_craters(args.labels_csv, n_per_class=args.per_class,
                            seed=args.seed, restrict_to=common)
    print(f"example craters ({args.per_class}/class): {', '.join(pick_ids)}")
    for c_name in CLASS_NAMES:
        got = sum(1 for c in pick_ids if ref_cls.get(c) == c_name)
        if got < args.per_class:
            print(f"  WARNING: only {got}/{args.per_class} example craters for class "
                  f"{c_name} - the figures will under-represent it.")
    seed_crops: dict[str, dict] = {cid: {} for cid in pick_ids}
    seed_class: dict[str, str] = {}

    # PASS 2: statistics and figures.
    all_stats = {}
    for name, (crops, cls, ids) in loaded.items():
        path = dict(variants)[name]
        s = variant_stats(crops, cls, name)
        s["source_mosaic"] = path
        s["normalizations"] = norm_stats_for_variant(crops, cls, s)

        # Histograms precomputed here so the dashboard never touches pixels.
        lo, hi = s["raw"]["p0.01"], s["raw"]["p99.99"]
        edges = np.linspace(lo, hi, args.bins + 1)
        s["_hists"] = {}
        for c_name in CLASS_NAMES:
            sel = [c for c, k in zip(crops, cls) if k == c_name]
            if not sel:
                continue
            g = np.concatenate([x.reshape(-1) for x in sel])
            d, _ = np.histogram(g, bins=edges, density=True)
            s["_hists"][c_name] = {"edges": edges.tolist(), "density": d.tolist()}
        all_stats[name] = s

        sat = s["stored_saturation"]
        print(f"  stored saturation {sat['overall_pct']:.4f}% overall "
              f"(eta^2 vs class = {sat['vs_class']['eta_sq']:.4f})")
        for c_name, d in sat["per_class"].items():
            print(f"    {c_name:5s} {d['pct_at_extremes']:7.4f}%")
        for norm in NORM_ORDER:
            nb = s["normalizations"][norm]
            pc = "  ".join(f"{c}={nb['per_class'][c]['clip_pct']:.3f}%"
                           for c in CLASS_NAMES if c in nb["per_class"])
            print(f"  {norm:12s} range [{nb['overall']['min']:.3f}, "
                  f"{nb['overall']['max']:.3f}]  >|5|="
                  f"{nb['overall']['pct_beyond_5']:6.3f}%  eta^2="
                  f"{nb['clip_vs_class']['eta_sq']:.4f}  clip: {pc}")

        fig_crop_grid(crops, cls, s, os.path.join(OUT_FIGS, f"crops_{name}.png"),
                      ids=ids, pick_ids=pick_ids)

        # Stash the seed craters' pixels so the cross-dataset grids can be
        # built once every variant has been read.
        pos = {cid: i for i, cid in enumerate(ids)}
        for cid in pick_ids:
            if cid in pos:
                seed_crops[cid][name] = (crops[pos[cid]], s)
                seed_class.setdefault(cid, cls[pos[cid]])

    if not all_stats:
        raise SystemExit("no variants available - has the high-pass rebuild finished?")

    fig_raw_distributions(all_stats, os.path.join(OUT_FIGS, "raw_distributions.png"))
    fig_clip_by_class(all_stats, os.path.join(OUT_FIGS, "clip_by_class.png"))

    # Cross-dataset grids: one per example crater, datasets down the rows.
    examples = []
    for cid in pick_ids:
        by_v = seed_crops.get(cid) or {}
        if not by_v:
            continue
        p = os.path.join(OUT_FIGS, f"crater_{cid}.png")
        fig_dataset_grid(by_v, cid, seed_class.get(cid, "?"), p)
        examples.append({"crater_id": cid, "class": seed_class.get(cid, "?"),
                         "png": p, "variants": sorted(by_v.keys())})
        print(f"  wrote {p}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump({"variants": all_stats, "skipped": skipped,
               "class_names": CLASS_NAMES, "class_long": CLASS_LONG,
               "norm_order": NORM_ORDER, "norm_label": NORM_LABEL,
               "labels_csv": args.labels_csv, "fov": args.fov,
               "figures_dir": OUT_FIGS,
               "example_craters": examples},
              open(args.out_json, "w"), indent=2)
    print(f"\nwrote {args.out_json}\n      {OUT_FIGS}/*.png")


if __name__ == "__main__":
    main()
