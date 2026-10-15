"""
norm_stats.py — fit every candidate normalization's constants for one raw
catalog, in one pass over one pixel sample, and freeze them to
`norm_stats.json` beside the catalog.

WHY THIS IS A SEPARATE FILE FROM scaling.json
---------------------------------------------
scaling.json holds only the 1st/99th percentiles, because when it was written
[0,1]-with-clipping was assumed to be *the* normalization. It is not: the clip
turned out to be severely class-dependent on this data (measured eta^2 = 0.52
against the degradation label - the freshest class lost 30.0% of its pixels to
saturation while the most degraded lost 0.07%, a 411x disparity, because fresh
craters ARE the high-dynamic-range ones and the percentile window is fitted on
a corpus that is ~63% degraded). A normalization that destroys the cue in
proportion to the label is not a neutral preprocessing step, so the choice has
to be made on evidence, which means all the candidates' constants have to
exist side by side.

This file is also deliberately backfillable onto an already-built catalog: it
reads only crops.f32 + index.csv, so it can be run after the fact without
rebuilding a multi-GB store (and without disturbing a build already in
flight, which has the old module loaded).

THE CANDIDATES
--------------
  global       (x-p1)/(p99-p1) clipped to [0,1]      - the incumbent
  global_wide  (x-p0.01)/(p99.99-p0.01) clipped      - same shape, clip ~inert
  zscore       (x-mean)/std                          - no clip, unbounded
  robust       (x-median)/(1.4826*MAD)               - no clip, outlier-immune
  asinh        asinh(robust)                         - no clip, tail compressed
  none         raw

The MAD is scaled by 1.4826 so that for Gaussian data 1.4826*MAD -> sigma,
i.e. `robust` and `zscore` land on the same scale for well-behaved data and
diverge only where the tails are heavy - which is exactly the regime this
project's fresh craters live in. The measured ratio std/(1.4826*MAD) is a
direct heavy-tail diagnostic: 1.00 for Gaussian data, but 1.86 on the raw
morphology mosaic and 2.45 on raw EMP.

WHY `asinh` IS HERE
-------------------
The raw mosaics are reflectance, which is bounded below by zero and unbounded
above, so it is strongly right-skewed (measured skew 3.53 morphology / 2.57
EMP) and NO affine map can fix that - centre-and-scale only slides the tail
around. After `robust`, the raw morphology mosaic spans [-3.96, +33.16], with
2.9% of pixels beyond |5| and 0.39% beyond |10|: not outliers, a genuinely fat
bright tail from fresh rims and ejecta.

asinh is the standard answer to exactly this in astronomy (Lupton et al. 1999,
the SDSS "asinh magnitude"), because it is linear near zero and logarithmic in
the tail: it leaves the bulk of the data essentially untouched, compresses the
extremes into a usable range, is defined for negative values (unlike log, which
matters for high-pass residuals), and - the key property the percentile clip
lacks - it is strictly monotone, so it PRESERVES THE ORDERING of the bright
pixels instead of collapsing them onto a single saturated value. Fresh craters
stay the brightest things in the image; they just stop being 33 sigma.

asinh(z) ~ z for |z| < 1 and ~ ln(2z) beyond, so the [-3.96, 33.16] above maps
to about [-2.09, +4.19].
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from src.data.build_catalog import load_catalog, load_index

NORM_STATS_FILE = "norm_stats.json"

# The normalizations the loader knows how to apply. "none" needs no constants.
NORMS = ("none", "global", "global_wide", "zscore", "robust", "asinh")


def _strided(a: np.ndarray, max_px: int) -> np.ndarray:
    """An even spatial subsample of a 2-D crop with at most ~max_px pixels."""
    step = int(np.ceil(np.sqrt(a.size / max_px))) if a.size > max_px else 1
    return a[::step, ::step].reshape(-1)


def fit(catalog_dir: str, sample: int = 20000, seed: int = 0,
        max_px_per_crater: int | None = None) -> dict:
    """One pixel sample, every normalization's constants. Sampling CRATERS
    (not pixels) matches fit_scaling/fit_global_scaling so the percentiles are
    directly comparable to the ones already frozen for existing runs.

    max_px_per_crater: if set, each sampled crop contributes an even strided
    subsample of at most ~this many pixels, read in chunks. Needed for the
    native-resolution Kaguya catalogs (8.4 m/px crops up to ~3,500 px a side:
    20k whole crops are ~34 GB before any copy, which OOM-killed the first
    postprocess). It also stops the largest craters dominating the pixel
    sample by area. None (default) = every pixel, as for all WAC catalogs."""
    idx = load_index(catalog_dir)
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(idx), size=min(sample, len(idx)), replace=False))
    if max_px_per_crater is None:
        crops = load_catalog(catalog_dir, scaling=None, rows=take)
        px = np.concatenate([c.reshape(-1) for c in crops])
    else:
        parts = []
        for k in range(0, len(take), 500):
            for c in load_catalog(catalog_dir, scaling=None, rows=take[k:k + 500]):
                parts.append(_strided(c, max_px_per_crater))
        px = np.concatenate(parts)

    # Validity by MAGNITUDE, not just np.isfinite. WAC_EMP_643's
    # nodata = -3.4028e38 is finite, so isfinite() lets it through, and a
    # float32 mean over ~7e8 pixels containing even one of them collapses to
    # -inf - which then makes every zscore/asinh constant inf. That happened:
    # 5 crops out of 100,000 in emp_raw, invisible in the percentiles (which
    # are immune to a few extreme values) and fatal to the mean.
    bad = ~np.isfinite(px) | (np.abs(px) >= 1e30)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"  WARNING: {n_bad:,}/{px.size:,} sampled pixels are non-finite or of "
              f"sentinel magnitude ({100.0 * n_bad / px.size:.6f}%) - fitting on the "
              f"rest. This means the CATALOG contains nodata pixels; check its "
              f"skipped_invalid.csv and the source mosaic's nodata handling.")
    px = px[~bad]
    if px.size == 0:
        raise ValueError(f"{catalog_dir}: no valid sampled pixels")

    q = np.percentile(px, [0.01, 0.1, 1.0, 25.0, 50.0, 75.0, 99.0, 99.9, 99.99])
    p0p01, p0p1, p1, p25, p50, p75, p99, p99p9, p99p99 = (float(v) for v in q)
    # MAD on a subsample: a median of |x-median| over ~1e9 pixels is pure cost
    # for three digits of precision we do not need.
    sub = px if px.size <= 20_000_000 else rng.choice(px, 20_000_000, replace=False)
    mad = float(np.median(np.abs(sub - p50)))
    mean, std = float(px.mean()), float(px.std())

    stats = {
        "raw": {"min": float(px.min()), "max": float(px.max()),
                "mean": mean, "std": std, "median": p50,
                "p0.01": p0p01, "p0.1": p0p1, "p1": p1, "p25": p25,
                "p75": p75, "p99": p99, "p99.9": p99p9, "p99.99": p99p99,
                "iqr": p75 - p25, "mad": mad,
                "skew": float(((px - mean) ** 3).mean() / std ** 3) if std else 0.0,
                "kurtosis_excess": (float(((px - mean) ** 4).mean() / std ** 4 - 3.0)
                                    if std else 0.0)},
        "global":      {"lo": p1, "hi": p99, "clip": True,
                        "lo_pct": 1.0, "hi_pct": 99.0},
        "global_wide": {"lo": p0p01, "hi": p99p99, "clip": True,
                        "lo_pct": 0.01, "hi_pct": 99.99},
        "zscore":      {"center": mean, "scale": std, "clip": False},
        "robust":      {"center": p50, "scale": 1.4826 * mad, "clip": False,
                        "scale_kind": "1.4826*MAD (sigma-equivalent for Gaussian)"},
        "asinh":       {"center": p50, "scale": 1.4826 * mad, "clip": False,
                        "compress": "asinh",
                        "softening": 1.0,
                        "scale_kind": "1.4826*MAD, then asinh(z/softening)"},
        # std/(1.4826*MAD): 1.00 for Gaussian data, higher the heavier the
        # tails. This is the number that says whether an affine normalization
        # is enough or a compression is warranted.
        "tail_ratio_std_over_madstd": (std / (1.4826 * mad)) if mad > 0 else float("inf"),
        "n_craters_sampled": int(len(take)),
        "max_px_per_crater": max_px_per_crater,
        "n_pixels_finite": int(px.size),
        # Non-finite OR sentinel-magnitude, not just NaN - the check above was
        # widened to catch WAC_EMP_643's finite -3.4028e38 nodata, so the field
        # name follows what is actually counted.
        "n_invalid_sampled": n_bad,
        "seed": seed,
        "catalog": catalog_dir,
    }
    for k in ("global", "global_wide", "zscore", "robust", "asinh"):
        d = stats[k]
        a, b = (d.get("lo"), d.get("hi")) if "lo" in d else (d["center"], d["scale"])
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"{catalog_dir}: non-finite constants for norm={k}: {d}")
    if stats["global"]["hi"] <= stats["global"]["lo"]:
        raise ValueError(f"{catalog_dir}: degenerate global window {stats['global']}")
    for k in ("zscore", "robust", "asinh"):
        if stats[k]["scale"] <= 0:
            raise ValueError(f"{catalog_dir}: non-positive scale for norm={k} - would "
                             f"divide by zero: {stats[k]}")
    return stats


def load(catalog_dir: str) -> dict:
    p = os.path.join(catalog_dir, NORM_STATS_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found - run `PYTHONPATH=. python -m src.data.norm_stats "
            f"--catalog {catalog_dir}` to fit it.")
    return json.load(open(p))


def apply(a: np.ndarray, norm: str, stats: dict) -> np.ndarray:
    """The single implementation of every normalization. Both the torch
    dataset and the analysis/dashboard code call THIS, so a figure in the app
    and a tensor in a model can never disagree about what `robust` means."""
    if norm in ("none", None):
        return a
    if norm not in NORMS:
        raise ValueError(f"unknown norm={norm!r}; known: {NORMS}")
    d = stats[norm]
    if d.get("clip"):
        lo, hi = float(d["lo"]), float(d["hi"])
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    z = (a - float(d["center"])) / float(d["scale"])
    if d.get("compress") == "asinh":
        # Strictly monotone, so the ordering of bright pixels survives - the
        # one property the percentile clip destroys. Linear for |z| < 1,
        # logarithmic beyond.
        return np.arcsinh(z / float(d.get("softening", 1.0)))
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, nargs="+",
                    help="Catalog directory/ies (each holding crops.f32 + index.csv).")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-px-per-crater", type=int, default=None,
                    help="Cap each sampled crop at ~N evenly strided pixels (read in "
                         "chunks). Use for native-resolution catalogs (e.g. Kaguya: "
                         "65536). Default: every pixel, as for the WAC catalogs.")
    ap.add_argument("--also-testset", action="store_true",
                    help="Also fit the catalog's testset/ subdirectory. Note this "
                         "writes the TESTSET's own constants; models should normally "
                         "use the parent catalog's, so that train and test share one "
                         "scaling. Useful only for the artifact analysis.")
    args = ap.parse_args()

    for cat in args.catalog:
        dirs = [cat] + ([os.path.join(cat, "testset")] if args.also_testset else [])
        for d in dirs:
            if not os.path.exists(os.path.join(d, "index.csv")):
                print(f"skip {d}: no index.csv")
                continue
            print(f"\n=== {d} ===", flush=True)
            s = fit(d, sample=args.sample, seed=args.seed,
                    max_px_per_crater=args.max_px_per_crater)
            json.dump(s, open(os.path.join(d, NORM_STATS_FILE), "w"), indent=2)
            r = s["raw"]
            print(f"  raw      [{r['min']:.6g}, {r['max']:.6g}]  mean={r['mean']:.6g} "
                  f"std={r['std']:.6g}  median={r['median']:.6g}")
            print(f"  skew={r['skew']:.3f}  excess kurtosis={r['kurtosis_excess']:.2f} "
                  f"(0 = Gaussian; high = heavy tails, i.e. the clip bites)")
            print(f"  global      [{s['global']['lo']:.6g}, {s['global']['hi']:.6g}]  (clip)")
            print(f"  global_wide [{s['global_wide']['lo']:.6g}, "
                  f"{s['global_wide']['hi']:.6g}]  (clip)")
            print(f"  zscore      center={s['zscore']['center']:.6g} "
                  f"scale={s['zscore']['scale']:.6g}")
            print(f"  robust      center={s['robust']['center']:.6g} "
                  f"scale={s['robust']['scale']:.6g}")
            print(f"  -> {os.path.join(d, NORM_STATS_FILE)}")


if __name__ == "__main__":
    main()
