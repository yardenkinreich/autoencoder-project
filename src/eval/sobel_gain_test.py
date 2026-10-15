"""
sobel_gain_test.py — does dividing each crater's texture by a LOCAL texture
level reduce the longitude artifact, and at what cost to the signal?

THE HYPOTHESIS
--------------
The mosaic seam is a CONTRAST (gain) difference, not a brightness offset:
adjacent source tiles were imaged at different incidence angles, so the local
dynamic range is scaled. That is why the Gaussian high-pass cannot remove it -
subtracting a blurred copy removes an additive offset and leaves a gain
untouched. The matched correction is therefore DIVISIVE: estimate the local
contrast level and divide by it.

WHY NO MOSAIC PASS IS NEEDED
----------------------------
The obvious implementation is `resid / sqrt(blur_sigma(resid^2))` over the
full raster: a multi-hour pass over ~4e9 pixels. It is also NOT approximable
by a decimated read, which is the tempting shortcut - `resid` is locally
zero-mean, so block averaging drives it toward 0 and destroys exactly the
quantity being estimated.

But the craters ARE a sample of the local texture level, already measured. So
the gain field is estimated from them directly: for each crater, the MEDIAN
sobel of its spatial neighbours within a radius. That is a KD-tree over 100k
points instead of a raster pass, runs in seconds, and is arguably the better
reference anyway - it is the typical texture of *craters* nearby, not of
whatever terrain surrounds them.

Median rather than mean: a few very fresh craters in a neighbourhood would
drag a mean upward and partially normalize away the signal under study.

WHAT IS MEASURED, ON BOTH AXES
------------------------------
Reducing the artifact is only half the question - a transform that destroyed
all texture would score perfectly on it. So both are reported:

  ARTIFACT  Kruskal-Wallis epsilon-squared of sobel across 10-degree longitude
            bins, plus the ratio between the most and least textured bin
            medians. Lower is better.
  SIGNAL    the spread of per-CLUSTER mean sobel (the 4.2x range that produced
            the visible sharp->smooth progression in the mean-crater grid) and
            the rank correlation between corrected and uncorrected per-cluster
            ordering. Higher / closer to 1 is better.

A correction that reduces the artifact proportionally MORE than the signal is
a win; one that shrinks both equally has merely rescaled everything.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd


def per_crater_sobel(dat_path: str, n_expected: int, size: int = 128,
                     batch: int = 512) -> np.ndarray:
    """Gradient-magnitude mean per crop - the same measure cluster_profile
    uses, recomputed here so this test needs nothing but the memmap."""
    from src.eval.holdout import load_holdout_dat
    data = load_holdout_dat(dat_path, size=size)
    if len(data) != n_expected:
        raise ValueError(f"{dat_path} has {len(data)} crops, table has {n_expected}")
    out = np.empty(len(data), dtype=np.float64)
    for i0 in range(0, len(data), batch):
        chunk = np.asarray(data[i0:i0 + batch], dtype=np.float32)[:, 0]
        gx = np.diff(chunk, axis=2)
        gy = np.diff(chunk, axis=1)
        m = np.sqrt(gx[:, :-1, :] ** 2 + gy[:, :, :-1] ** 2)
        out[i0:i0 + len(chunk)] = m.reshape(len(chunk), -1).mean(axis=1)
    return out


def local_gain(x: np.ndarray, y: np.ndarray, sobel: np.ndarray,
               radius_km: float, min_neighbours: int = 12):
    """Median sobel of the neighbours within `radius_km`, per crater.

    The crater itself is included: at ~80 neighbours it contributes ~1%, and
    excluding it would cost an extra query per point for no measurable
    difference."""
    from scipy.spatial import cKDTree
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    idx = tree.query_ball_point(pts, r=radius_km * 1000.0)
    gain = np.empty(len(sobel), dtype=np.float64)
    counts = np.empty(len(sobel), dtype=np.int64)
    for i, nb in enumerate(idx):
        counts[i] = len(nb)
        gain[i] = np.median(sobel[nb]) if len(nb) >= min_neighbours else np.nan
    info = {"radius_km": radius_km,
            "neighbours_median": float(np.median(counts)),
            "neighbours_p5": float(np.percentile(counts, 5)),
            "n_too_sparse": int(np.isnan(gain).sum())}
    # Sparse neighbourhoods fall back to the GLOBAL median, leaving those
    # craters effectively uncorrected. Using their own sobel as the gain would
    # force them to exactly 1.0 - inventing uniformity rather than measuring it.
    gain[np.isnan(gain)] = float(np.median(sobel))
    return gain, info


def longitude_artifact(sobel: np.ndarray, lon: np.ndarray,
                       bin_deg: float = 10.0) -> dict:
    """Kruskal-Wallis epsilon-squared of sobel across longitude bins, plus the
    ratio of the most to least textured bin median - a scale-free step
    measure, so it stays comparable after a transform that rescales."""
    from scipy import stats as sps
    bins = pd.cut(pd.Series(lon), np.arange(-180, 180 + bin_deg, bin_deg))
    groups = [sobel[(bins == b).to_numpy()] for b in bins.cat.categories]
    groups = [g for g in groups if len(g) >= 20]
    if len(groups) < 2:
        return {"eta_sq": float("nan")}
    H, p = sps.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    med = np.array([np.median(g) for g in groups])
    return {"H": float(H), "p": float(p),
            "eta_sq": float((H - k + 1) / (n - k)),
            "bin_median_min": float(med.min()), "bin_median_max": float(med.max()),
            "bin_median_ratio": float(med.max() / med.min()) if med.min() > 0 else float("inf"),
            "n_bins": int(k)}


def cluster_signal(sobel: np.ndarray, clusters: np.ndarray) -> dict:
    """Between-cluster texture spread - the quantity behind the visible
    sharp->smooth progression. A RATIO, so it survives rescaling and stays
    comparable before and after."""
    ids = sorted(pd.unique(clusters))
    means = np.array([sobel[clusters == c].mean() for c in ids])
    return {"cluster_mean_min": float(means.min()),
            "cluster_mean_max": float(means.max()),
            "cluster_spread_ratio": float(means.max() / means.min())
            if means.min() > 0 else float("inf"),
            "means_by_cluster": {int(c): float(m) for c, m in zip(ids, means)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--radii-km", type=float, nargs="+",
                    default=[50.0, 100.0, 200.0, 400.0],
                    help="Neighbourhood radii. Must be >> crater size (1-12 km) so a "
                         "crater cannot normalize itself, and << the ~2700 km tile "
                         "width so the gain field can still track the seam.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    csv = os.path.join(args.map_dir, f"geographic_clusters_k{args.k}.csv")
    meta = os.path.join(args.map_dir, f"geographic_map_k{args.k}.json")
    df = pd.read_csv(csv, dtype={"id": str})
    dat_path = json.load(open(meta))["dat_path"]
    print(f"{len(df):,} craters from {csv}\ncrops: {dat_path}\n")

    sobel = per_crater_sobel(dat_path, len(df))
    lon = df["lon"].to_numpy(dtype=float)
    clusters = df["cluster"].to_numpy()

    base_art = longitude_artifact(sobel, lon)
    base_sig = cluster_signal(sobel, clusters)
    print(f"{'variant':>14}  {'lon eta^2':>10} {'bin ratio':>10} "
          f"{'clust spread':>12} {'rank rho':>9}  verdict")
    print(f"{'UNCORRECTED':>14}  {base_art['eta_sq']:10.4f} "
          f"{base_art['bin_median_ratio']:10.3f} "
          f"{base_sig['cluster_spread_ratio']:12.3f} {'-':>9}")

    from scipy import stats as sps
    base_means = np.array([base_sig["means_by_cluster"][c]
                           for c in sorted(base_sig["means_by_cluster"])])
    results = {"uncorrected": {"artifact": base_art, "signal": base_sig}}
    for r in args.radii_km:
        gain, info = local_gain(df["x"].to_numpy(float), df["y"].to_numpy(float),
                                sobel, r)
        corr = np.nan_to_num(sobel / np.where(gain > 0, gain, np.nan), nan=1.0)
        art = longitude_artifact(corr, lon)
        sig = cluster_signal(corr, clusters)
        new_means = np.array([sig["means_by_cluster"][c]
                              for c in sorted(sig["means_by_cluster"])])
        rho = float(sps.spearmanr(base_means, new_means).statistic)

        art_drop = (1.0 - art["eta_sq"] / base_art["eta_sq"]
                    if base_art["eta_sq"] else float("nan"))
        sig_drop = (1.0 - ((sig["cluster_spread_ratio"] - 1.0)
                           / (base_sig["cluster_spread_ratio"] - 1.0))
                    if base_sig["cluster_spread_ratio"] > 1 else float("nan"))
        verdict = ("WIN - artifact drops more than signal"
                   if art_drop > sig_drop + 0.05 else
                   "no better than a rescale" if abs(art_drop - sig_drop) <= 0.05 else
                   "LOSS - signal drops more than artifact")
        print(f"{f'r={r:g}km':>14}  {art['eta_sq']:10.4f} "
              f"{art['bin_median_ratio']:10.3f} {sig['cluster_spread_ratio']:12.3f} "
              f"{rho:9.3f}  {verdict}")
        print(f"{'':>14}  artifact -{art_drop:.1%}, signal -{sig_drop:.1%}, "
              f"neighbours median {info['neighbours_median']:.0f} "
              f"(p5 {info['neighbours_p5']:.0f})")
        results[f"radius_{r:g}km"] = {"artifact": art, "signal": sig,
                                      "neighbourhood": info,
                                      "rank_rho_vs_uncorrected": rho,
                                      "artifact_reduction": float(art_drop),
                                      "signal_reduction": float(sig_drop),
                                      "verdict": verdict}

    out = args.out_json or os.path.join(args.map_dir,
                                        f"sobel_gain_test_k{args.k}.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("\nrank rho = does the corrected per-cluster texture ORDERING still match the "
          "uncorrected one? Near 1.0 means the sharp->smooth axis survived; well below "
          "means the correction reshuffled it.")


if __name__ == "__main__":
    main()
