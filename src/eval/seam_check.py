"""
seam_check.py — is there a constant-longitude (vertical) discontinuity in
either the imagery or the clustering?

The WAC morphology mosaic is stitched from orbital swaths, and this project
already drops one band (180-270 E) for exactly that reason: incidence-angle
discontinuities along constant-longitude seams (see preprocess_2.py's
filter_craters_by_diameter docstring and the Snakefile's --exclude_lon_bounds).
That leaves the retained data spanning signed longitude -90 to +180, whose
two EDGES (-90, i.e. 270 E, and +180) are themselves seam boundaries. If the
imagery is systematically different near those edges, then a "cluster
concentrated at the edge of the map" is an imaging artifact, not geology -
and the geographic maps would be showing acquisition geometry rather than
the Moon.

This walks longitude in bins and reports, per bin:
  - crater count (a real, expected signal: maria are young and undercratered)
  - mean pixel brightness and mean local contrast, straight from the memmap
  - the cluster composition of that bin

Measuring pixel statistics here is the POINT (the question is literally
whether the imagery varies with longitude), which is a different thing from
letting a pixel statistic reorder or recolor a result - see
geographic_map.py's module docstring on why that was deliberately removed
there.

How to read it: a smooth trend across longitude is ordinary lunar geology
(nearside maria vs. farside highlands is a real, strong, global asymmetry).
A STEP - a discontinuity at a fixed longitude, especially at -90 or +180 -
is the artifact signature, because nothing about real geology knows where a
mosaic's swath boundaries are.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

# The seam boundaries of the retained region, in signed longitude. -90 is
# 270 E and +180 is 180 E - the two edges of the band this project excludes.
_SEAMS = (-90.0, 180.0)


def per_crater_stats(dat_path: str, batch_size: int = 512) -> pd.DataFrame:
    """Mean pixel value and a local-contrast proxy per crater, off the
    memmap. Contrast is the mean absolute horizontal neighbour difference -
    a cheap stand-in for an edge/Sobel magnitude that needs no extra
    dependency and answers the same question (is this crop crisp or washed
    out). Batched so a 6GB memmap never lands in RAM at once."""
    from src.eval.holdout import load_holdout_dat
    data = load_holdout_dat(dat_path)
    mean = np.empty(len(data), dtype=np.float64)
    contrast = np.empty(len(data), dtype=np.float64)
    for i in range(0, len(data), batch_size):
        chunk = np.asarray(data[i:i + batch_size], dtype=np.float64)
        flat = chunk.reshape(len(chunk), -1)
        mean[i:i + batch_size] = flat.mean(axis=1)
        # chunk is (N, C, H, W); horizontal neighbour difference along W
        contrast[i:i + batch_size] = np.abs(np.diff(chunk, axis=-1)).reshape(len(chunk), -1).mean(axis=1)
    return pd.DataFrame({"brightness": mean, "contrast": contrast})


def zone_of(lon: np.ndarray, edge_width: float) -> np.ndarray:
    """Label each crater left-seam / middle / right-seam by how close it is
    to a seam boundary of the retained region."""
    z = np.full(len(lon), "middle", dtype=object)
    z[lon <= _SEAMS[0] + edge_width] = "left-seam"
    z[lon >= _SEAMS[1] - edge_width] = "right-seam"
    return z


def compare_zones(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Kruskal-Wallis per measure across the three zones - the same
    non-parametric convention metrics.confound_correlation() already uses in
    this project, so the numbers are read the same way. eta_sq is the
    H-based effect size; >=0.14 is the "large effect" bar used there."""
    from scipy.stats import kruskal
    zones = ["left-seam", "middle", "right-seam"]
    rows = []
    for c in cols:
        groups = [df.loc[df["zone"] == z, c].dropna().to_numpy() for z in zones]
        if min(len(g) for g in groups) < 3:
            rows.append({"measure": c, "H": np.nan, "p_value": np.nan, "eta_sq": np.nan})
            continue
        h, p = kruskal(*groups)
        n = sum(len(g) for g in groups)
        eta_sq = (h - len(groups) + 1) / (n - len(groups))
        row = {"measure": c, "H": h, "p_value": p, "eta_sq": eta_sq}
        for z, g in zip(zones, groups):
            row[f"mean_{z}"] = float(g.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_profiles(binned: pd.DataFrame, cluster_frac: pd.DataFrame, out_png: str,
                  title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.geographic_map import _MAP_CATEGORICAL

    # Four separate panels sharing the longitude axis rather than twinned
    # y-axes: count, brightness, contrast and composition are four
    # different scales, and overlaying different-scale measures on one pair
    # of axes is the single most misleading thing a chart like this can do.
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    def _seams(ax):
        for s in _SEAMS:
            ax.axvline(s, color="#b22222", linestyle="--", linewidth=1.2, zorder=1)
        ax.grid(alpha=0.25, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        ax.tick_params(labelsize=8, colors="#555555")

    axes[0].plot(binned["lon_center"], binned["n"], color="#333333", linewidth=1.8)
    axes[0].set_ylabel("craters per bin", fontsize=9, color="#555555")
    axes[0].set_title("crater count — a real dip over the young, undercratered maria is expected",
                      fontsize=9, color="#555555", loc="left")
    _seams(axes[0])

    axes[1].plot(binned["lon_center"], binned["brightness"], color="#0072b2", linewidth=1.8)
    axes[1].set_ylabel("mean brightness", fontsize=9, color="#555555")
    axes[1].set_title("mean pixel brightness — a STEP at a seam is the artifact signature",
                      fontsize=9, color="#555555", loc="left")
    _seams(axes[1])

    axes[2].plot(binned["lon_center"], binned["contrast"], color="#009e73", linewidth=1.8)
    axes[2].set_ylabel("mean local contrast", fontsize=9, color="#555555")
    axes[2].set_title("mean local contrast (neighbour difference)", fontsize=9,
                      color="#555555", loc="left")
    _seams(axes[2])

    for i, c in enumerate(sorted(cluster_frac.columns)):
        axes[3].plot(cluster_frac.index, cluster_frac[c], linewidth=1.8,
                     color=_MAP_CATEGORICAL[i % len(_MAP_CATEGORICAL)],
                     label=f"cluster {c}")
    axes[3].set_ylabel("fraction of bin", fontsize=9, color="#555555")
    axes[3].set_xlabel("signed longitude (deg; -90 = 270 E, dashed red = seam boundary)",
                       fontsize=9, color="#555555")
    axes[3].set_title("cluster composition per longitude bin", fontsize=9,
                      color="#555555", loc="left")
    axes[3].legend(fontsize=8, ncol=4, frameon=True, framealpha=0.9)
    _seams(axes[3])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters_csv", required=True,
                    help="geographic_clusters_k*.csv written by eval/geographic_map.py")
    ap.add_argument("--dat_path",
                    default="data/processed_wac_100m_new/sigma/100/geomap_100k_d1-12/craters_wide.dat")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bin_width", type=float, default=7.5,
                    help="Longitude bin width in degrees.")
    ap.add_argument("--edge_width", type=float, default=15.0,
                    help="How many degrees from a seam boundary counts as that seam's zone.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.clusters_csv, dtype={"id": str})
    stats = per_crater_stats(args.dat_path)
    if len(stats) != len(df):
        raise ValueError(f"{args.clusters_csv} has {len(df)} rows but the memmap has "
                         f"{len(stats)} - they must come from the same preprocess run.")
    df = pd.concat([df.reset_index(drop=True), stats], axis=1)
    df["zone"] = zone_of(df["lon"].to_numpy(), args.edge_width)

    print(f"zone counts:\n{df['zone'].value_counts().to_string()}\n")

    zone_tbl = compare_zones(df, ["brightness", "contrast", "diam"])
    print("=== Kruskal-Wallis across left-seam / middle / right-seam ===")
    print(zone_tbl.to_string(index=False))
    zone_tbl.to_csv(os.path.join(args.out_dir, "seam_zone_comparison.csv"), index=False)

    edges = np.arange(-90, 180.001, args.bin_width)
    df["lon_bin"] = pd.cut(df["lon"], bins=edges, include_lowest=True)
    binned = (df.groupby("lon_bin", observed=True)
                .agg(n=("id", "size"), brightness=("brightness", "mean"),
                     contrast=("contrast", "mean"), mean_diam=("diam", "mean"))
                .reset_index())
    binned["lon_center"] = [iv.mid for iv in binned["lon_bin"]]
    binned.drop(columns="lon_bin").to_csv(
        os.path.join(args.out_dir, "seam_longitude_profile.csv"), index=False)

    frac = (df.groupby(["lon_bin", "cluster"], observed=True).size()
              .unstack(fill_value=0))
    frac = frac.div(frac.sum(axis=1), axis=0)
    frac.index = [iv.mid for iv in frac.index]

    png = os.path.join(args.out_dir, "seam_check.png")
    plot_profiles(binned, frac, png,
                  title=f"{args.label} — longitude profile of imagery and cluster composition "
                        f"({len(df):,} craters)")
    json.dump({"label": args.label, "clusters_csv": args.clusters_csv,
               "n_craters": int(len(df)), "edge_width_deg": args.edge_width,
               "zone_counts": df["zone"].value_counts().to_dict()},
              open(os.path.join(args.out_dir, "seam_check.json"), "w"), indent=2, default=float)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
