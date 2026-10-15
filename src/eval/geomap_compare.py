"""
geomap_compare.py — how geographically structured is each 100k-crater cluster
map, and does the choice of source mosaic reduce it?

This is the 100,000-crater counterpart to the labeled-set confound test. The
labeled set has 642 craters and asks "do the k-means clusters differ in
longitude"; this asks the same question over the whole mapped surface, where
there are no labels but ~150x more craters.

STATISTIC: Cramer's V between the cluster id and 10-degree geographic bins.

    V = sqrt( chi2 / (n * (min(rows, cols) - 1)) )

Chosen over the Kruskal-Wallis epsilon-squared used elsewhere because BOTH
variables here are categorical (cluster id x longitude bin), which is a
contingency table, not a one-way ANOVA on ranks. V is bounded 0-1 and, unlike
raw chi-square, does not grow with n - which matters a great deal at n =
100,000, where a negligible association still yields an astronomically
significant p-value. For that reason p-values are not reported: at this
sample size they are all ~0 and carry no information.

Latitude is computed alongside longitude deliberately. They are separate
artifacts with separate causes - longitude banding comes from mosaic tile
acquisition, while a latitude gradient tracks illumination geometry - and a
processing choice can easily improve one while worsening the other.
"""
from __future__ import annotations
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = "logs/data_profiles/geomap"
OUT_CSV = "logs/data_profiles/geomap_compare.csv"
_K_RE = re.compile(r"geographic_clusters_k(\d+)\.csv$")


def cramers_v(cluster: pd.Series, binned: pd.Series) -> tuple[float, int]:
    from scipy import stats
    ct = pd.crosstab(binned, cluster)
    ct = ct.loc[ct.sum(axis=1) > 0]
    if min(ct.shape) < 2:
        return float("nan"), int(ct.values.sum())
    chi2 = stats.chi2_contingency(ct)[0]
    n = int(ct.values.sum())
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))), n


def collect(root: str = ROOT, bin_deg: float = 10.0) -> pd.DataFrame:
    rows = []
    for csv in sorted(glob.glob(os.path.join(root, "*", "geographic_clusters_k*.csv"))):
        m = _K_RE.search(os.path.basename(csv))
        if not m:
            continue
        k = int(m.group(1))
        run = os.path.basename(os.path.dirname(csv))
        df = pd.read_csv(csv, dtype={"id": str})
        if not {"lat", "lon", "cluster"} <= set(df.columns):
            continue
        v_lon, n = cramers_v(df["cluster"],
                             pd.cut(df["lon"], np.arange(-180, 180 + bin_deg, bin_deg)))
        v_lat, _ = cramers_v(df["cluster"],
                             pd.cut(df["lat"], np.arange(-60, 60 + bin_deg, bin_deg)))
        # Cluster-size balance: a map whose clusters are wildly uneven can
        # look structured for trivial reasons.
        sizes = df["cluster"].value_counts()
        rows.append({
            "run": run, "k": k, "n_craters": n,
            "V_longitude": v_lon, "V_latitude": v_lat,
            "n_clusters_found": int(df["cluster"].nunique()),
            "largest_cluster_frac": float(sizes.max() / len(df)),
            "csv": csv,
            "geojson": (csv.replace(".csv", ".geojson")
                        if os.path.exists(csv.replace(".csv", ".geojson")) else None),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--bin-deg", type=float, default=10.0)
    ap.add_argument("--out-csv", default=OUT_CSV)
    args = ap.parse_args()

    df = collect(args.root, args.bin_deg)
    if df.empty:
        raise SystemExit(f"no geographic_clusters_k*.csv under {args.root}")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    show = ["run", "k", "V_longitude", "V_latitude", "n_clusters_found",
            "largest_cluster_frac"]
    print(f"Cramer's V, cluster id vs {args.bin_deg:g}-degree geographic bins "
          f"(0 = no association)\n")
    print(df.sort_values(["k", "run"])[show]
            .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
