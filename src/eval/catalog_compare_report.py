"""
catalog_compare_report.py — collapse the per-catalog eval-suite summaries into
one table ranked by the things that actually matter for choosing a data
representation.

WHAT IT RANKS BY, AND WHY NOT ACCURACY
--------------------------------------
Accuracy and recall are the wrong headline here twice over. First, they are
Hungarian-aligned cluster agreement scores on a set that is 63% one class, so
they move for reasons that have nothing to do with representation quality.
Second, and worse: a supervised probe on these same embeddings reads ~0.23
accuracy HIGHER than the unsupervised suite, because it fits a classifier to
whatever information is present however it is arranged - which is precisely
the question we are not asking. What we want to know is whether the
representation ORGANIZES itself by degradation.

So the table leads with:

  silhouette_true  - do the true classes form separated groups in latent space
  map              - mean average precision
  ari / nmi        - agreement of unsupervised clusters with the truth

and pairs every one of them with the confound columns, because a score is
uninterpretable without them:

  lon_eta / lat_eta        - is the latent space organized by GEOGRAPHY
                             (the acquisition artifact - the thing to minimize)
  brightness / sobel /
  frac_saturated eta       - or by intensity shortcuts
  diam_eta                 - or simply by crater size

THE TRADE-OFF IS REAL, SO THE TABLE SHOWS BOTH SIDES
----------------------------------------------------
Across 67 previously recorded evaluations, corr(lon_eta, silhouette_true) =
+0.248 (p = 0.043): representations with MORE longitude artifact score
slightly BETTER separation, because part of the apparent class structure is
riding on the artifact. A change that genuinely removes the artifact should
therefore be expected to lower separation somewhat. Reading either column
alone will mislead; that is why `score` below is reported next to `lon_eta`
and never collapsed into a single number.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "logs/data_profiles/catalog_eval_suite"
OUT_CSV = "logs/data_profiles/catalog_compare.csv"


def _metric(block: dict, key: str):
    """Bootstrapped metrics live at the TOP LEVEL of each labeled set (not
    under a "metrics" sub-dict) as {"point", "lo", "hi", "n_boot"}; a metric
    that was not bootstrapped is a bare scalar. Returns (point, lo, hi)."""
    v = block.get(key)
    if isinstance(v, dict):
        return v.get("point"), v.get("lo"), v.get("hi")
    return v, None, None


def collect(root: str = ROOT) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        d = json.load(open(p))
        for sname, s in (d.get("labeled_sets") or {}).items():
            cc = s.get("confound_correlation") or {}
            gc = s.get("geo_correlation") or {}
            sep = s.get("separation") or {}
            cat, _, norm = sname.partition("__")
            mp, mp_lo, mp_hi = _metric(s, "map")
            acc, _, _ = _metric(s, "accuracy")
            qwk, _, _ = _metric(s, "quadratic_weighted_kappa")

            def eta(name):
                return (cc.get(name) or {}).get("eta_sq")

            def r(name):
                return (gc.get(name) or {}).get("r")

            rows.append({
                "catalog": cat, "norm": norm,
                "silhouette_true": sep.get("silhouette_true"),
                "map": mp, "map_lo": mp_lo, "map_hi": mp_hi,
                # ARI/NMI/V-measure are point estimates under "agreement",
                # not bootstrapped like accuracy/QWK/mAP.
                "ari": (s.get("agreement") or {}).get("ari"),
                "nmi": (s.get("agreement") or {}).get("nmi"),
                "accuracy": acc, "qwk": qwk,
                "purity": sep.get("purity"),
                "lon_eta": eta("lon"), "lat_eta": eta("lat"),
                "brightness_eta": eta("brightness"), "sobel_eta": eta("sobel_mean"),
                "frac_sat_eta": eta("frac_saturated"), "diam_eta": eta("diam_km"),
                "pc1_vs_lon": r("pc1_vs_lon"), "pc2_vs_lat": r("pc2_vs_lat"),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-csv", default=OUT_CSV)
    args = ap.parse_args()

    df = collect(args.root)
    if df.empty:
        raise SystemExit(f"no summary.json found under {args.root}")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"{len(df)} cells from {df['catalog'].nunique()} catalogs "
          f"x {df['norm'].nunique()} normalizations\n")

    show = ["catalog", "norm", "silhouette_true", "map", "ari", "nmi",
            "lon_eta", "lat_eta", "brightness_eta", "sobel_eta", "diam_eta"]
    fmt = {c: "{:.3f}" for c in show if c not in ("catalog", "norm")}

    print("=== ranked by SILHOUETTE_TRUE (class separation in latent space) ===")
    print(df.sort_values("silhouette_true", ascending=False)[show]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== ranked by LON_ETA, ascending (least geographic artifact) ===")
    print(df.sort_values("lon_eta")[show]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # The cells that are good on BOTH axes. Reported as a set rather than a
    # single winner because the two objectives genuinely conflict (see the
    # module docstring's +0.248 correlation) and a weighted sum would hide
    # which side any given cell won on.
    ms, ml = df["silhouette_true"].median(), df["lon_eta"].median()
    both = df[(df["silhouette_true"] > ms) & (df["lon_eta"] < ml)]
    print(f"\n=== ABOVE-median separation AND BELOW-median lon artifact "
          f"(sil>{ms:.3f}, lon<{ml:.3f}): {len(both)} of {len(df)} ===")
    if len(both):
        print(both.sort_values("silhouette_true", ascending=False)[show]
                  .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("  none - no cell wins on both axes.")

    print("\n=== marginal means ===")
    for key in ("catalog", "norm"):
        g = (df.groupby(key)[["silhouette_true", "map", "ari",
                              "lon_eta", "brightness_eta", "sobel_eta"]]
               .mean().sort_values("lon_eta"))
        print(f"\nby {key} (sorted by least longitude artifact):")
        print(g.to_string(float_format=lambda v: f"{v:.3f}"))

    from scipy import stats
    ok = df[["lon_eta", "silhouette_true"]].dropna()
    if len(ok) > 2:
        r, p = stats.pearsonr(ok["lon_eta"], ok["silhouette_true"])
        print(f"\ncorr(lon_eta, silhouette_true) within these cells: "
              f"r={r:.3f} p={p:.3g} n={len(ok)}")
        print("  (positive => separation is partly riding on the artifact, so a "
              "genuine fix should be expected to cost some separation)")
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
