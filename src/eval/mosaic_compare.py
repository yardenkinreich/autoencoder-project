"""
mosaic_compare.py — the same craters, cut from two different source mosaics,
side by side.

Built for the WAC_GLOBAL-morphology vs WAC_EMP_643-photometrically-normalized
question. The two products differ in a way that matters specifically for this
project and can't be settled by a summary statistic:

  - morphology (WAC_GLOBAL): assembled from three observation windows to hold
    incidence angle in ~55-75 deg, i.e. deliberately RETAINS strong shading,
    because it is built for morphologic interpretation. Comes with
    tile-boundary contrast steps of 23/20/14% at 0/+-90 deg longitude.
  - EMP (WAC_EMP_643): every pixel photometrically normalized to a fixed
    geometry of incidence 30 deg, emission 0 deg, phase 30 deg (Boyd et al.
    2012) using GLD100 topography, then median-stacked over ~142 observations.
    Tile-boundary steps drop to 8/7/0.7%, but incidence 30 deg is near-overhead
    sun, which MINIMIZES topographic shading.

That is the tension: this project's degradation signal is substantially "bright
rim, deep shadow" (see preprocess_2.normalize_global's docstring), and the
normalization that removes the seam also removes some of the shading that
signal rides on. A number can't adjudicate "did we lose the cue" - so this
renders matched crops and lets the eye do it.

Rows are grouped by expert degradation class, so the question can be asked in
the form that actually matters: is the fresh-vs-degraded DIFFERENCE still
visible in EMP, or did normalization flatten the classes together?
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd


def _load_crop(crops_dir: str, crater_id: str) -> np.ndarray | None:
    """A --save_raw_crops crop is a (channels, H, W) .npy already in the
    model's normalized range - same convention visualize._load_crop_for_display
    reads."""
    p = os.path.join(crops_dir, f"{crater_id}.npy")
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return a[0] if a.ndim == 3 else a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops-a", default="data/processed_wac_100m_new/sigma/100/new_test_set/crater_crops",
                    help="Reference crops (morphology mosaic).")
    ap.add_argument("--crops-b", default="data/processed_emp643/sigma/100/new_test_set/crater_crops",
                    help="Comparison crops (EMP mosaic).")
    ap.add_argument("--label-a", default="WAC_GLOBAL morphology")
    ap.add_argument("--label-b", default="WAC_EMP_643 normalized")
    ap.add_argument("--labels-csv", default="configs/liorr_labels.csv")
    ap.add_argument("--class-names", nargs="+", default=["v-fr", "r-fr", "r-dr", "v-dr"])
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = pd.read_csv(args.labels_csv, dtype={"CRATER_ID": str})
    id_col = "CRATER_ID" if "CRATER_ID" in labels.columns else labels.columns[0]
    lab_col = "degree" if "degree" in labels.columns else labels.columns[1]

    rng = np.random.RandomState(args.seed)
    picks: list[tuple[str, str]] = []
    for cls in args.class_names:
        ids = labels.loc[labels[lab_col] == cls, id_col].astype(str).tolist()
        # only craters present in BOTH sets - a missing crop on either side
        # would make the row a comparison of different things
        ids = [i for i in ids
               if os.path.exists(os.path.join(args.crops_a, f"{i}.npy"))
               and os.path.exists(os.path.join(args.crops_b, f"{i}.npy"))]
        if not ids:
            print(f"WARNING: no craters of class {cls} present in both sets")
            continue
        sel = rng.choice(ids, size=min(args.per_class, len(ids)), replace=False)
        picks.extend((cls, str(i)) for i in sel)

    if not picks:
        raise SystemExit("no craters available in both sets - was the EMP preprocess run?")

    ncols = args.per_class
    # two rows per class: mosaic A on top, mosaic B directly beneath it, so
    # the eye compares vertically on the SAME crater rather than across
    # unrelated examples.
    nrows = 2 * len(args.class_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.15 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    stats = []
    for cls_i, cls in enumerate(args.class_names):
        cls_picks = [cid for c, cid in picks if c == cls]
        for j, cid in enumerate(cls_picks):
            for k, (crops, lbl) in enumerate(((args.crops_a, args.label_a),
                                              (args.crops_b, args.label_b))):
                img = _load_crop(crops, cid)
                ax = axes[2 * cls_i + k][j]
                ax.axis("off")
                if img is None:
                    continue
                # Identical display scaling for both rows (the crops are
                # already globally scaled by their own mosaic's percentiles),
                # so an apparent contrast difference is a real difference in
                # the data and not a per-panel autoscale artifact.
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                if j == 0:
                    ax.set_ylabel(lbl, fontsize=7)
                    ax.axis("on")
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_color("#cccccc")
                if k == 0:
                    ax.set_title(f"{cls}  {cid}", fontsize=6.5)
                stats.append({"class": cls, "crater_id": cid, "mosaic": lbl,
                              "mean": float(np.nanmean(img)),
                              "std": float(np.nanstd(img)),
                              "p2_p98_range": float(np.nanpercentile(img, 98)
                                                    - np.nanpercentile(img, 2))})

    fig.suptitle(f"Same craters, two source mosaics — top row of each pair: {args.label_a}; "
                 f"bottom: {args.label_b}\nidentical display scaling (0-1) in every panel",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sdf = pd.DataFrame(stats)
    summary = (sdf.groupby(["mosaic", "class"])
                  .agg(mean=("mean", "mean"), std=("std", "mean"),
                       p2_p98_range=("p2_p98_range", "mean"), n=("crater_id", "size"))
                  .reset_index())
    print("\n=== per-class crop statistics (the numbers behind the picture) ===")
    print(summary.to_string(index=False))
    csv = os.path.splitext(args.out)[0] + "_stats.csv"
    summary.to_csv(csv, index=False)

    # The decisive quantity: does contrast still SEPARATE the classes in each
    # mosaic? If EMP's fresh-to-degraded spread collapses relative to
    # morphology's, normalization cost us the cue.
    print("\n=== fresh->degraded spread in mean per-crop std (higher = classes more distinct) ===")
    for m in summary["mosaic"].unique():
        sub = summary[summary["mosaic"] == m].set_index("class")
        vals = [sub.loc[c, "std"] for c in args.class_names if c in sub.index]
        print(f"  {m:28} per-class std: "
              f"{', '.join(f'{c}={v:.4f}' for c, v in zip(args.class_names, vals))}"
              f"   spread={max(vals) - min(vals):.4f}")
    print(f"\nwrote {args.out}\n      {csv}")


if __name__ == "__main__":
    main()
