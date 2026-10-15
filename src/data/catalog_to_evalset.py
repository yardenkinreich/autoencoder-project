"""
catalog_to_evalset.py — materialize a catalog's 642-crater testset, under one
chosen normalization, into the memmap triple that src/eval/evaluate.py's
labeled-set contract expects.

WHY THIS BRIDGE EXISTS
----------------------
The catalogs deliberately store crops RAW and at NATIVE, per-crater-variable
size (see build_catalog.py). The eval suite deliberately wants a FIXED-size
float32 memmap plus a row-aligned metadata.csv (see
eval/pipeline.labels_from_memmap_csv and eval/holdout.load_holdout_dat, which
hardcodes 128). Both choices are right for their own purpose, so something has
to convert - and this is that something, in one place, so no evaluation ever
reimplements the resize/normalize decision.

WHAT IT DOES *NOT* DO
---------------------
It does not decide the normalization. `norm` is required and unguarded on
purpose: the whole point of the catalogs is that the presentation is a recorded
choice, and a default here would quietly reintroduce the thing we spent this
session removing. Constants come from the PARENT catalog's norm_stats.json
(stats_from), never the testset's own - fitting normalization on the evaluation
set would let test data define its own scale.

WHY THE FULL SUITE AND NOT A PROBE
----------------------------------
A supervised probe asks "is the label linearly decodable from the embedding",
which is both easier than and different from the question this project is
about: does the representation ORGANIZE itself by degradation without
supervision. The eval suite answers the latter - silhouette of the true classes
in latent space, mAP, ARI/NMI from Hungarian-aligned k-means clusters, QWK,
ordinal MAE - and it also carries the two confound checks that matter most
here for free:

  latent_geo_correlation  -> is the latent space organized by lat/lon?
                             (the acquisition-artifact question)
  confound_correlation    -> is it organized by brightness?
                             (the maria/highland albedo question)

Measured on the 642 labeled craters, a single scalar (crop mean of a
sigma-100 residual) already reaches 0.64 macro recall against frozen DINOv2's
0.65, so a score alone cannot distinguish "read the morphology" from "read the
brightness". The confound columns are what separate them.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

# Matches eval/holdout.load_holdout_dat's hardcoded expectation. Changing it
# here alone would produce a memmap the eval suite silently mis-shapes, since
# that loader infers N from file size and a fixed pixel count.
EVAL_SIZE = 128
EVAL_CHANNELS = 1
MOON_RADIUS_M = 1737400.0


def build(testset_dir: str, norm: str, out_dir: str, *,
          size: int = EVAL_SIZE, stats_from: str | None = None,
          save_crops: bool = True, resize_interp: str = "cubic") -> dict:
    from src.data.crater_catalog_dataset import CraterCatalog

    parent = stats_from or os.path.dirname(testset_dir.rstrip("/"))
    ds = CraterCatalog(root=testset_dir, norm=norm, resize=size,
                       stats_from=parent, num_channels=EVAL_CHANNELS,
                       resize_interp=resize_interp)
    idx = ds.index
    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "crater_crops")
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    dat_path = os.path.join(out_dir, "craters.dat")
    rows = []
    lo = hi = None
    with open(dat_path, "wb") as f:
        for i in range(len(ds)):
            a = np.asarray(ds[i][0], dtype=np.float32)   # (C, size, size)
            if a.shape != (EVAL_CHANNELS, size, size):
                raise ValueError(f"row {i}: got {a.shape}, expected "
                                 f"{(EVAL_CHANNELS, size, size)}")
            if not np.isfinite(a).all():
                raise ValueError(f"row {i} ({idx.iloc[i]['crater_id']}): non-finite "
                                 f"pixels after normalization - refusing to write an "
                                 f"eval set that would silently produce NaN metrics.")
            f.write(a.tobytes(order="C"))
            lo = float(a.min()) if lo is None else min(lo, float(a.min()))
            hi = float(a.max()) if hi is None else max(hi, float(a.max()))

            r = idx.iloc[i]
            cid = str(r["crater_id"])
            lat, lon = float(r["lat"]), float(r["lon"])
            # metadata.csv's row order MUST stay positionally aligned with
            # craters.dat - labels_from_memmap_csv returns a row_idx that
            # indexes straight into the memmap. Both are written in this one
            # loop for exactly that reason.
            #
            # x/y are a plain equirectangular projection of lat/lon, recomputed
            # here rather than carried through the catalog. They are read only
            # by geographic_map.spatial_concentration (a relative,
            # scale-invariant measure), never by the metric suite.
            rows.append({
                "id": cid, "lat": lat,
                # eval convention is lon in [-180, 180]; the catalog stores the
                # Robbins database's [0, 360].
                "lon": lon - 360.0 if lon > 180.0 else lon,
                "x": MOON_RADIUS_M * np.deg2rad(lon - 360.0 if lon > 180.0 else lon),
                "y": MOON_RADIUS_M * np.deg2rad(lat),
                "diam": float(r["diam_km"]) * 1000.0,   # metres, matching preprocess_2
            })
            if save_crops:
                np.save(os.path.join(crops_dir, f"{cid}.npy"), a)

    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    manifest = {
        "source_testset": testset_dir, "stats_from": parent,
        "norm": norm, "norm_constants": ds.policy.get("norm_constants"),
        "size": size, "channels": EVAL_CHANNELS, "resize_interp": resize_interp,
        "n_craters": int(len(ds)),
        "value_range_after_norm": [lo, hi],
        "dat_layout": f"float32 (N, {EVAL_CHANNELS}, {size}, {size}) C-order; "
                      f"row i == metadata.csv row i",
        "resize_note": "resize BEFORE normalize (crater_catalog_dataset."
                       "get_image_data) - the clip does not commute with cubic "
                       "interpolation.",
    }
    json.dump(manifest, open(os.path.join(out_dir, "evalset_manifest.json"), "w"),
              indent=2)
    print(f"  {out_dir}: n={manifest['n_craters']} norm={norm} "
          f"range=[{lo:.3f}, {hi:.3f}]")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogs", nargs="+", required=True,
                    help="Catalog directories (the parent, not testset/).")
    ap.add_argument("--norms", nargs="+", required=True,
                    help="Normalizations to materialize. No default - see the "
                         "module docstring.")
    ap.add_argument("--out-root", default="data/evalsets")
    ap.add_argument("--use-catalog-root", action="store_true",
                    help="Materialize the catalog's FULL pool (~100k craters) instead "
                         "of its 642-crater testset/. For the geographic map, which "
                         "needs the whole pool and no labels. Note the size: ~6.5 GB "
                         "of memmap per catalog at 128x128 float32.")
    ap.add_argument("--no-crops", action="store_true",
                    help="Skip the per-crater .npy display crops. Required in practice "
                         "for a 100k pool - they are only used for display and "
                         "confound checks on labeled sets, and 100k of them is both "
                         "slow and pointless here.")
    ap.add_argument("--size", type=int, default=EVAL_SIZE)
    ap.add_argument("--resize-interp", choices=["cubic", "auto"], default="cubic",
                    help="auto = area-average when shrinking (use for native-resolution "
                         "Kaguya catalogs); cubic = the WAC eval sets' behaviour.")
    ap.add_argument("--labels-csv", default="configs/liorr_labels.csv")
    ap.add_argument("--config-out", default="configs/eval_suite_catalogs.yaml",
                    help="Write an eval-suite config covering every set built.")
    args = ap.parse_args()

    built = []
    for cat in args.catalogs:
        # The parent catalog and its testset/ have the same on-disk shape, so
        # build() reads either. stats_from is the parent in BOTH cases, so a
        # test set never fits its own scaling.
        src = cat if args.use_catalog_root else os.path.join(cat, "testset")
        if not os.path.exists(os.path.join(src, "index.csv")):
            print(f"skip {cat}: no {src}/index.csv")
            continue
        suffix = "__pool" if args.use_catalog_root else ""
        for norm in args.norms:
            name = f"{os.path.basename(cat.rstrip('/'))}__{norm}{suffix}"
            out = os.path.join(args.out_root, name)
            m = build(src, norm, out, size=args.size, stats_from=cat,
                      save_crops=not args.no_crops, resize_interp=args.resize_interp)
            built.append({"name": name, "dir": out, **m})

    if not built:
        raise SystemExit("nothing built")

    # An eval-suite config listing every set, so one evaluate.py invocation
    # scores a checkpoint on all of them and the numbers land in one place.
    import yaml
    cfg = {
        "class_names": ["v-fr", "r-fr", "r-dr", "v-dr"],
        "n_clusters": 4,
        "seed": 0,
        "scaling_json": "configs/global_scaling.json",
        "png_max": 255,
        "labeled_sets": {
            b["name"]: {
                "dat_path": os.path.join(b["dir"], "craters.dat"),
                "metadata_csv": os.path.join(b["dir"], "metadata.csv"),
                "imgs_dir": os.path.join(b["dir"], "crater_crops"),
                "labels_csv": args.labels_csv,
                "class_names": ["v-fr", "r-fr", "r-dr", "v-dr"],
            } for b in built
        },
    }
    with open(args.config_out, "w") as f:
        f.write("# GENERATED by src/data/catalog_to_evalset.py - do not hand-edit.\n"
                "# One labeled set per (catalog x normalization), all on the SAME 642\n"
                "# craters, so a difference between sets is a difference in data\n"
                "# presentation and nothing else. Every set is 4-class, so n_clusters\n"
                "# is 4 here rather than eval_suite.yaml's 3-class default.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\nbuilt {len(built)} eval sets -> {args.out_root}")
    print(f"wrote {args.config_out}")


if __name__ == "__main__":
    main()
