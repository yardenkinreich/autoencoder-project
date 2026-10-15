"""
backfill_geojson.py — regenerate the GeoJSON + .prj + zip exports for
geographic-map runs that predate them, from the per-crater CSVs already on
disk.

The cluster assignments were always saved (`geographic_clusters_k*.csv`); only
the export was missing. So no re-embedding and no re-clustering is needed - and
must not happen, because re-running KMeans would assign different arbitrary
cluster ids and silently invalidate any figure already published from the old
CSV.

    PYTHONPATH=. python -m src.eval.backfill_geojson --scan logs
    PYTHONPATH=. python -m src.eval.backfill_geojson --dir logs/.../geographic_map
"""
from __future__ import annotations
import argparse
import glob
import os
import re

import pandas as pd

from src.eval.geographic_map import (
    write_geojson, write_gpkg, craters_crs_wkt, raster_crs_wkt, _DEFAULT_BASEMAP,
)

_K_RE = re.compile(r"geographic_clusters_k(\d+)\.csv$")


def backfill_one(csv_path: str, *, label: str | None = None,
                 overwrite: bool = False, crs_from: str = _DEFAULT_BASEMAP) -> str | None:
    m = _K_RE.search(os.path.basename(csv_path))
    if not m:
        return None
    k = int(m.group(1))
    d = os.path.dirname(csv_path)
    gpkg_probe = os.path.join(d, f"geographic_clusters_k{k}.gpkg")
    if os.path.exists(gpkg_probe) and not overwrite:
        print(f"  skip (exists): {gpkg_probe}")
        return None

    df = pd.read_csv(csv_path, dtype={"id": str})
    need = {"id", "lat", "lon", "cluster"}
    missing = need - set(df.columns)
    if missing:
        print(f"  skip {csv_path}: missing columns {sorted(missing)}")
        return None
    if "diam" not in df.columns:
        df["diam"] = float("nan")

    # Label from the run directory when not given - the geographic_map dir sits
    # under the checkpoint's run dir, whose path encodes the model.
    if label is None:
        run_dir = os.path.dirname(os.path.dirname(csv_path.rstrip(os.sep)))
        label = os.path.basename(run_dir) or "geographic map"

    # PROJECTED first, since that is the one that actually overlays on the
    # mosaic. It needs x/y; every clusters CSV has them, but be explicit
    # rather than producing a silently-empty geometry.
    if not {"x", "y"} <= set(df.columns):
        print(f"  skip {csv_path}: no x/y columns, cannot write a GIS layer")
        return None
    _, pw2 = raster_crs_wkt(crs_from)

    # GeoPackage ONLY. A single file with the CRS embedded internally, already
    # in the mosaic's own projection so QGIS reprojects nothing. The GeoJSON
    # variants this used to also emit were strictly worse at the same job -
    # they needed a .prj sidecar that travels separately, relied on a legacy
    # `crs` member RFC 7946 removed, and forced a choice between two files
    # that looked equally plausible. Anything needing lon/lat has it as
    # attribute columns here and in the CSV.
    gpkg_out = os.path.join(d, f"geographic_clusters_k{k}.gpkg")
    write_gpkg(df, gpkg_out, label=label, k=k, crs_wkt=pw2)
    print(f"  wrote {gpkg_out}  n={len(df):,} k={k}")

    # Remove every superseded export so nobody loads a stale or ambiguous one.
    base = os.path.join(d, f"geographic_clusters_k{k}")
    for stale in (f"{base}.geojson", f"{base}.prj", f"{base}_geojson.zip",
                  f"{base}_projected.geojson", f"{base}_projected.prj",
                  f"{base}_projected_geojson.zip",
                  f"{base}_geographic.geojson", f"{base}_geographic.prj",
                  f"{base}_geographic_geojson.zip"):
        if os.path.exists(stale):
            os.remove(stale)
            print(f"  removed superseded {os.path.basename(stale)}")
    return gpkg_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default=None,
                    help="Root to search recursively for geographic_clusters_k*.csv.")
    ap.add_argument("--dir", nargs="*", default=None,
                    help="Specific geographic_map directories.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--crs-from", default=_DEFAULT_BASEMAP,
                    help="Raster whose CRS the projected export uses - must be the "
                         "mosaic the points will be displayed over.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rewrite exports that already exist.")
    args = ap.parse_args()

    if not args.scan and not args.dir:
        args.scan = "logs"

    csvs: list[str] = []
    if args.scan:
        csvs += glob.glob(os.path.join(args.scan, "**", "geographic_clusters_k*.csv"),
                          recursive=True)
    for d in (args.dir or []):
        csvs += glob.glob(os.path.join(d, "geographic_clusters_k*.csv"))
    # A *_summary.csv also matches the glob's prefix; exclude it.
    csvs = sorted({c for c in csvs if _K_RE.search(os.path.basename(c))})

    if not csvs:
        raise SystemExit("no geographic_clusters_k*.csv found")
    print(f"CRS being applied (from preprocess_2.get_craters_crs()):\n"
          f"  {craters_crs_wkt()[:120]}...\n")
    print(f"{len(csvs)} cluster CSVs found")
    n = 0
    for c in csvs:
        print(f"\n{c}")
        if backfill_one(c, label=args.label, overwrite=args.overwrite,
                        crs_from=args.crs_from):
            n += 1
    print(f"\nbackfilled {n} of {len(csvs)}")


if __name__ == "__main__":
    main()
