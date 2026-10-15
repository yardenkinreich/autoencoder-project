"""
build_catalog.py — native-resolution, raw-valued crater catalogs.

One catalog = one (source mosaic, filter) pair. Six of them span the
comparison this exists for: {morphology, EMP_643} x {raw, sigma100, sigma200}.

WHY THIS RATHER THAN preprocess_2.py
------------------------------------
preprocess_2.py resizes every crop to a fixed OUTPUT_SIZE (128) and writes a
fixed-shape memmap. That is wrong for this purpose in two ways:

  1. The median native crop at 1-12 km / fov 1.0 is only 32x32 px, so
     resizing to 128 UPSAMPLES the majority of craters 3-4x - inventing no
     information while inflating storage ~6x (67 GB vs 11 GB per million
     craters). At the small end a 20x20 crop becomes 9x9 patches of pure
     interpolation for a patch-14 ViT.
  2. It also means two resamplings on the way into a model (native ->128 at
     preprocess, then RandomResizedCrop ->224 at load). Storing native gives
     one.

So this writes RAGGED output: crops concatenated into a single flat float32
binary, plus an index carrying each crater's byte offset and native shape.

WHY RAW, NOT NORMALIZED
-----------------------
Values are stored exactly as the mosaic holds them (all six source mosaics
are already float32, so this is byte-faithful - normalize()'s "raw" mode is
literally astype(float32), a no-op). Global normalization is an affine map
plus a CLIP at the 1st/99th percentiles; the affine part is recoverable, the
clip is not. Storing raw keeps the ability to refit percentiles, try
z-scoring, or run the artifact analysis on unclipped values - and the
per-crater statistics in the index are computed pre-clip for exactly that
reason.

The cost is that raw crops are NOT model-ready: feeding a +-0.05 high-pass
residual to a network expecting [0,1] gives near-zero input variance and
degenerate features, silently. load_catalog() below is therefore the only
sanctioned reader and REQUIRES the scaling, rather than defaulting it.

WHAT IS STILL A RESAMPLING EVEN IN "RAW" MODE
---------------------------------------------
crop_crater() stretches each crop vertically by 1/cos(lat) to undo
equirectangular distortion. That is a geometric correction, not a value
change, and it is necessary (otherwise high-latitude craters are squashed) -
but it IS an interpolation and the manifest records it.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import subprocess

import numpy as np
import pandas as pd


CROPS_FILE = "crops.f32"
INDEX_FILE = "index.csv"
SCALING_FILE = "scaling.json"
MANIFEST_FILE = "catalog_manifest.json"

# Anything of this magnitude or larger is a nodata sentinel, not a
# measurement. WAC_EMP_643 declares nodata = -3.4028e38, which is FINITE and
# therefore invisible to np.isfinite(); real WAC reflectance and high-pass
# residuals are order 1e-2. Same constant and same reasoning as
# highpass_mosaic.SENTINEL_MAG.
SENTINEL_MAG = 1e30


# ── reading ──────────────────────────────────────────────────────────────────

def load_index(catalog_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(catalog_dir, INDEX_FILE), dtype={"crater_id": str})


def load_scaling(catalog_dir: str) -> dict:
    return json.load(open(os.path.join(catalog_dir, SCALING_FILE)))


def load_catalog(catalog_dir: str, scaling: dict | None, rows=None,
                 out_size: int | None = None, assert_range: bool = True):
    """The ONLY sanctioned way to read a catalog's pixels.

    scaling is REQUIRED and not defaulted, deliberately: pass the dict from
    load_scaling() to get model-ready [0,1] crops, or pass None to state
    explicitly that you want raw values (the artifact analysis does this).
    A silent default here is precisely how an unnormalized dataset ends up
    inside a model with nothing raising.

    out_size: if given, every crop is resized to (out_size, out_size) so the
    result is a stacked array; otherwise a list of native-shaped arrays is
    returned, since native crops have different shapes per crater.
    """
    import cv2
    idx = load_index(catalog_dir)
    if rows is not None:
        idx = idx.iloc[rows]
    path = os.path.join(catalog_dir, CROPS_FILE)
    out = []
    with open(path, "rb") as f:
        for r in idx.itertuples():
            f.seek(int(r.byte_offset))
            a = np.frombuffer(f.read(int(r.h) * int(r.w) * 4), dtype=np.float32)
            a = a.reshape(int(r.h), int(r.w))
            # Resize BEFORE normalize: the affine part commutes with
            # interpolation but the clip does not, so clipping first lets
            # INTER_CUBIC's overshoot escape [0,1]. See
            # crater_catalog_dataset.get_image_data for the measured case.
            if out_size is not None:
                a = cv2.resize(a, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
            if scaling is not None:
                lo, hi = float(scaling["lo"]), float(scaling["hi"])
                a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
            out.append(a)
    if scaling is not None and assert_range and out:
        mn = min(float(x.min()) for x in out)
        mx = max(float(x.max()) for x in out)
        if not (-1e-6 <= mn and mx <= 1.0 + 1e-6):
            raise ValueError(f"normalized crops out of [0,1]: min={mn} max={mx}")
    return np.stack(out) if out_size is not None else out


# ── writing ──────────────────────────────────────────────────────────────────

def _crop_stats(a: np.ndarray) -> dict:
    """Per-crater statistics on RAW values, pre-clip. These - not the pixels -
    are what the longitude/latitude artifact analysis consumes, so it reads a
    ~20 MB CSV instead of scanning a multi-GB pixel store."""
    flat = a.reshape(-1)
    p1, p50, p99 = np.percentile(flat, [1, 50, 99])
    return {
        "mean": float(flat.mean()), "std": float(flat.std()),
        # mean |horizontal neighbour difference|: a cheap local-contrast /
        # sharpness proxy, the same measure the seam analysis used
        "contrast": float(np.abs(np.diff(a, axis=-1)).mean()) if a.shape[-1] > 1 else 0.0,
        "p1": float(p1), "p50": float(p50), "p99": float(p99),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _wac_crops(map_file, craters, clean_offset):
    """The original source: one global mosaic via crop_crater. Yields
    (row, crop or None, extra fields, error or None) in crater order."""
    import pyproj
    import rasterio
    from src.helper_functions import crop_crater
    from src.data.preprocess_2 import get_craters_crs
    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            get_craters_crs(), map_ref.crs.to_string(), always_xy=True)
        for _, c in craters.iterrows():
            try:
                a = crop_crater(map_ref, c["LAT_CIRC_IMG"], c["LON_CIRC_IMG"],
                                c["DIAM_CIRC_IMG"], clean_offset, transformer)
                yield c, a, {}, None
            except Exception as e:
                yield c, None, {}, f"{type(e).__name__}: {e}"


_CROPPER = None


def _cropper_init(cropper):
    global _CROPPER
    _CROPPER = cropper


def _cropper_job(args):
    lat, lon, diam, offset = args
    try:
        a, extra = _CROPPER.crop(lat, lon, diam, offset)
        return a, extra, None
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {e}"


def _cropper_crops(cropper, craters, clean_offset, workers):
    """A pluggable cropper (e.g. kaguya_tco.KaguyaCropper), optionally over a
    process pool. imap is ORDERED, so crater order - and therefore the
    byte layout - is identical for any worker count."""
    from multiprocessing import Pool
    jobs = [(float(c["LAT_CIRC_IMG"]), float(c["LON_CIRC_IMG"]),
             float(c["DIAM_CIRC_IMG"]), clean_offset) for _, c in craters.iterrows()]
    rows = (c for _, c in craters.iterrows())
    if workers <= 1:
        _cropper_init(cropper)
        for c, j in zip(rows, jobs):
            yield (c, *_cropper_job(j))
        return
    with Pool(workers, initializer=_cropper_init, initargs=(cropper,)) as pool:
        for c, r in zip(rows, pool.imap(_cropper_job, jobs, chunksize=4)):
            yield (c, *r)


def build(map_file: str, craters_csv: str, out_dir: str, *,
          min_diameter: float, max_diameter: float, lat_bounds: tuple,
          craters_to_output: int, clean_offset: float,
          exclude_crater_ids=None, only_crater_ids=None,
          exclude_lon_bounds=None, filter_desc: str = "none",
          cropper=None, workers: int = 1) -> pd.DataFrame:
    """cropper=None crops map_file with crop_crater (the WAC catalogs).
    Otherwise cropper.crop(lat, lon, diam_km, offset) -> (array, extra dict)
    supplies the pixels and extra per-crater index columns; a crop it cannot
    make raises, and the crater is recorded in skipped_invalid.csv."""
    from src.data.preprocess_2 import load_and_filter_craters

    os.makedirs(out_dir, exist_ok=True)

    # Identical selection semantics to preprocess_2.py, including its
    # filtered.sample(random_state=42) - so every catalog built with the same
    # arguments contains the SAME craters, and catalog differences can never
    # be confounded with sampling differences.
    craters = load_and_filter_craters(
        craters_csv, min_diameter, max_diameter, lat_bounds, craters_to_output,
        exclude_lon_bounds=exclude_lon_bounds,
        exclude_crater_ids=exclude_crater_ids, only_crater_ids=only_crater_ids)
    n = len(craters)
    print(f"  selected {n:,} craters")

    rows = []
    skipped_invalid: list[dict] = []
    offset = 0
    crops_path = os.path.join(out_dir, CROPS_FILE)
    crops = (_wac_crops(map_file, craters, clean_offset) if cropper is None
             else _cropper_crops(cropper, craters, clean_offset, workers))
    with open(crops_path, "wb") as fout:
        for i, (c, a, extra, err) in enumerate(crops):
            if i % 10_000 == 0:
                print(f"  {i:,}/{n:,}", flush=True)
            lat, lon, diam, cid = (c["LAT_CIRC_IMG"], c["LON_CIRC_IMG"],
                                   c["DIAM_CIRC_IMG"], c["CRATER_ID"])
            if err is not None:
                print(f"    skip {cid}: {err}")
                if cropper is not None:
                    # A pluggable source records every crater it cannot cut
                    # (e.g. outside the Kaguya 60S-60N coverage), so the
                    # missing set stays provenance, not a log line.
                    skipped_invalid.append({"crater_id": str(cid), "n_bad": -1, "n_px": 0,
                                            "lat": float(lat), "lon": float(lon),
                                            "diam_km": float(diam), "reason": err})
                continue
            if a is None or a.size == 0 or a.ndim != 2:
                continue
            a = np.ascontiguousarray(a, dtype=np.float32)
            # NOT `np.isfinite(a).all()` alone. WAC_EMP_643 declares
            # nodata = -3.4028e38, and that value IS finite - so an isfinite
            # check passes it straight through into storage. Five crops out of
            # 100,000 in the first emp_raw build carried it, and those five
            # were enough to drive the catalog's mean to -inf and every
            # zscore/asinh constant to inf. (Their p1/p50/p99 looked perfectly
            # normal, because a percentile is immune to a handful of extreme
            # values in a way a mean is not - which is why the failure was
            # invisible until the constants were fitted.)
            #
            # So validity is judged by MAGNITUDE, matching
            # highpass_mosaic.py's SENTINEL_MAG guard. Real WAC reflectance and
            # high-pass residuals are order 1e-2; 1e30 is many orders of
            # magnitude past anything physical, so this cannot reject real data.
            bad = ~np.isfinite(a) | (np.abs(a) >= SENTINEL_MAG)
            n_bad = int(bad.sum())
            if n_bad:
                # Skip the crop entirely rather than patch it. A crater whose
                # crop overlaps a mosaic hole is not a measurement of that
                # crater, and substituting zeros would quietly manufacture a
                # flat region that the model would read as morphology.
                skipped_invalid.append({"crater_id": str(cid), "n_bad": n_bad,
                                        "n_px": int(a.size), "lat": float(lat),
                                        "lon": float(lon), "diam_km": float(diam),
                                        "reason": "nodata pixels"})
                continue
            fout.write(a.tobytes(order="C"))
            rows.append({"crater_id": str(cid), "byte_offset": offset,
                         "h": a.shape[0], "w": a.shape[1],
                         "lat": float(lat), "lon": float(lon),
                         "diam_km": float(diam), **_crop_stats(a), **extra})
            offset += a.nbytes
    idx = pd.DataFrame(rows)
    idx.to_csv(os.path.join(out_dir, INDEX_FILE), index=False)
    print(f"  wrote {len(idx):,} crops, {offset/1e9:.2f} GB -> {crops_path}")
    if skipped_invalid:
        # Recorded as an artifact, not just a log line: which craters a
        # catalog is MISSING is provenance, and it is also what makes the
        # catalogs comparable - a crater dropped here must be dropped from
        # every catalog, or the "same 100k craters everywhere" guarantee that
        # justifies comparing them silently stops holding.
        p = os.path.join(out_dir, "skipped_invalid.csv")
        pd.DataFrame(skipped_invalid).to_csv(p, index=False)
        print(f"  SKIPPED {len(skipped_invalid):,} crops containing nodata/sentinel "
              f"pixels (|v|>={SENTINEL_MAG:g}) -> {p}")
        print(f"  NOTE: pass --exclude_invalid_from {p} to the other catalog builds "
              f"so all catalogs hold the same crater set.")
    return idx


def fit_scaling(idx: pd.DataFrame, catalog_dir: str, sample: int = 20000,
                lo_pct: float = 1.0, hi_pct: float = 99.0, seed: int = 0) -> dict:
    """Fixed 1st/99th percentiles over a crater subsample, matching
    fit_global_scaling.py's convention. nan-aware and refuses to emit a
    non-finite result - a NaN lo/hi silently normalizes every crop to NaN."""
    rng = np.random.default_rng(seed)
    take = rng.choice(len(idx), size=min(sample, len(idx)), replace=False)
    crops = load_catalog(catalog_dir, scaling=None, rows=np.sort(take))
    pixels = np.concatenate([c.reshape(-1) for c in crops])
    n_nan = int(np.isnan(pixels).sum())
    lo = float(np.nanpercentile(pixels, lo_pct))
    hi = float(np.nanpercentile(pixels, hi_pct))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError(f"bad percentiles lo={lo} hi={hi} (n_nan={n_nan})")
    out = {"lo": lo, "hi": hi, "lo_pct": lo_pct, "hi_pct": hi_pct,
           "n_craters_sampled": int(len(take)), "n_pixels": int(pixels.size),
           "n_nan_sampled": n_nan,
           "raw_min": float(np.nanmin(pixels)), "raw_max": float(np.nanmax(pixels))}
    json.dump(out, open(os.path.join(catalog_dir, SCALING_FILE), "w"), indent=2)
    print(f"  scaling: lo={lo:.6g} hi={hi:.6g} (raw {out['raw_min']:.6g}..{out['raw_max']:.6g})")
    return out


def write_manifest(out_dir: str, **fields) -> None:
    import rasterio
    mf = fields.get("map_file")
    src_info = fields.pop("source_mosaic_info", {})
    if mf and os.path.isfile(mf):
        st = os.stat(mf)
        with rasterio.open(mf) as s:
            src_info = {"dtype": s.dtypes[0], "nodata": (None if s.nodata is None else float(s.nodata)),
                        "res_m": float(s.res[0]), "width": s.width, "height": s.height}
        src_info.update({"size_bytes": st.st_size,
                         "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
    # Caller-supplied values/geometry_note/size_policy (a non-WAC source)
    # override the WAC defaults below; everything else is unchanged.
    overrides = {k: fields.pop(k) for k in ("values", "geometry_note", "size_policy", "layout")
                 if k in fields}
    m = {
        **fields,
        "source_mosaic_info": src_info,
        "values": "RAW - exactly as stored in the source mosaic (all sources are "
                  "float32; preprocess_2.normalize() raw mode is a no-op cast). "
                  "NOT normalized; apply scaling.json via build_catalog.load_catalog().",
        "geometry_note": "crop_crater() stretches each crop by 1/cos(lat) to undo "
                         "equirectangular distortion - a geometric interpolation applied "
                         "even in raw mode.",
        "size_policy": "NATIVE resolution, variable per crater (span = diameter*(1+fov)); "
                       "no resize to a fixed size. Resize at model-input time.",
        "layout": f"{CROPS_FILE} = concatenated C-order float32; {INDEX_FILE} carries "
                  f"byte_offset/h/w plus per-crater RAW (pre-clip) statistics.",
        "git_sha": _git_sha(),
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    m.update(overrides)
    json.dump(m, open(os.path.join(out_dir, MANIFEST_FILE), "w"), indent=2, default=str)


def kaguya_manifest_fields(cropper, tile_dir: str) -> dict:
    """Manifest fields that replace the WAC wording for a Kaguya catalog."""
    from src.data import kaguya_tco as K
    low = (f"L_sigma = sunlit-weighted (DN>1) Gaussian low-pass of log reflectance from the "
           f"overview, sigma = max({cropper.sigma_min_km} km, {cropper.albedo_k} x D) "
           f"(per-crater sigma_km in index.csv)")
    alb = {"none": "reflectance = DN * 0.01",
           "ratio": f"reflectance / exp(L_sigma) (albedo divided out, linear); {low}",
           "logratio": f"log(reflectance) - L_sigma; {low}",
           "subtract": f"reflectance - exp(L_sigma) (WAC-style subtractive high-pass); {low}"}[cropper.albedo]
    return {
        "source": "kaguya_tco",
        "source_mosaic_info": {"tile_dir": tile_dir, "product": "SLN-L-TC-5-ORTHO-MAP-SEAMLESS-V2.0",
                               "res_m": K.M_PER_PX, "dtype": "uint16 big-endian DN",
                               "nodata": K.DUMMY, "low_saturation": K.LOW_SAT},
        "albedo": cropper.albedo, "albedo_sigma_min_km": cropper.sigma_min_km,
        "albedo_k": cropper.albedo_k, "overview_dir": cropper.overview_dir,
        "registration": cropper.offsets_csv or "none",
        "max_frac_lowsat": cropper.max_frac_lowsat,
        "values": f"{alb}. Float32, NOT normalized; apply scaling.json via "
                  f"build_catalog.load_catalog(). DN 1 (deep shadow) kept at the floor; "
                  f"crops touching DN 0 skipped (skipped_invalid.csv).",
        "geometry_note": "each crop resampled ONCE onto a local azimuthal-equidistant grid "
                         "centred on the crater (north up, square 8.4232 m pixels): x "
                         "area-averaged by cos(lat) then cv2.remap INTER_CUBIC. Registration "
                         "(if any) shifts the sampling centre by the per-tile Kaguya-Robbins "
                         "offset (reg_dx_m/reg_dy_m in index.csv).",
        "size_policy": "NATIVE 8.4232 m/px, odd side = ceil(diameter*(1+fov)/8.4232 m); "
                       "variable per crater. Resize at model-input time.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mosaic", "kaguya_tco"], default="mosaic",
                    help="mosaic: one raster via --map_file and crop_crater (the WAC "
                         "catalogs). kaguya_tco: the Kaguya TC tiles via "
                         "kaguya_tco.KaguyaCropper (native 8.42 m/px, local aeqd).")
    ap.add_argument("--map_file", default=None, help="required for --source mosaic")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--filter_desc", required=True,
                    help='e.g. "none", "gaussian_highpass_sigma100"')
    ap.add_argument("--craters_csv", default="data/raw/lunar_crater_database_robbins_2018.csv")
    ap.add_argument("--min_diameter", type=float, default=1.0)
    ap.add_argument("--max_diameter", type=float, default=12.0)
    ap.add_argument("--lat_bounds", type=float, nargs=2, default=[-60, 60])
    ap.add_argument("--craters_to_output", type=int, default=100000)
    ap.add_argument("--clean_offset", type=float, default=1.0)
    ap.add_argument("--exclude_crater_ids", nargs="*", default=None,
                    help="EXTRA held-out-ID csv paths to exclude from the main pool, "
                         "on top of --labels_csv (which is always excluded - see the "
                         "note where this is consumed).")
    ap.add_argument("--only_crater_ids", nargs="*", default=None,
                    help="Restrict the MAIN POOL to these crater IDs (CSV with a "
                         "CRATER_ID column), e.g. to cut a WAC catalog on exactly the "
                         "craters of a Kaguya one. Exclusions still apply.")
    ap.add_argument("--labels_csv", default="configs/liorr_labels.csv",
                    help="Labeled test craters: excluded from the main pool AND "
                         "materialized separately under testset/.")
    ap.add_argument("--seed_per_class", type=int, default=6)
    ap.add_argument("--skip_testset", action="store_true")
    ap.add_argument("--testset_diameter", type=float, nargs=2, default=None,
                    help="labeled testset diameter band (km). Default: no band - the "
                         "label file defines the set (the liorr labels span 1-14.9 km; "
                         "the old 1-5 km band was the old set's range).")
    ap.add_argument("--testset_only", action="store_true",
                    help="(Re)build only testset/ + seedset/ under an EXISTING catalog, "
                         "e.g. after the labeled set changed. The main pool, its "
                         "scaling.json and norm_stats are left untouched - the testset "
                         "keeps using the parent's constants.")
    ap.add_argument("--testset_offset", type=float, default=None,
                    help="labeled testset FOV offset. Default: 0.5 for mosaic, the main "
                         "pool's --clean_offset for kaguya_tco (one framing for probe "
                         "and training crops).")
    # ── kaguya_tco source ────────────────────────────────────────────────────
    ap.add_argument("--tile_dir", default=None)
    ap.add_argument("--albedo", choices=["none", "ratio", "logratio", "subtract"], default="none")
    ap.add_argument("--overview_dir", default="data/raw/kaguya_tco_overview")
    ap.add_argument("--albedo_sigma_min_km", type=float, default=10.0)
    ap.add_argument("--albedo_k", type=float, default=2.0,
                    help="sigma = max(sigma_min, k * D); 0 = fixed sigma_min")
    ap.add_argument("--register", choices=["robbins", "none"], default="none")
    ap.add_argument("--offsets_csv", default="data/raw/kaguya_tco_registration/tile_offsets.csv")
    ap.add_argument("--max_frac_lowsat", type=float, default=1.0,
                    help="skip (and record) Kaguya crops with more than this fraction "
                         "of DN-1 (clipped shadow) pixels; 1.0 = keep all")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cropper, extra_manifest = None, {}
    if args.source == "mosaic":
        if not args.map_file:
            ap.error("--source mosaic requires --map_file")
        ts_diam = args.testset_diameter or [0.0, 1e9]
        ts_off = 0.5 if args.testset_offset is None else args.testset_offset
    else:
        from src.data.kaguya_tco import KaguyaCropper, TILE_DIR
        tile_dir = args.tile_dir or TILE_DIR
        cropper = KaguyaCropper(
            tile_dir=tile_dir, albedo=args.albedo,
            overview_dir=args.overview_dir if args.albedo != "none" else None,
            sigma_min_km=args.albedo_sigma_min_km, albedo_k=args.albedo_k,
            offsets_csv=args.offsets_csv if args.register == "robbins" else None,
            max_frac_lowsat=args.max_frac_lowsat)
        extra_manifest = kaguya_manifest_fields(cropper, tile_dir)
        args.map_file = tile_dir
        ts_diam = args.testset_diameter or [0.0, 1e9]
        ts_off = args.clean_offset if args.testset_offset is None else args.testset_offset

    common = dict(min_diameter=args.min_diameter, max_diameter=args.max_diameter,
                  lat_bounds=tuple(args.lat_bounds), clean_offset=args.clean_offset,
                  filter_desc=args.filter_desc, cropper=cropper, workers=args.workers)
    # UNION, never a replacement. This used to read
    # `args.exclude_crater_ids or [args.labels_csv]`, so passing any
    # extra exclusion file silently dropped the labels_csv one and
    # leaked the 642 labeled test craters into the training pool -
    # a train/test contamination with no symptom until an
    # evaluation came back suspiciously good.
    excluded = [args.labels_csv] + list(args.exclude_crater_ids or [])

    if args.testset_only:
        if not os.path.exists(os.path.join(args.out_dir, INDEX_FILE)):
            ap.error(f"--testset_only needs an existing catalog at {args.out_dir}")
        print(f"=== testset only: main pool in {args.out_dir} left untouched")
    else:
        _build_main_pool(args, common, excluded, extra_manifest)

    _build_testset(args, common, ts_diam, ts_off, extra_manifest)


def _build_main_pool(args, common, excluded, extra_manifest):
    print(f"=== main pool -> {args.out_dir}")
    idx = build(args.map_file, args.craters_csv, args.out_dir,
                craters_to_output=args.craters_to_output,
                exclude_crater_ids=excluded, only_crater_ids=args.only_crater_ids,
                exclude_lon_bounds=None,   # full longitude ON PURPOSE - the seam is under test
                **common)
    scaling = fit_scaling(idx, args.out_dir)
    write_manifest(args.out_dir, map_file=args.map_file, filter=args.filter_desc,
                   role="main_pool", n_craters=int(len(idx)),
                   diameter_km=[args.min_diameter, args.max_diameter],
                   lat_bounds=list(args.lat_bounds), fov_offset=args.clean_offset,
                   longitude_exclusion="NONE - full 360 deg kept deliberately, the "
                                       "mosaic-seam band is part of what is under test",
                   excluded_crater_ids=excluded, only_crater_ids=args.only_crater_ids,
                   crater_selection="preprocess_2.load_and_filter_craters, "
                                    "filtered.sample(random_state=42) - identical crater "
                                    "set across catalogs built with identical arguments",
                   scaling=scaling, **extra_manifest)


def _build_testset(args, common, ts_diam, ts_off, extra_manifest):
    if not args.skip_testset:
        ts_dir = os.path.join(args.out_dir, "testset")
        print(f"\n=== labeled test set -> {ts_dir}")
        ts_idx = build(args.map_file, args.craters_csv, ts_dir,
                       craters_to_output=-1, only_crater_ids=[args.labels_csv],
                       exclude_lon_bounds=None,
                       **{**common, "min_diameter": ts_diam[0], "max_diameter": ts_diam[1],
                          "clean_offset": ts_off})
        write_manifest(ts_dir, map_file=args.map_file, filter=args.filter_desc,
                       role="labeled_testset", n_craters=int(len(ts_idx)),
                       only_crater_ids=[args.labels_csv], fov_offset=ts_off,
                       diameter_km=(list(ts_diam) if ts_diam[1] < 1e9 else "all (label file defines the set)"),
                       lat_bounds=list(args.lat_bounds),
                       scaling_note="uses the MAIN POOL's scaling.json - the test set must "
                                    "not define its own normalization",
                       longitude_exclusion="NONE", **extra_manifest)

        # seed set: a fixed handful per class, saved individually so the same
        # crater can be eyeballed across all six catalogs
        labels = pd.read_csv(args.labels_csv, dtype={"CRATER_ID": str})
        idcol = "CRATER_ID" if "CRATER_ID" in labels.columns else labels.columns[0]
        labcol = "degree" if "degree" in labels.columns else labels.columns[1]
        seed_dir = os.path.join(args.out_dir, "seedset")
        os.makedirs(seed_dir, exist_ok=True)
        present = set(ts_idx["crater_id"])
        picked = []
        for cls in sorted(labels[labcol].dropna().unique()):
            ids = [i for i in labels.loc[labels[labcol] == cls, idcol].astype(str)
                   if i in present]
            picked += [(cls, i) for i in sorted(ids)[:args.seed_per_class]]
        rowmap = {cid: k for k, cid in enumerate(ts_idx["crater_id"])}
        for cls, cid in picked:
            arr = load_catalog(ts_dir, scaling=None, rows=[rowmap[cid]])[0]
            np.save(os.path.join(seed_dir, f"{cls}__{cid}.npy"), arr)
        json.dump({"seed_craters": [{"class": c, "crater_id": i} for c, i in picked],
                   "note": "raw values, native size; the two source mosaics differ by "
                           "0.25% in pixel scale (100.00006 vs 99.74786 m/px) so "
                           "pixel-wise comparison across sources needs a common grid"},
                  open(os.path.join(seed_dir, "seedset.json"), "w"), indent=2)
        print(f"  seed set: {len(picked)} craters -> {seed_dir}")

    print("\ndone.")


if __name__ == "__main__":
    main()
