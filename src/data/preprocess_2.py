import os
import math
import json
import numpy as np
import pandas as pd
import pyproj
import rasterio
import cv2
import argparse
from src.helper_functions import *

# ── output resolution ─────────────────────────────────────────────────────────
# Recommended: 128 with patch_size=8 (256 patches) for Facebook MAE trained
# from scratch.  224 works if you are fine-tuning pretrained ViT weights
# (positional embeddings can be interpolated, but 224 is cleanest for that).
# Any value divisible by your MAE patch size (8 or 16) is valid.
OUTPUT_SIZE = 128   # ← 224 | 128 | 64

# ── augmented-branch geometry ─────────────────────────────────────────────────
# We need an intermediate tile large enough that after an arbitrary rotation
# a centre-crop of OUTPUT_SIZE never samples outside the valid image region.
# Requirement: AUG_EXTRACT_SIZE ≥ OUTPUT_SIZE × √2
# We also round up to the next even number for cleanliness.
_raw = math.ceil(OUTPUT_SIZE * math.sqrt(2))
AUG_EXTRACT_SIZE = _raw + (_raw % 2)          # e.g. 182 for OUTPUT_SIZE=128

# Offset passed to crop_crater so that after
#   resize(AUG_EXTRACT_SIZE) → rotate → centre_crop(OUTPUT_SIZE)
# the surviving FOV equals exactly a clean 0.5-offset crop.
#   aug_offset × (OUTPUT_SIZE / AUG_EXTRACT_SIZE) = 0.5
#   → aug_offset = 0.5 × (AUG_EXTRACT_SIZE / OUTPUT_SIZE)
AUG_OFFSET = 0.5 * (AUG_EXTRACT_SIZE / OUTPUT_SIZE)   # ≈ 0.711 for 128


# ── normalization mode (set in __main__ from CLI) ─────────────────────────────
# "raw"            : NO scaling — store raw float crops. Use this once to dump a
#                    .dat you can fit global percentiles on (fit_global_scaling.py).
# "global"         : clip to FROZEN [lo, hi] from scaling_json, scale to [0,1].
#                    This is what you train on (and what eval matches).
# "per_sample"     : OLD behaviour (per-crater min-max). Kept for comparison only.
NORM_MODE   = "global"
GLOBAL_LO   = None      # filled from scaling_json when NORM_MODE == "global"
GLOBAL_HI   = None


# ── helpers ───────────────────────────────────────────────────────────────────

def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Square centre-crop of a 2-D (H, W) image."""
    h, w = img.shape[:2]
    top  = (h - size) // 2
    left = (w - size) // 2
    return img[top : top + size, left : left + size]


def random_rotate(img: np.ndarray) -> np.ndarray:
    """
    Rotate a 2-D image by a uniformly random angle in [0°, 360°).

    BORDER_REFLECT_101 mirrors pixels at the edge so there are no black
    corners, though corners are cropped away anyway by the subsequent
    centre-crop.  Full 360° rotation is standard for rotationally symmetric
    objects with no preferred orientation; craters qualify.
    """
    angle = np.random.uniform(0.0, 360.0)
    h, w  = img.shape[:2]
    # NOTE: cv2.getRotationMatrix2D takes DEGREES. `angle` is already in degrees,
    # so it is passed through directly. (The previous `angle*180/np.pi` was a bug
    # that mapped a 0–360° draw to 0–20626°; harmless for uniform rotation but
    # not what the docstring claimed.)
    M     = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale=1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def normalize_per_sample(img: np.ndarray) -> np.ndarray:
    """Per-sample min–max normalisation → [0, 1]. (OLD behaviour.)"""
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return (img.astype(np.float32) - lo) / (hi - lo)
    return np.zeros_like(img, dtype=np.float32)


def normalize_global(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Clip to the FIXED [lo, hi] (frozen percentiles from the reference .dat) and
    scale to [0, 1]. Constants are identical across ALL craters, so absolute
    contrast — the fresh-vs-eroded cue — is preserved. This is the scaling the
    encoder is trained on and that eval (Julie's PNGs) must match.
    """
    x = np.clip(img.astype(np.float32), lo, hi)
    return (x - lo) / (hi - lo)


def normalize(img: np.ndarray) -> np.ndarray:
    """Dispatch on the module-level NORM_MODE."""
    if NORM_MODE == "raw":
        return img.astype(np.float32)                       # no scaling
    if NORM_MODE == "global":
        if GLOBAL_LO is None:
            raise RuntimeError("NORM_MODE=global but scaling not loaded; "
                               "pass --scaling_json")
        return normalize_global(img, GLOBAL_LO, GLOBAL_HI)
    if NORM_MODE == "per_sample":
        return normalize_per_sample(img)
    raise ValueError(f"unknown NORM_MODE={NORM_MODE!r}")


def to_tensor(img_2d: np.ndarray, num_channels: int) -> np.ndarray:
    """Normalised 2-D (H, W) image → float32 (C, H, W) tensor."""
    norm = normalize(img_2d)
    return (norm[np.newaxis]           if num_channels == 1
            else np.stack([norm] * 3))  # (C, H, W)


# ── crater loading ────────────────────────────────────────────────────────────

def _to_360(lon):
    """Normalise any longitude (Series or scalar) to [0, 360)."""
    return lon % 360.0


def load_and_filter_craters(craters_csv, min_diameter, max_diameter,
                             latitude_bounds, craters_to_output,
                             exclude_lon_bounds=None):
    """
    Filter the crater catalog by diameter and latitude, and optionally drop a
    longitude band corrupted by mosaic stitching (the WAC morphology mosaic has
    incidence-angle discontinuities along constant-longitude seams between its
    three compositing windows, which deepen/distort shadows on one side and
    throw off crater categorisation).

    exclude_lon_bounds : (lo, hi) in degrees East. Craters with longitude inside
        this interval are removed. The interval may wrap past 360 (e.g.
        (300, 60) excludes 300°→360° and 0°→60°). Bounds given in the -180..180
        convention are accepted and normalised internally, so e.g. (-180, -120)
        works as well as (180, 240).
    """
    craters = pd.read_csv(craters_csv)
    filtered = craters[
        (craters['DIAM_CIRC_IMG'] >= min_diameter) &
        (craters['DIAM_CIRC_IMG'] <= max_diameter) &
        (craters['LAT_CIRC_IMG']  >= latitude_bounds[0]) &
        (craters['LAT_CIRC_IMG']  <= latitude_bounds[1])
    ]

    if exclude_lon_bounds is not None:
        lo, hi = (_to_360(b) for b in exclude_lon_bounds)
        lon = _to_360(filtered['LON_CIRC_IMG'])
        if lo <= hi:
            in_band = (lon >= lo) & (lon <= hi)          # simple interval
        else:
            in_band = (lon >= lo) | (lon <= hi)          # wraps across 360→0
        n_before = len(filtered)
        filtered = filtered[~in_band]
        print(f"Excluded longitude band [{lo:.1f}°, {hi:.1f}°] E "
              f"→ dropped {n_before - len(filtered)} of {n_before} craters")

    if craters_to_output > 0:
        filtered = filtered.sample(n=craters_to_output, random_state=42)
    return filtered


def get_craters_crs():
    craters_wkt = """
        GEOGCS["GCS_Moon",
        DATUM["D_Moon_2000",
        SPHEROID["Moon_2000_IAU_IAG",1737400,0, LENGTHUNIT["metre",1]]],
        PRIMEM["Reference_Meridian",0],
        UNIT["metre",1]],
        PROJECTION["Equirectangular"],
        PARAMETER["standard_parallel_1",0],
        PARAMETER["central_meridian",0],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["metre",1, AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["ESRI","103881"]]
    """
    return pyproj.CRS.from_wkt(craters_wkt)


# ── main processing ───────────────────────────────────────────────────────────

def _flush_and_print_stats(memmap, name, path, N, num_channels):
    memmap.flush()
    sample = memmap[: min(5_000, N)]
    print(f"\n── {name} statistics (first {len(sample)} samples) ──")
    print(f"  Range : [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"  Mean  : {sample.mean():.4f}   Std: {sample.std():.4f}")
    if num_channels == 3:
        print(f"  Per-channel mean : {sample.mean(axis=(0, 2, 3))}")
        print(f"  Per-channel std  : {sample.std(axis=(0, 2, 3))}")
    stats_path = path.replace('.dat', '_stats.npz')
    np.savez(stats_path,
             mean=sample.mean(axis=(0, 2, 3)),
             std=sample.std(axis=(0, 2, 3)))
    print(f"  Stats saved → {stats_path}")


def process_and_save_crater_crops(
    filtered_craters,
    map_file,
    output_dir_clean,
    output_dir_aug=None,
    np_output_path_clean=None,
    np_output_path_aug=None,
    save_raw_crops=True,
    save_np_array=True,
    num_channels=1,
    clean_offset=0.5,
):
    """
    Two-branch preprocessing pipeline
    ══════════════════════════════════

    Clean branch  (offset = clean_offset, default 0.5)
    ─────────────────────────────────────────────────
      crop_crater(offset=clean_offset)
        → resize(OUTPUT_SIZE × OUTPUT_SIZE)
        → normalise (mode-dependent: raw / global / per_sample)

    Augmented branch  (offset ≈ 0.71) — OPTIONAL, skipped entirely if
    output_dir_aug/np_output_path_aug are both None. Skip this for
    pipelines (e.g. DINOv2) that do their own rotation/multi-crop
    augmentation at train time, so it isn't computed for nothing.
    ──────────────────────────────────
      crop_crater(offset=AUG_OFFSET)
        → resize(AUG_EXTRACT_SIZE × AUG_EXTRACT_SIZE)
        → random rotation ∈ [0°, 360°)       ← full rotation, no artifacts
        → centre_crop(OUTPUT_SIZE × OUTPUT_SIZE)
        → normalise (mode-dependent)

    The augmented branch's centre-crop preserves the same physical FOV as
    the clean branch's 0.5-offset crop (only when clean_offset=0.5 — the
    two offsets aren't coupled if you override clean_offset).  Any further
    augmentation (flips, brightness jitter, etc.) should be applied
    on-the-fly in the DataLoader.

    Both branches produce tensors of shape (num_channels, OUTPUT_SIZE, OUTPUT_SIZE).
    """
    do_aug = output_dir_aug is not None or np_output_path_aug is not None

    os.makedirs(output_dir_clean, exist_ok=True)
    if do_aug and output_dir_aug is not None:
        os.makedirs(output_dir_aug, exist_ok=True)

    craters_crs = get_craters_crs()
    N     = len(filtered_craters)
    shape = (N, num_channels, OUTPUT_SIZE, OUTPUT_SIZE)

    clean_mm = aug_mm = None
    if save_np_array:
        if np_output_path_clean:
            clean_mm = np.memmap(np_output_path_clean, dtype=np.float32,
                                 mode="w+", shape=shape)
        if np_output_path_aug:
            aug_mm   = np.memmap(np_output_path_aug,   dtype=np.float32,
                                 mode="w+", shape=shape)

    print(f"\n{'='*60}")
    print(f"  Craters        : {N}")
    print(f"  Output size    : {OUTPUT_SIZE}×{OUTPUT_SIZE}")
    print(f"  Clean offset   : {clean_offset:.3f}")
    if do_aug:
        print(f"  Aug extract    : {AUG_EXTRACT_SIZE}×{AUG_EXTRACT_SIZE}  (offset ≈ {AUG_OFFSET:.3f})")
    else:
        print(f"  Aug branch     : skipped")
    print(f"  Channels       : {num_channels}")
    print(f"  Norm mode      : {NORM_MODE}"
          + (f"  (lo={GLOBAL_LO:.4f}, hi={GLOBAL_HI:.4f})" if NORM_MODE == 'global' else ""))
    print(f"{'='*60}\n")

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )

        for i, (_, crater) in enumerate(filtered_craters.iterrows()):
            if i % 10_000 == 0:
                print(f"  {i}/{N}")

            lat  = crater["LAT_CIRC_IMG"]
            lon  = crater["LON_CIRC_IMG"]
            diam = crater["DIAM_CIRC_IMG"]
            cid  = crater["CRATER_ID"]

            # ── clean branch ──────────────────────────────────────────────
            raw_clean    = crop_crater(map_ref, lat, lon, diam, clean_offset, transformer)
            img_clean    = cv2.resize(raw_clean, (OUTPUT_SIZE, OUTPUT_SIZE),
                                      interpolation=cv2.INTER_CUBIC)
            tensor_clean = to_tensor(img_clean, num_channels)

            # ── store ─────────────────────────────────────────────────────
            if clean_mm is not None:
                clean_mm[i] = tensor_clean
            if save_raw_crops:
                np.save(os.path.join(output_dir_clean, f"{cid}.npy"), tensor_clean)

            # ── augmented branch (optional) ─────────────────────────────
            if do_aug:
                raw_aug    = crop_crater(map_ref, lat, lon, diam, AUG_OFFSET, transformer)
                img_aug    = cv2.resize(raw_aug, (AUG_EXTRACT_SIZE, AUG_EXTRACT_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                img_aug    = random_rotate(img_aug)           # random 0–360°
                img_aug    = center_crop(img_aug, OUTPUT_SIZE) # drop rotation border
                tensor_aug = to_tensor(img_aug, num_channels)

                if aug_mm is not None:
                    aug_mm[i] = tensor_aug
                if save_raw_crops and output_dir_aug is not None:
                    np.save(os.path.join(output_dir_aug, f"{cid}.npy"), tensor_aug)

    # ── stats & flush ─────────────────────────────────────────────────────
    for name, mm, path in [
        ("clean", clean_mm, np_output_path_clean),
        ("aug",   aug_mm,   np_output_path_aug),
    ]:
        if mm is not None:
            _flush_and_print_stats(mm, name, path, N, num_channels)

    n_branches = 2 if do_aug else 1
    print(f"\n✅  Done. {N} craters × {n_branches} branch{'es' if n_branches > 1 else ''}.")
    print(f"   Tensor shape : ({num_channels}, {OUTPUT_SIZE}, {OUTPUT_SIZE})")


# ── metadata ──────────────────────────────────────────────────────────────────

def save_crater_metadata(filtered_craters, map_file, output_path):
    craters_crs = get_craters_crs()
    buffer      = []
    batch_size  = 10_000

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )
        for i, crater in filtered_craters.iterrows():
            lon = crater['LON_CIRC_IMG']
            lat = crater['LAT_CIRC_IMG']
            if lon > 180:
                lon -= 360
            x, y = transformer.transform(lon, lat)
            buffer.append({
                'id':   crater['CRATER_ID'],
                'lat':  lat,
                'lon':  lon,
                'x':    x,
                'y':    y,
                'diam': crater['DIAM_CIRC_IMG'] * 1_000,   # km → m
            })
            if (i + 1) % batch_size == 0:
                print(f"Metadata: processed {i + 1}")
                df   = pd.DataFrame(buffer)
                mode = 'a' if os.path.exists(output_path) else 'w'
                df.to_csv(output_path, mode=mode,
                          header=not os.path.exists(output_path), index=False)
                buffer.clear()

        if buffer:
            df   = pd.DataFrame(buffer)
            mode = 'a' if os.path.exists(output_path) else 'w'
            df.to_csv(output_path, mode=mode,
                      header=not os.path.exists(output_path), index=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Crater preprocessing: clean + augmented branches"
    )
    parser.add_argument('--map_file',             required=True)
    parser.add_argument('--craters_csv',          required=True)
    parser.add_argument('--output_dir_clean',     required=True,
                        help="Directory for clean (non-augmented) .npy crops")
    parser.add_argument('--output_dir_aug',       default=None,
                        help="Directory for augmented (rotated) .npy crops. "
                             "Omit (along with --np_output_path_aug) to skip "
                             "the aug branch entirely, e.g. for DINOv2 which "
                             "does its own rotation/multi-crop augmentation.")
    parser.add_argument('--np_output_path_clean', required=True,
                        help="Memory-mapped .dat file for clean branch")
    parser.add_argument('--np_output_path_aug',   default=None,
                        help="Memory-mapped .dat file for augmented branch (optional, see --output_dir_aug)")
    parser.add_argument('--info_output_path',     required=True)
    parser.add_argument('--min_diameter',         type=float, default=3.0)
    parser.add_argument('--max_diameter',         type=float, default=10.0)
    parser.add_argument('--clean_offset',         type=float, default=0.5,
                        help="Crop offset for the clean branch (fraction of radius "
                             "added as margin). 0.5 = MAE/CAE default (tight crop). "
                             "~1.0 = ~2x diameter FOV, e.g. for DINOv2 so local/global "
                             "multi-crops have genuinely different content to sample.")
    parser.add_argument('--latitude_bounds',      type=float, nargs=2, default=[-60, 60])
    parser.add_argument('--exclude_lon_bounds',   type=float, nargs=2, default=None,
                        help="Drop craters whose longitude (deg E) falls in this "
                             "[lo, hi] band, e.g. the seam strip. Accepts -180..180 "
                             "or 0..360; may wrap (lo>hi). Omit to keep all longitudes.")
    parser.add_argument('--craters_to_output',    type=int,   default=-1)
    parser.add_argument('--save_raw_crops',       action='store_true')
    parser.add_argument('--save_np_array',        action='store_true')
    parser.add_argument('--autoencoder_model',    type=str,
                        choices=['cae', 'mae'],   default='mae')

    # ── normalization control ────────────────────────────────────────────────
    parser.add_argument('--norm_mode', choices=['raw', 'global', 'per_sample'],
                        default='global',
                        help="raw: no scaling (dump a .dat to fit percentiles on). "
                             "global: clip to frozen lo/hi from --scaling_json, "
                             "scale [0,1] (train on this). per_sample: old behaviour.")
    parser.add_argument('--scaling_json', default=None,
                        help="Path to frozen percentiles JSON "
                             "(required when --norm_mode global).")
    args = parser.parse_args()

    # set module-level normalization state from CLI
    NORM_MODE = args.norm_mode
    if NORM_MODE == 'global':
        if not args.scaling_json or not os.path.exists(args.scaling_json):
            parser.error("--norm_mode global requires an existing --scaling_json "
                         "(run fit_global_scaling.py first).")
        _sc = json.load(open(args.scaling_json))
        GLOBAL_LO, GLOBAL_HI = float(_sc["lo"]), float(_sc["hi"])
        print(f"Loaded global scaling: lo={GLOBAL_LO:.4f} hi={GLOBAL_HI:.4f} "
              f"(from {args.scaling_json})")

    os.makedirs(args.output_dir_clean, exist_ok=True)
    if args.output_dir_aug is not None:
        os.makedirs(args.output_dir_aug, exist_ok=True)

    filtered = load_and_filter_craters(
        args.craters_csv, args.min_diameter, args.max_diameter,
        args.latitude_bounds, args.craters_to_output,
        exclude_lon_bounds=args.exclude_lon_bounds,
    )
    print(f"Filtered {len(filtered)} craters")

    num_channels = 1   # 1 for CAE/MAE grayscale; 3 if using RGB pretrained ViT

    process_and_save_crater_crops(
        filtered,
        args.map_file,
        args.output_dir_clean,
        args.output_dir_aug,
        np_output_path_clean=args.np_output_path_clean,
        np_output_path_aug=args.np_output_path_aug,
        save_raw_crops=args.save_raw_crops,
        save_np_array=args.save_np_array,
        num_channels=num_channels,
        clean_offset=args.clean_offset,
    )
    save_crater_metadata(filtered, args.map_file, args.info_output_path)
    print("All done.")