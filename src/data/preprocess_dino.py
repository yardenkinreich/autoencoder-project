"""
export_dino_pngs.py
═══════════════════
Export crater crops as PNGs for DINOv2 self-supervised pretraining.

Key differences vs. the MAE preprocessing (preprocess_2.py):

  • WIDER field of view.  offset = 1.0 (≈2× crater diameter) instead of the
    MAE clean branch's 0.5.  The crater sits in the centre with surrounding
    terrain / ejecta visible — richer context for DINO's global↔local
    prediction task and for reading degradation state.

  • NO augmentation.  DINOv2's DataAugmentationDINO does all multi-crop /
    rotation / flip / jitter at train time.  We export clean, un-augmented
    images only.

  • FIXED scaling, not per-sample min–max.  Per-image min–max would wash out
    the relative-contrast cue that distinguishes fresh (high local contrast)
    from degraded (washed-out) craters, and would also stack badly with
    DINO's own Normalize().  We scale every crop with the SAME mapping so
    cross-crater contrast is preserved.  DINO's transform then standardises
    with the dataset mean/std (compute once, drop into the config).

  • 3-channel PNG at 224×224 (multiple of patch-14) so the stock DINOv2
    ViT/14 ingests it unchanged.

Reuses the crater filtering + left-band (180–270°E) seam exclusion from
preprocess_2.py so nothing domain-specific is duplicated.

At 100 m/px a 3 km crater ≈ 30 px and a 10 km crater ≈ 100 px; at offset 1.0
the FOV is ~2× diameter, so even the smallest crater keeps ~30 px of detail
inside a ~60 px field before the resize to 224 — plenty of signal.
"""

import os
import argparse
import numpy as np
import pandas as pd
import pyproj
import rasterio
import cv2
from PIL import Image
from src.helper_functions import *      # crop_crater, etc. (same as preprocess.py)

# Reuse the exact filtering / CRS / seam-exclusion logic from preprocess.py
from preprocess import load_and_filter_craters, get_craters_crs


# ── config ──────────────────────────────────────────────────────────────────
OUTPUT_SIZE = 224          # multiple of 14 → stock DINOv2 ViT/14 ingests as-is
DINO_OFFSET = 1.0          # ≈2× crater diameter FOV (crater fills centre half)


# ── fixed (non per-sample) scaling ───────────────────────────────────────────
def scale_fixed(img: np.ndarray, mode: str, lo: float, hi: float) -> np.ndarray:
    """
    Map a raw crop to uint8 [0,255] with a FIXED mapping shared across all
    crops, so relative brightness/contrast between craters is preserved.

    mode = 'global'  : clip to the dataset-wide [lo, hi] then scale.
    mode = 'raw01'   : assume input already in [0,1]; just clip & scale.
    """
    if mode == "raw01":
        x = np.clip(img.astype(np.float32), 0.0, 1.0)
    elif mode == "global":
        x = (np.clip(img.astype(np.float32), lo, hi) - lo) / max(hi - lo, 1e-8)
    else:
        raise ValueError(f"unknown scaling mode {mode!r}")
    return (x * 255.0 + 0.5).astype(np.uint8)


def first_pass_global_range(map_ref, transformer, craters, percentile, sample_n):
    """
    Estimate dataset-wide [lo, hi] from a random sample of crops so the fixed
    scaling uses robust percentile bounds rather than absolute min/max
    (which a single bright/dark crop would blow out).
    """
    idx = (craters.sample(n=min(sample_n, len(craters)), random_state=0)
                  .index)
    vals = []
    for i in idx:
        c = craters.loc[i]
        raw = crop_crater(map_ref, c["LAT_CIRC_IMG"], c["LON_CIRC_IMG"],
                          c["DIAM_CIRC_IMG"], DINO_OFFSET, transformer)
        vals.append(raw.ravel())
    allv = np.concatenate(vals)
    lo = float(np.percentile(allv, percentile))
    hi = float(np.percentile(allv, 100 - percentile))
    print(f"  Global scaling range (p{percentile}): [{lo:.4f}, {hi:.4f}]")
    return lo, hi


# ── main export ───────────────────────────────────────────────────────────--
def export(filtered, map_file, out_dir, scaling, percentile, sample_n):
    os.makedirs(out_dir, exist_ok=True)
    craters_crs = get_craters_crs()
    N = len(filtered)

    # running stats for the dataset mean/std you'll paste into the DINO config
    sum_, sumsq_, count_ = 0.0, 0.0, 0

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )

        lo = hi = 0.0
        if scaling == "global":
            lo, hi = first_pass_global_range(map_ref, transformer,
                                             filtered, percentile, sample_n)

        print(f"\n{'='*60}")
        print(f"  Exporting        : {N} crater PNGs")
        print(f"  Output size      : {OUTPUT_SIZE}×{OUTPUT_SIZE} (3-channel)")
        print(f"  Offset (FOV)     : {DINO_OFFSET}  (~{2*DINO_OFFSET:.1f}× diameter)")
        print(f"  Scaling          : {scaling}")
        print(f"  Augmentation     : NONE (DINO does it at train time)")
        print(f"{'='*60}\n")

        for k, (_, c) in enumerate(filtered.iterrows()):
            if k % 10_000 == 0:
                print(f"  {k}/{N}")

            raw = crop_crater(map_ref, c["LAT_CIRC_IMG"], c["LON_CIRC_IMG"],
                              c["DIAM_CIRC_IMG"], DINO_OFFSET, transformer)
            img = cv2.resize(raw, (OUTPUT_SIZE, OUTPUT_SIZE),
                             interpolation=cv2.INTER_CUBIC)
            img8 = scale_fixed(img, scaling, lo, hi)        # (H,W) uint8
            rgb  = np.stack([img8] * 3, axis=-1)            # grayscale → 3ch

            # accumulate stats on the [0,1] float version
            f = img8.astype(np.float64) / 255.0
            sum_   += f.sum()
            sumsq_ += (f * f).sum()
            count_ += f.size

            cid = c["CRATER_ID"]
            Image.fromarray(rgb, "RGB").save(
                os.path.join(out_dir, f"{cid}.png"), optimize=False)

    mean = sum_ / count_
    std  = (sumsq_ / count_ - mean * mean) ** 0.5
    print(f"\n✅  Exported {N} PNGs → {out_dir}")
    print(f"\n   Dataset stats (single channel, replicate ×3 for RGB):")
    print(f"     mean = {mean:.5f}")
    print(f"     std  = {std:.5f}")
    print(f"\n   In your DINO config / augmentations, set Normalize to:")
    print(f"     mean = ({mean:.5f}, {mean:.5f}, {mean:.5f})")
    print(f"     std  = ({std:.5f}, {std:.5f}, {std:.5f})")
    np.savez(os.path.join(out_dir, "_dataset_stats.npz"),
             mean=mean, std=std, n=count_)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export wider-FOV crater PNGs for DINOv2 pretraining")
    p.add_argument("--map_file",          required=True)
    p.add_argument("--craters_csv",       required=True)
    p.add_argument("--out_dir",           required=True,
                   help="Flat folder of {CRATER_ID}.png for the DINO loader")
    p.add_argument("--min_diameter",      type=float, default=3.0)
    p.add_argument("--max_diameter",      type=float, default=10.0)
    p.add_argument("--latitude_bounds",   type=float, nargs=2, default=[-60, 60])
    p.add_argument("--exclude_lon_bounds", type=float, nargs=2, default=[180, 270],
                   help="Seam band to drop (deg E). Default 180-270 (the bad tile).")
    p.add_argument("--craters_to_output", type=int, default=-1)
    p.add_argument("--scaling", choices=["global", "raw01"], default="global",
                   help="'global' = robust dataset-wide percentile range (default); "
                        "'raw01' = assume crops already in [0,1].")
    p.add_argument("--percentile", type=float, default=1.0,
                   help="Lower percentile for 'global' scaling (uses p and 100-p).")
    p.add_argument("--range_sample_n", type=int, default=2000,
                   help="How many crops to sample when estimating global range.")
    args = p.parse_args()

    filtered = load_and_filter_craters(
        args.craters_csv, args.min_diameter, args.max_diameter,
        args.latitude_bounds, args.craters_to_output,
        exclude_lon_bounds=args.exclude_lon_bounds,
    )
    print(f"Filtered {len(filtered)} craters")

    export(filtered, args.map_file, args.out_dir,
           args.scaling, args.percentile, args.range_sample_n)