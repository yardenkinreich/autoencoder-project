"""
highpass_mosaic.py — parameterized version of process_full_mosaic.py's
σ-100 high-pass step, so the same filter can be applied to a DIFFERENT
source mosaic (e.g. the photometrically normalized WAC_EMP_643) without
editing hardcoded paths.

Why this exists: the seam investigation established that the tile-boundary
contrast steps in the crater data are inherent to the WAC_GLOBAL morphology
mosaic (they are the same magnitude before and after high-pass filtering -
the filter is not the cause), and that WAC_EMP_643 is the only
photometrically normalized WAC product available at this project's native
~100 m/pixel. Testing EMP therefore means re-running this exact filter on
it, which process_full_mosaic.py could not do without modification.

NODATA IS THE DANGEROUS PART. WAC_GLOBAL declares no nodata value, so
process_full_mosaic.py never had to think about it. WAC_EMP_643 declares
nodata = -3.4e38. A Gaussian filter is a weighted average over a ~sigma-
sized neighbourhood, so even a handful of -3.4e38 pixels would smear
catastrophic values across a ~100-pixel radius and silently corrupt large
regions of the output - the kind of failure that produces plausible-looking
garbage rather than an error. So nodata is masked, filled before filtering,
and restored as NaN afterwards.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sigma", type=float, default=100.0)
    ap.add_argument("--cpu", action="store_true",
                    help="Use the scipy CPU filter instead of the cupy GPU one "
                         "(full_mosaic_processing exposes both).")
    ap.add_argument("--divisive_sigma", type=float, default=None,
                    help="If set, divide the residual by its own local contrast "
                         "level, sqrt(blur(resid^2)), with this sigma IN PIXELS. "
                         "The high-pass is SUBTRACTIVE and removes a regional albedo "
                         "offset; the seam is a multiplicative GAIN difference that "
                         "subtraction cannot touch. 500 px (50 km) sits in the gap "
                         "between crater scale (10-120 px) and tile scale (~27000 px). "
                         "See full_mosaic_processing.divisive_normalize_gpu.")
    ap.add_argument("--divisive_floor_pct", type=float, default=1.0,
                    help="Clamp the divisor at this percentile of the gain field so "
                         "genuinely smooth terrain is not blown up by dividing by "
                         "~zero.")
    ap.add_argument("--legacy_clip", action="store_true",
                    help="Reproduce the historical products by re-enabling the "
                         "filter's global 0.1/99.9 percentile clip. Destroys "
                         "fresh-crater pixels in proportion to freshness "
                         "(6.0%% of v-fr vs 0.0%% of v-dr on morphology, 11.9%% "
                         "overall on EMP) - see full_mosaic_processing."
                         "_LEGACY_CLIP_PCT. For reproduction only.")
    ap.add_argument("--max_saturated_pct", type=float, default=0.5,
                    help="Refuse to save if more than this %% of output pixels sit "
                         "on exactly the min or max value. Catches a clip or a "
                         "dtype ceiling before it becomes a 17GB product.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        raise FileExistsError(f"{args.output} already exists - refusing to overwrite a "
                              f"multi-GB derived product.")

    with rasterio.open(args.input) as src:
        print(f"input: {args.input}")
        print(f"  shape {src.shape}  dtype {src.dtypes[0]}  nodata {src.nodata}  res {src.res}")
        image = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    image = image.astype(np.float32, copy=False)

    # --- invalid-pixel guard (see module docstring) -------------------------
    # Deliberately NOT `image == src.nodata`. That comparison is unreliable:
    # rasterio hands back nodata as a float64, and a float32 sentinel whose
    # bit pattern differs even slightly compares False - so sentinel pixels
    # survive into the "valid" set. A single surviving -3.4e38 is enough to
    # make a float32 .mean() over ~4e9 elements collapse to -inf, which then
    # becomes the fill value and gets smeared across a ~6*sigma radius by the
    # convolution. That exact failure produced a 16GB corrupt mosaic on the
    # first run of this script (max=inf, mean=inf, 99.29% valid) - it did not
    # raise, it just silently poisoned 0.7% of the output.
    #
    # So: flag anything non-finite OR of implausible magnitude, whatever its
    # exact value. Real WAC reflectance / high-pass residuals are order 1e-2;
    # 1e30 is many orders of magnitude past anything physical, so this cannot
    # catch real data.
    SENTINEL_MAG = 1e30
    invalid = ~np.isfinite(image) | (np.abs(image) >= SENTINEL_MAG)
    n_bad = int(invalid.sum())
    print(f"  invalid pixels (non-finite or |v|>={SENTINEL_MAG:g}): "
          f"{n_bad:,} ({100.0 * n_bad / image.size:.6f}%)")
    mask = None
    if n_bad:
        mask = invalid
        # Robust fill from a SAMPLE of valid pixels: a median over a sample
        # can neither overflow float32 (unlike a full-array mean) nor be
        # dragged by a sentinel that slipped through. A finite constant near
        # the data's own centre also avoids creating an artificial edge at the
        # mask boundary, which filling with 0 would do.
        rng = np.random.default_rng(0)
        flat = image.reshape(-1)
        idx = rng.choice(flat.size, size=min(5_000_000, flat.size), replace=False)
        samp = flat[idx]
        samp = samp[np.isfinite(samp) & (np.abs(samp) < SENTINEL_MAG)]
        fill = float(np.median(samp))
        image[mask] = fill
        print(f"  filled with median of {samp.size:,} sampled valid pixels: {fill:.6g}")
        if not np.isfinite(fill):
            raise ValueError("fill value is not finite - refusing to filter")
    # Belt and braces: the convolution must never see a non-finite input.
    if not np.isfinite(image).all():
        raise ValueError("image still contains non-finite values after masking")

    from src.data.full_mosaic_processing import (
        local_highpass_filter, local_highpass_filter_gpu, _LEGACY_CLIP_PCT,
        divisive_normalize_gpu,
    )
    filt = local_highpass_filter if args.cpu else local_highpass_filter_gpu
    # clip_pct=None unless explicitly asked for. The filter's historical
    # default clipped the residual to the mosaic's GLOBAL 0.1/99.9 percentiles,
    # which pinned 6.0% of the freshest craters' pixels and 0.0% of the most
    # degraded ones onto the bounds - a label-correlated destruction of exactly
    # the bright-rim/deep-shadow cue this project measures. Outlier handling
    # belongs in normalization (src/data/norm_stats.py), where it is a recorded
    # and changeable choice, not baked into the stored product.
    clip_pct = _LEGACY_CLIP_PCT if args.legacy_clip else None
    print(f"applying high-pass filter (sigma={args.sigma}, "
          f"{'CPU/scipy' if args.cpu else 'GPU/cupy'}, clip={clip_pct})...")
    out = filt(image, sigma=args.sigma, clip_pct=clip_pct)
    out = np.asarray(out, dtype=np.float32)

    if args.divisive_sigma:
        # Applied AFTER the high-pass, on its residual - the two are
        # complementary, not alternatives: subtractive kills the albedo
        # offset, divisive kills the contrast gain.
        if args.cpu:
            raise SystemExit("--divisive_sigma is GPU-only (drop --cpu)")
        print(f"applying divisive local-contrast normalization "
              f"(sigma={args.divisive_sigma})...")
        out = np.asarray(divisive_normalize_gpu(
            out, sigma=args.divisive_sigma,
            floor_pct=args.divisive_floor_pct), dtype=np.float32)
        if mask is not None:
            out[mask] = 0.0

    if mask is not None:
        # Zero the holes - NOT NaN. An earlier version wrote NaN here (with
        # nodata=NaN) on the reasoning that a hole should be visible as a
        # hole. That silently destroyed the entire downstream dataset: only
        # ~1 crater crop in 2000 actually contained one of these pixels, but
        # fit_global_scaling.py uses np.percentile, so that single crop drove
        # the global lo/hi to NaN, and every one of the 642 test-set crops
        # then normalized to 100% NaN. Nothing raised anywhere along the way.
        #
        # 0.0 is the right finite filler for a high-pass residual (whose mean
        # is ~0 by construction), it is inert for the percentile fit, and it
        # affects 0.0002% of pixels. No nodata is declared, so downstream
        # readers treat the product as fully valid - which, at 9,130 filled
        # pixels out of ~4e9, is the honest description.
        out[mask] = 0.0

    valid = np.isfinite(out)
    print(f"output range: min={np.nanmin(out):.6g} max={np.nanmax(out):.6g} "
          f"mean={np.nanmean(out):.6g}  valid {100.0 * valid.mean():.4f}%")
    # Fail loudly rather than writing another silently-corrupt 16GB product:
    # anything beyond the masked holes being invalid means the filter itself
    # propagated something it should not have.
    expected_invalid = 0 if mask is None else int(mask.sum())
    actual_invalid = int((~valid).sum())
    if actual_invalid > expected_invalid:
        raise ValueError(
            f"filter produced {actual_invalid:,} invalid output pixels but only "
            f"{expected_invalid:,} were masked going in - the convolution spread "
            f"a bad value; refusing to save.")
    if not np.isfinite(np.nanmean(out)):
        raise ValueError("output mean is not finite - refusing to save.")

    # --- saturation gate ----------------------------------------------------
    # A high-pass residual is a difference of two smooth fields; its value
    # distribution should be continuous, with essentially no pixels sharing the
    # exact extreme value. A spike AT the min or max means something clipped:
    # the filter's legacy percentile clip, a dtype ceiling, or a fill value.
    # This gate exists because that failure shipped once already - the EMP
    # sigma-100 product had 11.87% of every crater crop pinned to exactly its
    # maximum, and nothing raised. It was found only when a normalization
    # analysis noticed p99 == p99.99 == max.
    fin = out[np.isfinite(out)]
    mn, mx = float(fin.min()), float(fin.max())
    n_at_edge = int((fin == mn).sum() + (fin == mx).sum())
    sat_pct = 100.0 * n_at_edge / fin.size
    print(f"  pixels at exactly min or max: {n_at_edge:,} ({sat_pct:.4f}%)")
    if not args.legacy_clip and sat_pct > args.max_saturated_pct:
        raise ValueError(
            f"{sat_pct:.4f}% of output pixels sit on exactly the min ({mn:.6g}) or "
            f"max ({mx:.6g}), above the {args.max_saturated_pct}% limit. A "
            f"high-pass residual should not have a value spike at its extremes - "
            f"something clipped it. Refusing to save.")

    profile.update(dtype=rasterio.float32, count=1, compress="lzw",
                   driver="GTiff", bigtiff="YES")
    print(f"saving to {args.output} ...")
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(out, 1)
    print("done.")


if __name__ == "__main__":
    main()
