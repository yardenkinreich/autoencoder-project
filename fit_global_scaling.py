#!/usr/bin/env python
"""
fit_global_scaling.py
─────────────────────
Compute FIXED 1st/99th intensity percentiles over the clean .dat (mosaic crops)
and freeze them to a JSON. This file becomes the single source of truth used by
BOTH training preprocessing and Julie's eval load, so the two can never drift.

    python fit_global_scaling.py \
        --clean-dat data/processed/craters_clean.dat \
        --n 152103 --channels 1 --size 128 \
        --out configs/global_scaling.json

The .dat is a float32 memmap of shape (N, C, H, W) written by your preprocessing
BEFORE the normalize() step would have run — i.e. it must hold RAW intensities,
not already-min-maxed ones, or the percentiles are meaningless. If your current
.dat is post-normalize, re-dump a raw one first (see note at bottom).
"""
from __future__ import annotations
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dat", required=True)
    ap.add_argument("--n", type=int, required=True, help="number of craters in the .dat")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lo-pct", type=float, default=1.0)
    ap.add_argument("--hi-pct", type=float, default=99.0)
    ap.add_argument("--sample", type=int, default=20000,
                    help="subsample this many craters for the percentile estimate")
    ap.add_argument("--out", default="configs/global_scaling.json")
    args = ap.parse_args()

    shape = (args.n, args.channels, args.size, args.size)
    mm = np.memmap(args.clean_dat, dtype=np.float32, mode="r", shape=shape)

    # subsample craters (not pixels) so the estimate is cheap but representative
    rng = np.random.default_rng(0)
    idx = rng.choice(args.n, size=min(args.sample, args.n), replace=False)
    pixels = np.asarray(mm[np.sort(idx)]).reshape(-1)

    lo = float(np.percentile(pixels, args.lo_pct))
    hi = float(np.percentile(pixels, args.hi_pct))
    if hi <= lo:
        raise ValueError(f"degenerate percentiles lo={lo} hi={hi}; is the .dat raw?")

    out = {
        "lo": lo, "hi": hi,
        "lo_pct": args.lo_pct, "hi_pct": args.hi_pct,
        "source": args.clean_dat, "n_sampled": len(idx),
        "raw_min": float(pixels.min()), "raw_max": float(pixels.max()),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"global scaling frozen -> {args.out}")
    print(f"  lo({args.lo_pct}%)={lo:.4f}  hi({args.hi_pct}%)={hi:.4f}  "
          f"[raw range {out['raw_min']:.4f}, {out['raw_max']:.4f}]")


if __name__ == "__main__":
    main()

# ── note ──────────────────────────────────────────────────────────────────────
# If your existing clean.dat was written with the OLD per-sample normalize()
# baked in, its values are already squashed to [0,1] per crater and these
# percentiles will be ~[0,1] and useless. In that case re-run preprocessing once
# with the new global normalize DISABLED (raw float crops) to dump a raw .dat,
# fit on it, then re-enable global normalize for the real training dump.
