"""
kaguya_overview.py — one pass over all Kaguya TCO tiles -> a ~253 m/px
overview of the whole 60S-60N mosaic.

Bands (each a .npy, shape (14400, 43200), float32, same grid convention as
kaguya_tco: overview pixel (i, j) = native block rows [30i, 30i+30),
cols [30j, 30j+30)):
  loglum.npy  mean log(reflectance) over the block's valid (DN>0) pixels
  valid.npy   fraction of DN>0 pixels
  lowsat.npy  fraction of DN==1 (low saturation / deep shadow)
  dn0.npy     fraction of DN==0 (no data)

Uses: the albedo low-pass for --albedo logratio (sigma >= 10 km is >= 40
overview px, so the block averaging is invisible to it), and QA - a global
quick-look plus a per-tile table to spot seams, brightness steps and bad
tiles before extracting anything.

Usage:
  PYTHONPATH=$(pwd) python -m src.data.kaguya_overview --out_dir data/raw/kaguya_tco_overview
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.data.kaguya_tco import (DN_SCALE, HIGH_SAT, LOW_SAT, M_PER_PX, MOON_R, N_COLS,
                                 N_ROWS, TILE_DIR, TILE_PX, KaguyaMosaic)

BLOCK = 30                       # divides TILE_PX=10800 -> 360 px per tile
BANDS = ("loglum", "valid", "lowsat", "dn0")


def _tile_blocks(args):
    path, tr, tc = args
    dn = np.fromfile(path, dtype=">u2").reshape(TILE_PX, TILE_PX)
    n = TILE_PX // BLOCK
    v = dn > 0
    logr = np.where(v, np.log(np.maximum(dn, 1).astype(np.float32) * DN_SCALE), 0.0)
    def bsum(a):
        return a.reshape(n, BLOCK, n, BLOCK).sum(axis=(1, 3), dtype=np.float64)
    nv = bsum(v)
    out = {
        "loglum": (bsum(logr) / np.maximum(nv, 1)).astype(np.float32),
        "valid": (nv / BLOCK ** 2).astype(np.float32),
        "lowsat": (bsum(dn == LOW_SAT) / BLOCK ** 2).astype(np.float32),
        "dn0": (bsum(dn == 0) / BLOCK ** 2).astype(np.float32),
    }
    stats = {"tr": tr, "tc": tc, "path": path,
             "frac_dn0": float((dn == 0).mean()),
             "frac_lowsat": float((dn == LOW_SAT).mean()),
             "frac_highsat": float((dn == HIGH_SAT).mean()),
             "mean_loglum": float(logr[v].mean()) if v.any() else np.nan,
             "p50_dn": float(np.median(dn[::16, ::16]))}
    return tr, tc, out, stats


def build(out_dir: str, tile_dir: str = TILE_DIR, workers: int = 32):
    os.makedirs(out_dir, exist_ok=True)
    mosaic = KaguyaMosaic(tile_dir)
    H, W = N_ROWS // BLOCK, N_COLS // BLOCK
    mm = {b: np.lib.format.open_memmap(os.path.join(out_dir, f"{b}.npy"), mode="w+",
                                       dtype=np.float32, shape=(H, W)) for b in BANDS}
    for b in BANDS:                       # missing tiles stay "all no-data"
        mm[b][:] = 1.0 if b == "dn0" else 0.0
    jobs = [(p, tr, tc) for (tr, tc), p in sorted(mosaic.paths.items())]
    n = TILE_PX // BLOCK
    rows = []
    with Pool(workers) as pool:
        for k, (tr, tc, out, stats) in enumerate(pool.imap_unordered(_tile_blocks, jobs, chunksize=2)):
            for b in BANDS:
                mm[b][tr * n:(tr + 1) * n, tc * n:(tc + 1) * n] = out[b]
            rows.append(stats)
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} tiles", flush=True)
    for b in BANDS:
        mm[b].flush()
    tiles = pd.DataFrame(rows).sort_values(["tr", "tc"])
    tiles.to_csv(os.path.join(out_dir, "tile_stats.csv"), index=False)
    json.dump({"block": BLOCK, "shape": [H, W], "m_per_px": M_PER_PX * BLOCK,
               "n_tiles": len(jobs), "tile_dir": tile_dir,
               "grid": "overview px (i,j) = native rows [30i,30i+30), cols [30j,30j+30); "
                       "native row 0 centre = 60N, col 0 centre = 0E"},
              open(os.path.join(out_dir, "overview.json"), "w"), indent=2)
    print(f"  wrote {H}x{W} overview -> {out_dir}; {len(jobs)} tiles")
    quicklook(out_dir)


def quicklook(out_dir: str, step: int = 10):
    """Global PNGs (1440x4320): log-reflectance, and no-data / low-saturation
    fractions, plus the worst tiles printed for eyeballing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = np.load(os.path.join(out_dir, "loglum.npy"), mmap_mode="r")[::step, ::step]
    V = np.load(os.path.join(out_dir, "valid.npy"), mmap_mode="r")[::step, ::step]
    S = np.load(os.path.join(out_dir, "lowsat.npy"), mmap_mode="r")[::step, ::step]
    ext = [0, 360, -60, 60]
    fig, ax = plt.subplots(3, 1, figsize=(18, 14), constrained_layout=True)
    Lm = np.where(V > 0.5, L, np.nan)
    lo, hi = np.nanpercentile(Lm, [1, 99])
    im = ax[0].imshow(Lm, cmap="gray", vmin=lo, vmax=hi, extent=ext)
    ax[0].set_title("Kaguya TCO: mean log reflectance (~2.5 km/px quick-look)")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    im = ax[1].imshow(1 - V, cmap="magma", vmin=0, vmax=0.2, extent=ext)
    ax[1].set_title("no-data fraction (DN 0)")
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    im = ax[2].imshow(S, cmap="magma", vmin=0, vmax=0.2, extent=ext)
    ax[2].set_title("low-saturation fraction (DN 1 = deep shadow)")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    for a in ax:
        a.set_xticks(range(0, 361, 30)); a.set_yticks(range(-60, 61, 30))
        a.set_xlabel("lon E"); a.set_ylabel("lat")
    p = os.path.join(out_dir, "quicklook.png")
    fig.savefig(p, dpi=90)
    print(f"  quick-look -> {p}")
    t = pd.read_csv(os.path.join(out_dir, "tile_stats.csv"))
    print("  worst tiles by DN0:\n", t.nlargest(8, "frac_dn0")[["path", "frac_dn0", "frac_lowsat"]].to_string(index=False))
    print("  worst tiles by DN1:\n", t.nlargest(8, "frac_lowsat")[["path", "frac_dn0", "frac_lowsat"]].to_string(index=False))


def sunlit_overview(out_dir: str, mask_lowsat: float = 0.2, mask_sigma_km: float = 10.0,
                    min_valid: float = 0.5):
    """An overview NOT distorted by clipped shadow: the low-sun block
    (~47-60S 143-210E, ~90% DN 1) and the patch near 41S 85E otherwise show as
    dark mottled rectangles, because loglum averages log(0.01) into every
    block that holds shadow.

      value  = SUNLIT log reflectance, exactly recovered from the stored bands:
               (loglum*valid - lowsat*log(0.01)) / (valid - lowsat)
      masked = NaN where the 10-km-smoothed DN-1 fraction exceeds mask_lowsat
               (the regional twin of the catalog's per-crop 0.2 cutoff) or
               where under min_valid of the block has data.

    Writes kaguya_sunlit_overview_180.tif - float32, nodata NaN, in the WAC
    basemap's own convention (Equirectangular metres, R=1737400, lon -180..180)
    so it drops straight into QGIS or geographic_map --basemap - and
    quicklook_sunlit.png with the masked areas drawn in a separate colour."""
    import math
    import cv2
    import rasterio
    from rasterio.transform import from_origin
    from scipy.ndimage import gaussian_filter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = np.load(os.path.join(out_dir, "loglum.npy"), mmap_mode="r")
    V = np.load(os.path.join(out_dir, "valid.npy"), mmap_mode="r")
    S = np.load(os.path.join(out_dir, "lowsat.npy"), mmap_mode="r")
    H, W = L.shape
    opx = M_PER_PX * BLOCK
    log1 = math.log(LOW_SAT * DN_SCALE)

    # regional shadow fraction on a 4x coarser grid (~1 km), then back up
    f = 4
    Sc = np.asarray(S, np.float32).reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    sig = mask_sigma_km * 1000.0 / (opx * f)
    Sr = gaussian_filter(Sc, sig, mode=("nearest", "wrap"))
    mask = cv2.resize((Sr > mask_lowsat).astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

    out = np.full((H, W), np.nan, np.float32)
    for r0 in range(0, H, 1200):                      # row chunks: bounded memory
        sl = slice(r0, min(r0 + 1200, H))
        lg, v, s = (np.asarray(a[sl], np.float32) for a in (L, V, S))
        w = v - s
        sun = np.where(w > 1e-6, (lg * v - s * log1) / np.maximum(w, 1e-6), np.nan)
        sun[(v < min_valid) | mask[sl]] = np.nan
        out[sl] = sun
    out = np.roll(out, W // 2, axis=1)                 # lon 0..360 -> -180..180
    mask_r = np.roll(mask, W // 2, axis=1)
    nodata_r = np.roll(np.asarray(V) < min_valid, W // 2, axis=1)

    deg = MOON_R * math.pi / 180.0
    tif = os.path.join(out_dir, "kaguya_sunlit_overview_180.tif")
    crs = "+proj=eqc +lat_ts=0 +lon_0=0 +R=1737400 +units=m +no_defs"
    # overview px (i, j) covers native rows [30i, 30i+30): its top edge is the
    # native pixel-centre grid shifted by half a NATIVE pixel
    transform = from_origin(-180.0 * deg - M_PER_PX / 2, 60.0 * deg + M_PER_PX / 2, opx, opx)
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=float("nan"), compress="deflate",
                       predictor=3, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES") as dst:
        dst.write(out, 1)
    print(f"  sunlit overview -> {tif}  (masked low-sun {100 * mask.mean():.2f}% of area)")

    step = 10
    q = out[::step, ::step]
    lo, hi = np.nanpercentile(q, [1, 99])
    fig, ax = plt.subplots(figsize=(18, 6.6), constrained_layout=True)
    from matplotlib.colors import ListedColormap
    ax.imshow(np.where(mask_r[::step, ::step], 1.0, np.nan), cmap=ListedColormap(["#ff8c00"]),
              extent=[-180, 180, -60, 60], interpolation="nearest")
    ax.imshow(np.where(nodata_r[::step, ::step] & ~mask_r[::step, ::step], 1.0, np.nan),
              cmap=ListedColormap(["#00bcd4"]), extent=[-180, 180, -60, 60], interpolation="nearest")
    im = ax.imshow(q, cmap="gray", vmin=lo, vmax=hi, extent=[-180, 180, -60, 60])
    ax.set_title("Kaguya TCO - SUNLIT log reflectance (clipped shadow excluded); "
                 f"orange = very-low-sun areas masked (10-km DN-1 fraction > {mask_lowsat}), "
                 "cyan = no data", fontsize=11)
    ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-60, 61, 30))
    ax.set_xlabel("lon E"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log reflectance")
    png = os.path.join(out_dir, "quicklook_sunlit.png")
    fig.savefig(png, dpi=90)
    print(f"  quick-look -> {png}")


def ratio_overview(out_dir: str, sigma_km: float = 10.0, band_deg: float = 6.0,
                   mask_lowsat: float = 0.2, min_valid: float = 0.5, mode: str = "ratio"):
    """What the RATIO catalog does, at map scale: sunlit reflectance divided by
    its own valid-weighted Gaussian low-pass (sigma on the ground - the
    catalog's value for craters up to 5 km; larger craters get 2xD). Drawn on
    a FIXED linear scale around 1, never per-image stretched, so dark/bright
    in the figure is dark/bright in the data.

    Writes kaguya_ratio_overview_180.tif (float32 ratio, NaN masked, same grid
    and CRS as kaguya_sunlit_overview_180.tif) and quicklook_ratio.png: global
    raw vs ratio + regional zooms (mare/highland contact, Copernicus rays,
    dark mare, farside highlands). Features smaller than ~sigma keep their own
    albedo (rays, fresh ejecta) - by design; only the regional albedo divides out.
    """
    import math
    import cv2
    import rasterio
    from rasterio.transform import from_origin
    from scipy.ndimage import gaussian_filter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = np.load(os.path.join(out_dir, "loglum.npy"), mmap_mode="r")
    V = np.load(os.path.join(out_dir, "valid.npy"), mmap_mode="r")
    S = np.load(os.path.join(out_dir, "lowsat.npy"), mmap_mode="r")
    H, W = L.shape
    opx = M_PER_PX * BLOCK
    log1 = math.log(LOW_SAT * DN_SCALE)
    sy = sigma_km * 1000.0 / opx
    rows_per_deg = H / 120.0
    band = int(band_deg * rows_per_deg)
    pad = int(math.ceil(3 * sy)) + 2
    # same regional low-sun mask as sunlit_overview
    f = 4
    Sc = np.asarray(S, np.float32).reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    Sr = gaussian_filter(Sc, 10.0 * 1000.0 / (opx * f), mode=("nearest", "wrap"))
    lowsun = cv2.resize((Sr > mask_lowsat).astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    sun = np.full((H, W), np.nan, np.float32)
    ratio = np.full((H, W), np.nan, np.float32)
    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)
        a0, a1 = max(r0 - pad, 0), min(r1 + pad, H)
        lg, v, s = (np.asarray(x[a0:a1], np.float32) for x in (L, V, S))
        w = np.clip(v - s, 0, None)
        ls = np.where(w > 1e-6, (lg * v - s * log1) / np.maximum(w, 1e-6), 0.0)
        lat_c = 60.0 - ((r0 + r1) / 2.0) / rows_per_deg
        sx = sy / max(math.cos(math.radians(lat_c)), 0.05)
        # A low-pass this smooth loses nothing on a 4x coarser grid (~1 km);
        # cost falls ~50x, which makes sigma 30-60 km practical.
        cf = 4 if sigma_km >= 20 else 1
        if cf > 1:
            hh, ww = (ls.shape[0] // cf) * cf, ls.shape[1]
            q = lambda z: z[:hh].reshape(hh // cf, cf, ww // cf, cf).mean(axis=(1, 3))
            lw_c, w_c = q(ls * w), q(w)
            num = gaussian_filter(lw_c, (sy / cf, sx / cf), mode=("constant", "wrap"), truncate=3.0)
            den = gaussian_filter(w_c, (sy / cf, sx / cf), mode=("constant", "wrap"), truncate=3.0)
            low = cv2.resize((num / np.maximum(den, 1e-6)).astype(np.float32), (ww, ls.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        else:
            num = gaussian_filter(ls * w, (sy, sx), mode=("constant", "wrap"), truncate=3.0)
            den = gaussian_filter(w, (sy, sx), mode=("constant", "wrap"), truncate=3.0)
            low = num / np.maximum(den, 1e-6)
        sl = slice(r0 - a0, r0 - a0 + (r1 - r0))
        bad = (v[sl] < min_valid) | lowsun[r0:r1]
        sb = np.where(bad, np.nan, ls[sl])
        sun[r0:r1] = sb
        ratio[r0:r1] = (np.exp(sb) - np.exp(low[sl])) if mode == "subtract" else np.exp(sb - low[sl])
        print(f"  rows {r0}-{r1} (lat {lat_c:+.0f}, sigma_x {sx:.0f} px)", flush=True)
    sun = np.roll(sun, W // 2, axis=1); ratio = np.roll(ratio, W // 2, axis=1)
    deg = MOON_R * math.pi / 180.0
    tag = ("" if sigma_km == 10.0 else f"_sigma{sigma_km:g}km") + ("_subtract" if mode == "subtract" else "")
    tif = os.path.join(out_dir, f"kaguya_ratio_overview_180{tag}.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs="+proj=eqc +lat_ts=0 +lon_0=0 +R=1737400 +units=m +no_defs",
                       transform=from_origin(-180.0 * deg - M_PER_PX / 2, 60.0 * deg + M_PER_PX / 2, opx, opx),
                       nodata=float("nan"), compress="deflate", predictor=3, tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="YES") as dst:
        dst.write(ratio, 1)
    print(f"  ratio overview -> {tif}")

    raw = np.exp(sun)                       # sunlit reflectance
    rlo, rhi = np.nanpercentile(raw[::10, ::10], [1, 99])
    qlo, qhi = (0.6, 1.4) if mode != "subtract" else (-3.0, 3.0)   # FIXED, symmetric around 1 / 0
    def win(lat0, lat1, lon0, lon1):
        i0, i1 = int((60 - lat1) * rows_per_deg), int((60 - lat0) * rows_per_deg)
        j0, j1 = int((lon0 + 180) * W / 360), int((lon1 + 180) * W / 360)
        return (slice(i0, i1), slice(j0, j1)), [lon0, lon1, lat0, lat1]
    regions = [("mare/highland contact (Serenitatis rim)", 16, 36, 4, 34),
               ("Copernicus + ray system", 0, 20, -32, -8),
               ("dark mare (Oceanus Procellarum)", 0, 20, -62, -38),
               ("farside highlands", 0, 20, 150, 174)]
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(2 + len(regions), 2, height_ratios=[1.3, 1.3] + [1] * len(regions))
    for row, (img, nm, lo, hi) in enumerate([(raw, "RAW sunlit reflectance", rlo, rhi),
                                             (ratio, (f"SUBTRACT = reflectance - its {sigma_km:g} km low-pass (WAC-style)"
                                                      if mode == "subtract" else
                                                      f"RATIO = reflectance / its {sigma_km:g} km low-pass"), qlo, qhi)]):
        ax = fig.add_subplot(gs[row, :])
        im = ax.imshow(img[::8, ::8], cmap="gray", vmin=lo, vmax=hi, extent=[-180, 180, -60, 60])
        ax.set_title(f"{nm} - fixed linear scale [{lo:.2f}, {hi:.2f}]", fontsize=12)
        ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-60, 61, 30))
        for _, a, b, c, d in regions:
            ax.add_patch(plt.Rectangle((c, a), d - c, b - a, fill=False, ec="#ff8c00", lw=1.2))
        fig.colorbar(im, ax=ax, shrink=0.8)
    for k, (nm, a, b, c, d) in enumerate(regions):
        sl, ext = win(a, b, c, d)
        for col, (img, lo, hi, lab) in enumerate([(raw, rlo, rhi, "raw"), (ratio, qlo, qhi, mode)]):
            ax = fig.add_subplot(gs[2 + k, col])
            ax.imshow(img[sl], cmap="gray", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
            ax.set_title(f"{nm} - {lab} (fixed [{lo:.2f}, {hi:.2f}])", fontsize=10)
    fig.suptitle("Kaguya ratio data at map scale (~253 m/px) vs raw - the same fixed scales the "
                 "model-input normalization implies, no per-image stretch; orange boxes = zooms", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = os.path.join(out_dir, f"quicklook_ratio{tag}.png")
    fig.savefig(png, dpi=70)
    print(f"  quick-look -> {png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/raw/kaguya_tco_overview")
    ap.add_argument("--tile_dir", default=TILE_DIR)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--quicklook_only", action="store_true")
    ap.add_argument("--ratio", action="store_true",
                    help="from an existing overview: the ratio (albedo-removed) map at "
                         "sigma=10 km - GeoTIFF + quick-look (see ratio_overview)")
    ap.add_argument("--sigma_km", type=float, default=10.0, help="--ratio low-pass sigma (ground km)")
    ap.add_argument("--subtract", action="store_true", help="with --ratio: subtract the low-pass (WAC-style) instead of dividing")
    ap.add_argument("--sunlit", action="store_true",
                    help="from an existing overview: write the shadow-free sunlit "
                         "overview GeoTIFF + quick-look (see sunlit_overview)")
    a = ap.parse_args()
    if a.ratio:
        ratio_overview(a.out_dir, sigma_km=a.sigma_km, mode="subtract" if a.subtract else "ratio")
    elif a.sunlit:
        sunlit_overview(a.out_dir)
    elif a.quicklook_only:
        quicklook(a.out_dir)
    else:
        build(a.out_dir, a.tile_dir, a.workers)
