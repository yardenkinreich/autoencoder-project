"""
kaguya_smoke_figure.py — side-by-side sheets of Kaguya TC crops vs the WAC
crops of the same craters, for eyeballing before any full catalog build.

Rows: labeled craters (all four classes, spread in diameter) plus deliberate
edge cases (high latitude, a 14 km crater, the 0 deg meridian, a tile corner,
a maria/highland contact, a heavily shadowed crater).
Columns:
  1  Kaguya raw reflectance        (registered, native 8.42 m/px)
  2  Kaguya ratio = reflectance / overview low-pass (albedo removed, linear)
  3  WAC morph sigma100            (the current training input, crop_crater, same FOV)
  4  WAC EMP raw                   (crop_crater, same FOV)
  5  registration: centre 1.4xD of the UNregistered Kaguya crop; orange =
     Robbins rim, cyan dashed = where the tile offset puts the crater
All panels are shown with nearest-neighbour interpolation so the true pixel
size of each source is visible.

Usage:
  PYTHONPATH=$(pwd) python -m src.data.kaguya_smoke_figure --out_dir data/catalogs/kaguya_tco_smoke_figs
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from src.data.kaguya_tco import (LAT_TOP, TILE_DEG, KaguyaCropper, KaguyaOverview,
                                 OutOfCoverage)

WAC_MORPH_S100 = "data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif"
WAC_EMP = "data/raw/wac_uniform/WAC_EMP_643_100m_180.tif"
CLASS_ORDER = ["v-fr", "r-fr", "r-dr", "v-dr"]


def pick_craters(labels_csv, robbins_csv, overview_dir, per_class=3, seed=0):
    rng = np.random.default_rng(seed)
    lab = pd.read_csv(labels_csv, dtype={"CRATER_ID": str})
    lab = lab[lab.lat.abs() < 58]
    picks = []
    for cls in CLASS_ORDER:
        g = lab[lab.degree == cls].sort_values("diam_km")
        if g.empty:
            continue
        for q in np.linspace(0.15, 0.95, per_class):          # spread in diameter
            r = g.iloc[min(int(q * len(g)), len(g) - 1)]
            picks.append(dict(id=r.CRATER_ID, lat=r.lat, lon=r.lon, D=r.diam_km, tag=f"label {cls}"))

    rob = pd.read_csv(robbins_csv, usecols=["CRATER_ID", "LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"])
    rob = rob[rob.LAT_CIRC_IMG.abs() < 58].rename(columns={
        "CRATER_ID": "id", "LAT_CIRC_IMG": "lat", "LON_CIRC_IMG": "lon", "DIAM_CIRC_IMG": "D"})
    rob = rob[~rob.id.astype(str).isin(set(lab.CRATER_ID))]
    def one(df, tag):
        if len(df):
            r = df.iloc[int(rng.integers(len(df)))]
            picks.append(dict(id=r.id, lat=r.lat, lon=r.lon, D=r.D, tag=tag))
    one(rob[(rob.lat.between(53, 57.5)) & rob.D.between(2, 5)], "edge: lat~55")
    one(rob[rob.D.between(13.5, 15) & (rob.lat.abs() < 40)], "edge: D~14 km")
    one(rob[((rob.lon < 0.15) | (rob.lon > 359.85)) & rob.D.between(1.5, 5)], "edge: 0 meridian")
    near = lambda v: np.abs(((v + 1.5) % TILE_DEG) - 1.5)
    one(rob[(near(rob.lat) < 0.01) & (near(rob.lon) < 0.01) & rob.D.between(1.5, 5)], "edge: tile corner")

    # maria/highland contact and deep shadow, judged on the overview
    ov = KaguyaOverview(overview_dir)
    cand = rob[rob.D.between(2, 6)].sample(20000, random_state=seed)
    b = ov.block
    i = ((LAT_TOP - cand.lat) * 3600 / b).astype(int).to_numpy()
    j = ((cand.lon % 360) * 3600 / b).astype(int).to_numpy()
    L, S = ov.loglum, np.load(os.path.join(overview_dir, "lowsat.npy"), mmap_mode="r")
    k = 6                                                          # ~1.5 km steps either side
    grad = np.array([abs(float(L[a, min(c + k, L.shape[1] - 1)]) - float(L[a, max(c - k, 0)]))
                     + abs(float(L[min(a + k, L.shape[0] - 1), c]) - float(L[max(a - k, 0), c]))
                     for a, c in zip(i, j)])
    shade = np.array([float(S[a, c]) for a, c in zip(i, j)])
    # a real mare/highland contact: strong albedo step with NO clipped shadow
    # nearby (the low-sun block's edge is also a strong step, and is not one)
    clear = np.array([float(np.max(S[max(a - k, 0):a + k + 1, max(c - k, 0):c + k + 1]))
                      for a, c in zip(i, j)]) < 0.005
    r = cand.iloc[int(np.argmax(np.where(clear & (cand.lat.abs() < 50).to_numpy(), grad, -1)))]
    picks.append(dict(id=r.id, lat=r.lat, lon=r.lon, D=r.D, tag="edge: maria/highland contact"))
    r = cand.iloc[int(np.argmax(shade))]
    picks.append(dict(id=r.id, lat=r.lat, lon=r.lon, D=r.D,
                      tag="edge: low-sun block (EXCLUDED, lowsat>0.2)"))
    return picks


def wac_crop(path, lat, lon, D, offset):
    import pyproj
    import rasterio
    from src.helper_functions import crop_crater
    from src.data.preprocess_2 import get_craters_crs
    with rasterio.open(path) as src:
        tr = pyproj.Transformer.from_crs(get_craters_crs(), src.crs.to_string(), always_xy=True)
        a = crop_crater(src, lat, lon, D, offset, tr).astype(np.float32)
    a[~np.isfinite(a) | (np.abs(a) > 1e30)] = np.nan
    return a


def _show(ax, a, title, cmap="gray"):
    v = a[np.isfinite(a)]
    lo, hi = (np.percentile(v, [1, 99]) if v.size else (0, 1))
    ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def make_sheets(picks, out_dir, offset, overview_dir, offsets_csv, rows_per_sheet=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    raw_r = KaguyaCropper(albedo="none", offsets_csv=offsets_csv)
    rat_r = KaguyaCropper(albedo="ratio", overview_dir=overview_dir, offsets_csv=offsets_csv)
    raw_u = KaguyaCropper(albedo="none")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in range(0, len(picks), rows_per_sheet):
        chunk = picks[s:s + rows_per_sheet]
        fig, axes = plt.subplots(len(chunk), 5, figsize=(17, 3.5 * len(chunk)), squeeze=False)
        for ax, p in zip(axes, chunk):
            lat, lon, D = float(p["lat"]), float(p["lon"]), float(p["D"])
            try:
                kr, ex = raw_r.crop(lat, lon, D, offset)
                kl, exl = rat_r.crop(lat, lon, D, offset)
                ku, _ = raw_u.crop(lat, lon, D, offset)
            except OutOfCoverage as e:
                ax[0].set_title(f"{p['tag']}: out of coverage ({e})", fontsize=8)
                continue
            px = ex["m_per_px"]
            head = f"{p['tag']}  {p['id']}\nD={D:.2f} km  ({lat:.2f}, {lon:.2f})"
            _show(ax[0], kr, f"{head}\nKaguya raw {kr.shape[1]}x{kr.shape[0]} @ {px:.1f} m/px"
                             f"  lowsat={ex['frac_lowsat']:.3f}")
            _show(ax[1], kl, f"Kaguya ratio, albedo removed (sigma={exl['sigma_km']:.0f} km)")
            for k, (path, name) in enumerate(((WAC_MORPH_S100, "WAC morph sigma100 (train input)"),
                                              (WAC_EMP, "WAC EMP raw"))):
                try:
                    w = wac_crop(path, lat, lon, D, offset)
                    _show(ax[2 + k], w, f"{name}\n{w.shape[1]}x{w.shape[0]} @ ~100 m/px")
                except Exception as e:
                    ax[2 + k].set_title(f"{name}: {type(e).__name__}", fontsize=8)
            # registration panel: central 1.4 D of the unregistered crop
            n = ku.shape[0]
            h = int(round(0.7 * D * 1000 / px))
            c = n // 2
            z = ku[max(c - h, 0):c + h + 1, max(c - h, 0):c + h + 1]
            _show(ax[4], z, f"registration (unregistered crop, 1.4xD)\n"
                            f"tile offset dx={ex['reg_dx_m']:+.0f} m dy={ex['reg_dy_m']:+.0f} m")
            zc = z.shape[0] // 2
            r_px = D * 1000 / 2 / px
            ax[4].add_patch(Circle((zc, zc), r_px, fill=False, ec="orange", lw=1.4))
            ax[4].add_patch(Circle((zc + ex["reg_dx_m"] / px, zc - ex["reg_dy_m"] / px), r_px,
                                   fill=False, ec="cyan", lw=1.2, ls="--"))
        fig.suptitle("Kaguya TCO vs WAC, same craters, same FOV "
                     f"(side = {1 + offset:.1f} x D); nearest-neighbour display", fontsize=11)
        fig.tight_layout()
        pth = os.path.join(out_dir, f"smoke_sheet_{s // rows_per_sheet + 1}.png")
        fig.savefig(pth, dpi=80)
        plt.close(fig)
        paths.append(pth)
        print(f"  sheet -> {pth}", flush=True)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/catalogs/kaguya_tco_smoke_figs")
    ap.add_argument("--labels_csv", default="configs/liorr_labels.csv")
    ap.add_argument("--craters_csv", default="data/raw/lunar_crater_database_robbins_2018.csv")
    ap.add_argument("--overview_dir", default="data/raw/kaguya_tco_overview")
    ap.add_argument("--offsets_csv", default="data/raw/kaguya_tco_registration/tile_offsets.csv")
    ap.add_argument("--clean_offset", type=float, default=1.0)
    ap.add_argument("--per_class", type=int, default=3)
    ap.add_argument("--raw_catalog", default="data/catalogs/kaguya_tco_raw_smoke")
    ap.add_argument("--ratio_catalog", default="data/catalogs/kaguya_tco_ratio_smoke")
    a = ap.parse_args()
    picks = pick_craters(a.labels_csv, a.craters_csv, a.overview_dir, a.per_class)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(picks).to_csv(os.path.join(a.out_dir, "picked_craters.csv"), index=False)
    make_sheets(picks, a.out_dir, a.clean_offset, a.overview_dir, a.offsets_csv)
    if os.path.isdir(a.raw_catalog) and os.path.isdir(a.ratio_catalog):
        albedo_sheet(a.raw_catalog, a.ratio_catalog, os.path.join(a.out_dir, "albedo_sheet.png"))


if __name__ == "__main__":
    main()


def albedo_sheet(raw_dir, ratio_dir, out_png, n_each=4):
    """What albedo removal is FOR: under the catalog's own global scaling
    (scaling.json, the normalization a model sees), mare craters are darker
    than highland ones in raw reflectance; after the ratio they are not.
    Top: per-crater median value, raw vs ratio. Bottom: darkest- and
    brightest-background craters, each row on ONE shared scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.build_catalog import load_catalog, load_index, load_scaling
    ir, iq = load_index(raw_dir), load_index(ratio_dir)
    assert (ir.crater_id.to_numpy() == iq.crater_id.to_numpy()).all()
    order = np.argsort(ir.p50.to_numpy())
    dark, bright = order[:n_each], order[-n_each:]
    rows = np.r_[dark, bright]
    sr, sq = load_scaling(raw_dir), load_scaling(ratio_dir)
    raw = load_catalog(raw_dir, sr, rows=rows, out_size=192)
    rat = load_catalog(ratio_dir, sq, rows=rows, out_size=192)
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2 * n_each)
    for k, (idx, name, sc) in enumerate(((ir, "raw reflectance", sr), (iq, "ratio (albedo removed)", sq))):
        ax = fig.add_subplot(gs[0, k * n_each:(k + 1) * n_each])
        v = (idx.p50 - sc["lo"]) / (sc["hi"] - sc["lo"])
        ax.hist(np.clip(v, -0.2, 1.2), bins=60, color="0.3")
        ax.set_title(f"{name}: per-crater median after the catalog's global scaling\n"
                     f"(spread = albedo differences the model must absorb); std={v.std():.3f}", fontsize=9)
        ax.set_xlim(-0.2, 1.2)
    for r, (imgs, name) in enumerate(((raw, "raw"), (rat, "ratio"))):
        for c in range(2 * n_each):
            ax = fig.add_subplot(gs[1 + r, c])
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            i = rows[c]
            ax.set_title(f"{name} {'darkest' if c < n_each else 'brightest'} bg\n"
                         f"({ir.lat[i]:.0f}, {ir.lon[i]:.0f}) D={ir.diam_km[i]:.1f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Albedo removal under the shared global scaling (rows 2-3: one fixed [0,1] scale for all panels)")
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    print(f"  albedo sheet -> {out_png}")
