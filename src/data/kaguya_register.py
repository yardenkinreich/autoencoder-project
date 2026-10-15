"""
kaguya_register.py — measure where Kaguya TCO shows Robbins craters, per tile,
and turn that into the offset field KaguyaCropper applies (--register robbins).

WHAT IS MEASURED
----------------
For each Robbins crater (1-5 km by default, up to --per_tile per 3x3 deg tile)
a 3xD equirectangular window at the Robbins position is resized to 128 px and
its rim-edge point-symmetry centre is found (kaguya_tco.symmetry_center). The
displacement of that centre from the window centre, in metres (+E, +N), is
"Kaguya minus Robbins" for that crater.

Per crater this is noisy (~0.2 D scatter, measured on 1,700 craters) and must
never be used to re-centre an individual crater. Per tile the MEDIAN over
hundreds of craters is precise (bootstrap s.e. ~10-30 m) and unbiased: the
same estimator on WAC EMP gives (0, 0) pooled, as it should, since Robbins
measured positions on WAC. The read-only pilot found Kaguya-Robbins tile
offsets mostly < 60 m, up to ~130 m (northern tiles lean north).

FROM TILE MEDIANS TO A FIELD
----------------------------
  1. tile median (dx, dy) + bootstrap s.e.; tiles with < MIN_N craters are
     treated as missing
  2. outliers vs the 3x3 neighbourhood median (> 3 x max(MAD, s.e.) and
     > 30 m) and missing tiles are replaced by that neighbourhood median
  3. what is APPLIED is chosen by --mode:
       gate   : the cleaned value where |value| > 2 s.e., else 0
       shrink : empirical-Bayes shrinkage of each tile toward 0 by its own
                s.e.: v * tau^2 / (tau^2 + se^2), tau^2 = between-tile
                variance net of noise
       none   : zeros
     The half-split validation below reports all three, so the choice is
     made on held-out craters, not assumed.

VALIDATION
----------
Craters in every tile are split at random into halves A and B. The field is
fitted on A only and evaluated on B: the per-tile median of
(B measurement - field_A at the crater) is the residual offset. Its typical
size is compared with no correction, and with the noise floor (B's own
per-tile s.e.), and written to validation.csv.

Usage:
  PYTHONPATH=$(pwd) python -m src.data.kaguya_register --out_dir data/raw/kaguya_tco_registration
"""
from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.data.kaguya_tco import (LAT_TOP, LOW_SAT, M_PER_PX, MOON_R, N_TILE_COLS,
                                 N_TILE_ROWS, PX_PER_DEG, TILE_DEG, TILE_DIR,
                                 KaguyaMosaic, OffsetField, OutOfCoverage,
                                 symmetry_center)

FOV = 3.0           # window side in crater diameters
MIN_N = 10          # fewer craters than this -> tile treated as missing
N_BOOT = 300

_MOSAIC = None


def _init(tile_dir):
    global _MOSAIC
    _MOSAIC = KaguyaMosaic(tile_dir)


def _measure_tile(job):
    """All craters of one tile -> list of per-crater records."""
    tr, tc, craters = job
    out = []
    for cid, lat, lon, D, half in craters:
        cl = math.cos(math.radians(lat))
        hy = FOV * D * 1000.0 / 2.0 / M_PER_PX
        hx = hy / cl
        row = (LAT_TOP - lat) * PX_PER_DEG
        col = (lon % 360.0) * PX_PER_DEG
        r0, r1, c0, c1 = int(row - hy), int(row + hy), int(col - hx), int(col + hx)
        try:
            dn = _MOSAIC.read_window(r0, r1, c0, c1)
        except OutOfCoverage:
            continue
        if (dn == 0).any() or (dn <= LOW_SAT).mean() > 0.02:
            continue
        fx, fy = symmetry_center(np.log(np.maximum(dn, 2).astype(np.float32)))
        # the integer window's own centre is off the crater by a sub-pixel amount
        fx += ((c0 + c1) / 2.0 - col) / (c1 - c0)
        fy += ((r0 + r1) / 2.0 - row) / (r1 - r0)
        span = FOV * D * 1000.0
        out.append((cid, tr, tc, lat, lon, D, half, fx * span, -fy * span))
    return out


def measure(craters_csv, tile_dir, min_d, max_d, per_tile, workers, seed=0):
    rob = pd.read_csv(craters_csv, usecols=["CRATER_ID", "LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"])
    rob = rob[(rob.DIAM_CIRC_IMG >= min_d) & (rob.DIAM_CIRC_IMG <= max_d)
              & (rob.LAT_CIRC_IMG.abs() < LAT_TOP)].copy()
    rob["tr"] = ((LAT_TOP - rob.LAT_CIRC_IMG) // TILE_DEG).astype(int)
    rob["tc"] = ((rob.LON_CIRC_IMG % 360.0) // TILE_DEG).astype(int)
    rng = np.random.default_rng(seed)
    jobs = []
    for (tr, tc), g in rob.groupby(["tr", "tc"]):
        g = g.sample(min(per_tile, len(g)), random_state=seed)
        half = rng.permutation(np.arange(len(g)) % 2)          # balanced A/B split
        jobs.append((tr, tc, list(zip(g.CRATER_ID, g.LAT_CIRC_IMG, g.LON_CIRC_IMG,
                                      g.DIAM_CIRC_IMG, np.where(half == 0, "A", "B")))))
    print(f"  {len(rob):,} craters in {min_d}-{max_d} km; measuring up to {per_tile}/tile "
          f"in {len(jobs)} tiles", flush=True)
    recs = []
    with Pool(workers, initializer=_init, initargs=(tile_dir,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_measure_tile, jobs, chunksize=1)):
            recs.extend(r)
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} tiles, {len(recs):,} craters", flush=True)
    return pd.DataFrame(recs, columns=["crater_id", "tr", "tc", "lat", "lon", "diam_km",
                                       "half", "dx_m", "dy_m"])


# ── tile statistics -> field ──────────────────────────────────────────────────

def tile_medians(df: pd.DataFrame, seed: int = 0):
    """(N_TILE_ROWS, N_TILE_COLS) arrays: n, median dx/dy, bootstrap s.e."""
    rng = np.random.default_rng(seed)
    shape = (N_TILE_ROWS, N_TILE_COLS)
    n = np.zeros(shape, int)
    med = {c: np.full(shape, np.nan) for c in ("dx_m", "dy_m")}
    se = {c: np.full(shape, np.nan) for c in ("dx_m", "dy_m")}
    for (tr, tc), g in df.groupby(["tr", "tc"]):
        n[tr, tc] = len(g)
        if len(g) < MIN_N:
            continue
        idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
        for c in ("dx_m", "dy_m"):
            v = g[c].to_numpy()
            med[c][tr, tc] = np.median(v)
            se[c][tr, tc] = np.median(v[idx], axis=1).std()
    return n, med, se


def _neigh(a, tr, tc):
    """Values of the 3x3 neighbourhood (wrapping in longitude)."""
    rs = range(max(tr - 1, 0), min(tr + 2, N_TILE_ROWS))
    return np.array([a[r, c % N_TILE_COLS] for r in rs for c in (tc - 1, tc, tc + 1)])


def fit_field(df: pd.DataFrame, mode: str):
    """Tile medians -> cleaned values -> applied offsets, per --mode."""
    n, med, se = tile_medians(df)
    clean, applied = {}, {}
    for c in ("dx_m", "dy_m"):
        m, s = med[c], se[c]
        v = m.copy()
        for tr in range(N_TILE_ROWS):
            for tc in range(N_TILE_COLS):
                nb = _neigh(m, tr, tc)
                nb = nb[np.isfinite(nb)]
                nb_med = float(np.median(nb)) if len(nb) else 0.0
                if not np.isfinite(m[tr, tc]):
                    v[tr, tc] = nb_med
                    continue
                mad = 1.4826 * float(np.median(np.abs(nb - nb_med))) if len(nb) > 2 else np.inf
                d = abs(m[tr, tc] - nb_med)
                if d > 3 * max(mad, s[tr, tc]) and d > 30.0:
                    v[tr, tc] = nb_med
        s_f = np.where(np.isfinite(s), s, np.nanmedian(s))
        if mode == "none":
            a = np.zeros_like(v)
        elif mode == "gate":
            a = np.where(np.abs(v) > 2 * s_f, v, 0.0)
        elif mode == "shrink":
            tau2 = max(float(np.nanvar(v) - np.nanmean(s_f ** 2)), 0.0)
            a = v * tau2 / (tau2 + s_f ** 2)
        else:
            raise ValueError(mode)
        clean[c], applied[c] = v, a
    return n, med, se, clean, applied


def field_from_arrays(adx, ady):
    f = OffsetField.__new__(OffsetField)
    f.dx, f.dy = adx, ady
    return f


def validate(df: pd.DataFrame):
    """Fit on half A, evaluate per-tile residual medians on half B."""
    A, B = df[df.half == "A"], df[df.half == "B"]
    nB, medB, seB = tile_medians(B, seed=1)
    ok = (nB >= MIN_N)
    rows = []
    for mode in ("none", "gate", "shrink"):
        *_, applied = fit_field(A, mode)
        f = field_from_arrays(applied["dx_m"], applied["dy_m"])
        corr = np.array([f(la, lo) for la, lo in zip(B.lat, B.lon)])
        Br = B.assign(dx_m=B.dx_m.to_numpy() - corr[:, 0], dy_m=B.dy_m.to_numpy() - corr[:, 1])
        _, medR, _ = tile_medians(Br, seed=1)
        mag = np.hypot(medR["dx_m"], medR["dy_m"])[ok]
        rows.append({"mode": mode, "tiles": int(ok.sum()),
                     "median_resid_m": float(np.median(mag)),
                     "p95_resid_m": float(np.percentile(mag, 95)),
                     "mean_dy_m": float(np.nanmean(medR["dy_m"][ok])),
                     "mean_dx_m": float(np.nanmean(medR["dx_m"][ok]))})
    floor = np.hypot(seB["dx_m"], seB["dy_m"])[ok]
    rows.append({"mode": "noise floor (B s.e.)", "tiles": int(ok.sum()),
                 "median_resid_m": float(np.median(floor)),
                 "p95_resid_m": float(np.percentile(floor, 95)),
                 "mean_dy_m": np.nan, "mean_dx_m": np.nan})
    return pd.DataFrame(rows)


def plot(out_dir, n, clean, applied, se):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon_c = np.arange(N_TILE_COLS) * TILE_DEG + 1.5
    lat_c = LAT_TOP - np.arange(N_TILE_ROWS) * TILE_DEG - 1.5
    X, Y = np.meshgrid(lon_c, lat_c)
    fig, ax = plt.subplots(3, 1, figsize=(18, 15), constrained_layout=True)
    mag = np.hypot(applied["dx_m"], applied["dy_m"])
    im = ax[0].imshow(mag, extent=[0, 360, -60, 60], cmap="viridis", vmin=0, vmax=150)
    ax[0].quiver(X, Y, applied["dx_m"], applied["dy_m"], color="w", scale=4000, width=0.0012)
    ax[0].set_title("applied Kaguya - Robbins offset per tile (m); arrows = direction")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    im = ax[1].imshow(np.hypot(se["dx_m"], se["dy_m"]), extent=[0, 360, -60, 60], cmap="magma", vmin=0, vmax=80)
    ax[1].set_title("bootstrap s.e. of tile median (m)")
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    im = ax[2].imshow(n, extent=[0, 360, -60, 60], cmap="cividis")
    ax[2].set_title("craters measured per tile")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    for a in ax:
        a.set_xlabel("lon E"); a.set_ylabel("lat")
    p = os.path.join(out_dir, "offset_field.png")
    fig.savefig(p, dpi=90)
    print(f"  field plot -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/raw/kaguya_tco_registration")
    ap.add_argument("--craters_csv", default="data/raw/lunar_crater_database_robbins_2018.csv")
    ap.add_argument("--tile_dir", default=TILE_DIR)
    ap.add_argument("--min_diameter", type=float, default=1.0)
    ap.add_argument("--max_diameter", type=float, default=5.0)
    ap.add_argument("--per_tile", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--mode", choices=["auto", "gate", "shrink", "none"], default="auto",
                    help="auto = whichever of gate/shrink has the lower held-out median residual")
    ap.add_argument("--reuse", action="store_true", help="reuse crater_measurements.csv if present")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    meas_p = os.path.join(a.out_dir, "crater_measurements.csv")
    if a.reuse and os.path.exists(meas_p):
        df = pd.read_csv(meas_p, dtype={"crater_id": str})
    else:
        df = measure(a.craters_csv, a.tile_dir, a.min_diameter, a.max_diameter, a.per_tile, a.workers)
        df.to_csv(meas_p, index=False)
    print(f"  measured {len(df):,} craters; pooled median Kaguya-Robbins "
          f"dx={df.dx_m.median():.1f} m dy={df.dy_m.median():.1f} m")

    val = validate(df)
    val.to_csv(os.path.join(a.out_dir, "validation.csv"), index=False)
    print("\n  half-split validation (fit on A, residual tile medians on B):")
    print(val.round(1).to_string(index=False))

    mode = a.mode
    if mode == "auto":
        v = val.set_index("mode")
        mode = min(("gate", "shrink"), key=lambda m: v.loc[m, "median_resid_m"])
        if v.loc[mode, "median_resid_m"] >= v.loc["none", "median_resid_m"]:
            mode = "none"
    print(f"\n  applied mode: {mode}")

    n, med, se, clean, applied = fit_field(df, mode)
    tr, tc = np.meshgrid(np.arange(N_TILE_ROWS), np.arange(N_TILE_COLS), indexing="ij")
    out = pd.DataFrame({
        "tr": tr.ravel(), "tc": tc.ravel(),
        "lat_top": LAT_TOP - tr.ravel() * TILE_DEG, "lon_left": tc.ravel() * TILE_DEG,
        "n": n.ravel(),
        "med_dx_m": med["dx_m"].ravel(), "med_dy_m": med["dy_m"].ravel(),
        "se_dx_m": se["dx_m"].ravel(), "se_dy_m": se["dy_m"].ravel(),
        "clean_dx_m": clean["dx_m"].ravel(), "clean_dy_m": clean["dy_m"].ravel(),
        "apply_dx_m": applied["dx_m"].ravel(), "apply_dy_m": applied["dy_m"].ravel()})
    out.to_csv(os.path.join(a.out_dir, "tile_offsets.csv"), index=False)
    json.dump({"mode": mode, "fov": FOV, "diameter_km": [a.min_diameter, a.max_diameter],
               "per_tile": a.per_tile, "n_craters": int(len(df)),
               "pooled_median_dx_m": float(df.dx_m.median()), "pooled_median_dy_m": float(df.dy_m.median()),
               "sign": "offsets are Kaguya minus Robbins, metres, +E +N; KaguyaCropper samples "
                       "the tiles at Robbins position + offset"},
              open(os.path.join(a.out_dir, "registration.json"), "w"), indent=2)
    mag = np.hypot(out.apply_dx_m, out.apply_dy_m)
    print(f"  applied |offset|: median {mag.median():.1f} m, p95 {mag.quantile(.95):.1f} m, max {mag.max():.1f} m")
    plot(a.out_dir, n, clean, applied, se)


if __name__ == "__main__":
    main()
