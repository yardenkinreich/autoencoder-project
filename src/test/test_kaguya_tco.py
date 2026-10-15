"""
Tests for data/kaguya_tco.py (tile geometry, window assembly, the local
reprojection, the log-ratio albedo removal, the symmetry-centre estimator).
Run with: PYTHONPATH=$(pwd) python -m pytest src/test/test_kaguya_tco.py
Tests that need the real tiles skip when the tile directory is absent.
"""
import math
import os
import types

import numpy as np
import pytest

from src.data import kaguya_tco as K

HAVE_TILES = os.path.isdir(K.TILE_DIR)
needs_tiles = pytest.mark.skipif(not HAVE_TILES, reason="Kaguya tiles not present")


def test_parse_tile_name():
    assert K.parse_tile_name("TCO_MAPs02_N03E000N00E003SC.img") == (3, 0)
    assert K.parse_tile_name("TCO_MAPs02_S57E357S60E360SC.img") == (-57, 357)
    assert K.parse_tile_name("whatever.lbl") is None


@needs_tiles
def test_geotransform_matches_gdal_label():
    from osgeo import gdal
    lbl = os.path.join(K.TILE_DIR, "lon000", "TCO_MAPs02_N03E000N00E003SC.lbl")
    ds = gdal.Open(lbl)
    np.testing.assert_allclose(ds.GetGeoTransform(), K.tile_geotransform(3, 0), atol=1e-3)


def _raw_tile(lat_top, lon_left):
    f = lambda a: ("N" if a >= 0 else "S") + f"{abs(a):02d}"
    p = os.path.join(K.TILE_DIR, f"lon{lon_left:03d}",
                     f"TCO_MAPs02_{f(lat_top)}E{lon_left:03d}{f(lat_top - 3)}E{lon_left + 3:03d}SC.img")
    return np.memmap(p, dtype=">u2", mode="r", shape=(K.TILE_PX, K.TILE_PX))


@needs_tiles
def test_read_window_across_tile_corner():
    m = K.KaguyaMosaic()
    # corner shared by tiles N03E000 (tr=19,tc=0), N03E003, N00E000 (tr=20), N00E003
    r, c = 20 * K.TILE_PX, K.TILE_PX
    w = m.read_window(r - 5, r + 5, c - 5, c + 5)
    a, b = _raw_tile(3, 0), _raw_tile(3, 3)
    cc, d = _raw_tile(0, 0), _raw_tile(0, 3)
    expect = np.block([[a[-5:, -5:], b[-5:, :5]], [cc[:5, -5:], d[:5, :5]]])
    np.testing.assert_array_equal(w, expect)


@needs_tiles
def test_read_window_wraps_0_360():
    m = K.KaguyaMosaic()
    r = 20 * K.TILE_PX + 100
    w = m.read_window(r, r + 3, -4, 4)
    east, west = _raw_tile(0, 0), _raw_tile(0, 357)
    np.testing.assert_array_equal(w[:, :4], west[100:103, -4:])
    np.testing.assert_array_equal(w[:, 4:], east[100:103, :4])


@needs_tiles
def test_read_window_rejects_outside_coverage():
    with pytest.raises(K.OutOfCoverage):
        K.KaguyaMosaic().read_window(-1, 10, 0, 10)


@needs_tiles
def test_equatorial_crop_matches_native_window():
    # At the equator the aeqd grid at native spacing is (nearly) the source
    # grid itself, so the reprojected crop must equal a plain window.
    row, col = 214200 + 7, 3600 + 11                  # a pixel centre near 0.5N 1E
    lat, lon = K.LAT_TOP - row / K.PX_PER_DEG, col / K.PX_PER_DEG
    crop, extra = K.KaguyaCropper().crop(lat, lon, 1.0, 1.0)
    h = crop.shape[0] // 2
    assert crop.shape[0] % 2 == 1
    ref = K.KaguyaMosaic().read_window(row - h, row + h + 1, col - h, col + h + 1)
    ref = ref.astype(np.float32) * K.DN_SCALE
    r = np.corrcoef(crop[8:-8, 8:-8].ravel(), ref[8:-8, 8:-8].ravel())[0, 1]
    assert r > 0.99, r
    assert extra["proj"] == "aeqd" and crop.dtype == np.float32


def test_logratio_removes_multiplicative_albedo_step():
    # Synthetic world: a texture times a x2 albedo step down the middle.
    # Fake both the mosaic and the overview so this runs without the tiles.
    rng = np.random.default_rng(0)
    H, W = 2000, 2000
    tex = np.exp(0.2 * rng.standard_normal((H, W))).astype(np.float32)
    alb = np.where(np.arange(W)[None, :] < W // 2, 1.0, 2.0).astype(np.float32)
    refl = np.clip(tex * alb * 0.1, 0.02, 300)
    dn = np.round(refl / K.DN_SCALE).astype(np.uint16)
    row0, col0 = 21 * K.TILE_PX, 50 * K.TILE_PX       # place it near the equator

    class FakeMosaic:
        def read_window(self, r0, r1, c0, c1):
            return dn[r0 - row0:r1 - row0, c0 - col0:c1 - col0]

    b = 10
    logr = np.log(dn.astype(np.float32) * K.DN_SCALE)
    ov_l = logr.reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    ov = types.SimpleNamespace(block=b, m_per_px=K.M_PER_PX * b,
                               window=lambda r0, r1, c0, c1: _ovwin(ov_l, row0 // b, col0 // b, r0, r1, c0, c1))

    lat = K.LAT_TOP - (row0 + H / 2) / K.PX_PER_DEG
    lon = (col0 + W / 2) / K.PX_PER_DEG
    for mode in ("logratio", "ratio", "subtract"):
        cr = K.KaguyaCropper(albedo=mode, overview_dir="unused", sigma_min_km=0.5, albedo_k=0)
        cr._mosaic, cr._overview = FakeMosaic(), ov
        out, _ = cr.crop(lat, lon, 4.0, 1.0)       # 8 km wide, straddles the step
        h = out.shape[1] // 2
        left, right = np.nanmean(out[:, h - 300:h - 100]), np.nanmean(out[:, h + 100:h + 300])
        if mode == "logratio":
            assert abs(right - left) < 0.05, (left, right)    # log(2)=0.69 before
        elif mode == "ratio":
            assert abs(right / left - 1) < 0.05, (left, right)  # x2 before
        else:   # subtract: both sides centred on ~0 (raw means 0.1 vs 0.2)
            assert abs(left) < 0.02 and abs(right) < 0.02, (left, right)


def _ovwin(ov_l, orow0, ocol0, r0, r1, c0, c1):
    L = np.zeros((r1 - r0, c1 - c0), np.float32)
    V = np.zeros_like(L)
    for i in range(r0, r1):
        for j in range(c0, c1):
            a, b = i - orow0, j - ocol0
            if 0 <= a < ov_l.shape[0] and 0 <= b < ov_l.shape[1]:
                L[i - r0, j - c0], V[i - r0, j - c0] = ov_l[a, b], 1.0
    return L, V


def test_symmetry_center_recovers_known_shift():
    n = 256
    yy, xx = np.mgrid[:n, :n]
    for sx, sy in [(10, -6), (0, 0), (-14, 8)]:
        ring = np.exp(-((np.hypot(xx - (n - 1) / 2 - sx, yy - (n - 1) / 2 - sy) - 42) ** 2) / 18)
        dx, dy = K.symmetry_center(ring + 0.3 * np.cos(xx / 9))
        assert abs(dx * n - sx) <= 1.0 and abs(dy * n - sy) <= 1.0, (dx * n, dy * n)


def test_offset_field_interpolates_and_wraps(tmp_path):
    import pandas as pd
    rows = [{"tr": tr, "tc": tc, "apply_dx_m": 0.0, "apply_dy_m": 0.0}
            for tr in range(K.N_TILE_ROWS) for tc in range(K.N_TILE_COLS)]
    df = pd.DataFrame(rows)
    df.loc[(df.tr == 20) & (df.tc == 0), "apply_dy_m"] = 100.0
    p = tmp_path / "off.csv"
    df.to_csv(p, index=False)
    f = K.OffsetField(str(p))
    assert f(-1.5, 1.5) == pytest.approx((0.0, 100.0))           # tile centre
    assert f(-1.5, 0.0)[1] == pytest.approx(50.0)                 # halfway to tc=119
    assert f(-1.5, 359.99)[1] == pytest.approx(50.0, abs=1)       # wraps


def test_overview_window_excludes_clipped_shadow(tmp_path):
    # One block of sunlit log-reflectance 0.0 mixed 50/50 with DN-1 shadow:
    # the stored all-valid mean is (0 + log 0.01)/2, the sunlit mean must be 0.
    import json
    log1 = math.log(0.01)
    L = np.full((4, 4), log1 / 2, np.float32)
    V = np.ones((4, 4), np.float32)
    S = np.full((4, 4), 0.5, np.float32)
    S[0, 0], L[0, 0] = 1.0, log1             # an all-shadow block: weight 0
    for name, a in (("loglum", L), ("valid", V), ("lowsat", S), ("dn0", 1 - V)):
        np.save(tmp_path / f"{name}.npy", a)
    json.dump({"block": 30, "shape": [4, 4]}, open(tmp_path / "overview.json", "w"))
    ov = K.KaguyaOverview(str(tmp_path))
    sun, w = ov.window(0, 4, 0, 4)
    np.testing.assert_allclose(sun[1:, 1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(w[1:, 1:], 0.5, atol=1e-6)
    assert w[0, 0] == 0.0
