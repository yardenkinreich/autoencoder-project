"""
kaguya_tco.py — read the JAXA SELENE/Kaguya TC "Ortho Map Seamless" tiles
(SLN-L-TC-5-ORTHO-MAP-SEAMLESS-V2.0, files TCO_MAPs02_*.img) as one global
mosaic, and cut native-resolution crater crops out of it.

THE SOURCE
----------
3x3 degree tiles, 10800x10800, big-endian uint16, reflectance = DN * 0.01,
simple cylindrical (equirectangular) on a 1737.4 km sphere, 3600 px/deg
(8.4232 m/px at the equator). DN 0 = no data (DUMMY), DN 1 = low saturation
(deep shadow - real, just clipped), DN 32767 = high saturation. On disk:
complete 60S-60N (4,800 tiles); stray tiles beyond that are ignored.

WHY NOT GDAL
------------
890 of the 4,813 tiles have no .lbl, and GDAL's PDS driver cannot open a bare
.img. The geometry is fully determined by the filename instead: a tile named
..._{N|S}aaE bbb... has its top-left PIXEL CENTRE at exactly (lat=+-aa,
lon=bbb) (verified against gdalinfo on labeled tiles), so global pixel
(row, col) has its centre at lat = 60 - row/3600, lon = col/3600. Tiles abut
with no overlap.

WHY A LOCAL REPROJECTION
------------------------
The WAC path (helper_functions.crop_crater) cuts an equirectangular window and
stretches it by 1/cos(lat). That is exact only at the crater's own latitude;
cos(lat) changes by ~6% across a 100 km window at 60 deg. Here every crop is
resampled ONCE onto a local azimuthal-equidistant grid centred on the crater
(north up, square ground pixels at the native 8.4232 m), which is exact
everywhere in the crop and makes tile edges and the 0/360 seam invisible.
"""
from __future__ import annotations

import math
import os
import re
from collections import OrderedDict

import cv2
import numpy as np
import pyproj
from scipy.ndimage import gaussian_filter, map_coordinates

TILE_DIR = "/home/liorr/craters_encoder/data/raw/kaguya_tco"

MOON_R = 1737400.0
PX_PER_DEG = 3600
TILE_DEG = 3
TILE_PX = TILE_DEG * PX_PER_DEG                    # 10800
LAT_TOP = 60                                       # row 0 centre is at 60N
N_TILE_ROWS = 2 * LAT_TOP // TILE_DEG              # 40
N_TILE_COLS = 360 // TILE_DEG                      # 120
N_ROWS = N_TILE_ROWS * TILE_PX                     # 432,000
N_COLS = N_TILE_COLS * TILE_PX                     # 1,296,000
M_PER_PX = MOON_R * math.pi / 180.0 / PX_PER_DEG   # 8.4231529 m

DN_SCALE = 0.01
DUMMY = 0
LOW_SAT = 1
HIGH_SAT = 32767

_NAME_RE = re.compile(r"TCO_MAPs02_([NS])(\d{2})E(\d{3})([NS])(\d{2})E(\d{3})SC\.img$")


class OutOfCoverage(ValueError):
    """The requested window leaves the 60S-60N band the tiles cover."""


class TooDark(ValueError):
    """More of the crop is clipped at low saturation (DN 1) than allowed -
    e.g. the very-low-sun block at ~47-60S, 143-210E, which is ~90% DN 1."""


def parse_tile_name(name: str):
    """'TCO_MAPs02_N03E000N00E003SC.img' -> (lat_top, lon_left) = (3, 0)."""
    m = _NAME_RE.search(name)
    if m is None:
        return None
    lat_top = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    return lat_top, int(m.group(3))


def tile_geotransform(lat_top: int, lon_left: int):
    """GDAL-style geotransform (metres, eqc lon_0=0) of one tile, pixel-corner
    origin. Matches what GDAL derives from the .lbl (see the module docstring);
    the unit test checks exactly that."""
    deg = MOON_R * math.pi / 180.0
    return (lon_left * deg - M_PER_PX / 2, M_PER_PX, 0.0,
            lat_top * deg + M_PER_PX / 2, 0.0, -M_PER_PX)


def lonlat_to_pix(lon, lat):
    """Global fractional (col, row) of a lon/lat; integers are pixel centres.
    lon is taken mod 360 - callers that need continuity across the seam
    unwrap themselves (see KaguyaCropper)."""
    return np.mod(lon, 360.0) * PX_PER_DEG, (LAT_TOP - np.asarray(lat)) * PX_PER_DEG


class KaguyaMosaic:
    """All tiles as one virtual (N_ROWS, N_COLS) uint16 raster.

    Tiles are opened lazily as read-only memmaps and kept in a small LRU, so a
    process touches only the tiles its crops need. A missing tile reads as
    DUMMY (0), which every consumer treats as no data.
    """

    def __init__(self, tile_dir: str = TILE_DIR, cache_size: int = 64):
        self.tile_dir = tile_dir
        self.cache_size = cache_size
        self._cache: OrderedDict = OrderedDict()
        self.paths = {}
        for sub in sorted(os.listdir(tile_dir)):
            d = os.path.join(tile_dir, sub)
            if not (sub.startswith("lon") and os.path.isdir(d)):
                continue
            for f in os.listdir(d):
                p = parse_tile_name(f)
                if p is None:
                    continue
                lat_top, lon_left = p
                if lat_top > LAT_TOP or lat_top - TILE_DEG < -LAT_TOP:
                    continue                   # the stray >60N tiles
                tr = (LAT_TOP - lat_top) // TILE_DEG
                tc = lon_left // TILE_DEG
                self.paths[(tr, tc)] = os.path.join(d, f)

    def __getstate__(self):                    # picklable for worker pools
        s = self.__dict__.copy()
        s["_cache"] = OrderedDict()
        return s

    def tile(self, tr: int, tc: int):
        key = (tr, tc)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        p = self.paths.get(key)
        mm = None if p is None else np.memmap(p, dtype=">u2", mode="r",
                                              shape=(TILE_PX, TILE_PX))
        self._cache[key] = mm
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return mm

    def read_window(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """uint16 [r0:r1, c0:c1] of the global raster. Columns wrap mod
        N_COLS (c0 may be negative or c1 > N_COLS); rows may not leave the
        coverage."""
        if r0 < 0 or r1 > N_ROWS or r1 <= r0 or c1 <= c0:
            raise OutOfCoverage(f"rows [{r0},{r1}) outside [0,{N_ROWS})")
        out = np.zeros((r1 - r0, c1 - c0), dtype=np.uint16)
        r = r0
        while r < r1:
            tr, rr = divmod(r, TILE_PX)
            rn = min(r1, (tr + 1) * TILE_PX)
            c = c0
            while c < c1:
                tcu, cc = divmod(c, TILE_PX)          # unwrapped tile column
                cn = min(c1, (tcu + 1) * TILE_PX)
                t = self.tile(tr, tcu % N_TILE_COLS)
                if t is not None:
                    out[r - r0:rn - r0, c - c0:cn - c0] = \
                        t[rr:rr + (rn - r), cc:cc + (cn - c)]
                c = cn
            r = rn
        return out


# ── overview access (built by kaguya_overview.py) ─────────────────────────────

class KaguyaOverview:
    """The block-averaged overview: mean log-reflectance + valid fraction on a
    regular grid, `block` native pixels per overview pixel."""

    def __init__(self, overview_dir: str):
        import json
        meta = json.load(open(os.path.join(overview_dir, "overview.json")))
        self.block = int(meta["block"])
        self.m_per_px = M_PER_PX * self.block
        shape = tuple(meta["shape"])
        self.loglum = np.load(os.path.join(overview_dir, "loglum.npy"), mmap_mode="r")
        self.valid = np.load(os.path.join(overview_dir, "valid.npy"), mmap_mode="r")
        self.lowsat = np.load(os.path.join(overview_dir, "lowsat.npy"), mmap_mode="r")
        assert self.loglum.shape == shape == self.valid.shape == self.lowsat.shape

    def window(self, r0, r1, c0, c1):
        """(sunlit loglum, sunlit weight) over overview rows/cols; columns
        wrap, rows outside the coverage come back with weight 0 so a
        normalized convolution simply ignores them.

        SUNLIT ONLY: clipped shadow (DN 1) is not albedo, and averaging it in
        drags the low-pass down - by up to log(100) in the low-sun block at
        ~47-60S 143-210E, which made the ratio crops there ~500x too bright
        (first full build). loglum averages log(r) over all DN>0 pixels, so the
        DN>1 mean is recovered exactly:
            (loglum*valid - lowsat*log(0.01)) / (valid - lowsat).
        """
        H, W = self.loglum.shape
        cols = np.arange(c0, c1) % W
        ra, rb = max(r0, 0), min(r1, H)
        L = np.zeros((r1 - r0, c1 - c0), np.float32)
        V = np.zeros_like(L)
        if rb > ra:
            lg = np.asarray(self.loglum[ra:rb][:, cols], np.float64)
            v = np.asarray(self.valid[ra:rb][:, cols], np.float64)
            s = np.asarray(self.lowsat[ra:rb][:, cols], np.float64)
            w = np.clip(v - s, 0.0, None)
            sun = np.where(w > 1e-6, (lg * v - s * math.log(LOW_SAT * DN_SCALE)) / np.maximum(w, 1e-6), 0.0)
            L[ra - r0:rb - r0] = sun
            V[ra - r0:rb - r0] = w
        return L, V


# ── registration field (built by kaguya_register.py) ──────────────────────────

class OffsetField:
    """Per-tile Kaguya-minus-Robbins offset (metres, +E +N), interpolated
    bilinearly between tile centres (wrapping in longitude) so the correction
    has no steps at tile edges."""

    def __init__(self, offsets_csv: str):
        import pandas as pd
        df = pd.read_csv(offsets_csv)
        self.dx = np.zeros((N_TILE_ROWS, N_TILE_COLS))
        self.dy = np.zeros((N_TILE_ROWS, N_TILE_COLS))
        self.dx[df.tr, df.tc] = df.apply_dx_m
        self.dy[df.tr, df.tc] = df.apply_dy_m

    def __call__(self, lat: float, lon: float):
        u = (LAT_TOP - lat) / TILE_DEG - 0.5           # fractional tile row
        v = (lon % 360.0) / TILE_DEG - 0.5             # fractional tile col
        u = min(max(u, 0.0), N_TILE_ROWS - 1.0)
        r0 = min(int(math.floor(u)), N_TILE_ROWS - 2)
        fu = u - r0
        c0 = int(math.floor(v))
        fv = v - c0
        out = []
        for f in (self.dx, self.dy):
            a = f[r0, c0 % N_TILE_COLS] * (1 - fv) + f[r0, (c0 + 1) % N_TILE_COLS] * fv
            b = f[r0 + 1, c0 % N_TILE_COLS] * (1 - fv) + f[r0 + 1, (c0 + 1) % N_TILE_COLS] * fv
            out.append(float(a * (1 - fu) + b * fu))
        return out[0], out[1]


# ── crater cropper ────────────────────────────────────────────────────────────

_LONLAT = "+proj=longlat +R=1737400 +no_defs"


def aeqd_lonlat_grid(lat0: float, lon0: float, side: int, m_per_px: float):
    """lon/lat (deg) of every pixel centre of a side x side north-up local
    azimuthal-equidistant grid centred on (lat0, lon0). Exact pyproj on a
    coarse lattice, cubic-interpolated to full resolution - the mapping is
    smooth on the scale of a crop, so this is accurate to far below a pixel
    and ~100x cheaper for the 3000+ px crops."""
    tr = pyproj.Transformer.from_crs(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +R=1737400 +units=m +no_defs",
        _LONLAT, always_xy=True)
    step = max(1, side // 48)
    nodes = np.unique(np.r_[np.arange(0, side, step), side - 1])
    half = (side - 1) / 2.0
    gx, gy = np.meshgrid((nodes - half) * m_per_px, (half - nodes) * m_per_px)
    lon, lat = tr.transform(gx, gy)
    lon = (lon - lon0 + 180.0) % 360.0 - 180.0 + lon0      # unwrap around lon0
    if step == 1:
        return lon, lat
    # fractional node index of every output pixel, for map_coordinates
    k = np.interp(np.arange(side), nodes, np.arange(len(nodes)))
    kk_y, kk_x = np.meshgrid(k, k, indexing="ij")
    full = [map_coordinates(a, [kk_y, kk_x], order=3, mode="nearest") for a in (lon, lat)]
    return full[0], full[1]


class KaguyaCropper:
    """crop(lat, lon, diam_km, offset) -> (float32 crop or None, extra fields).

    Crop side = diam * (1 + offset) at the native 8.4232 m/px - the same FOV
    convention as crop_crater (offset 1.0 = 2xD, the repo's "wide"). Returned
    values:
      albedo="none"     reflectance (DN * 0.01)
      albedo="ratio"    reflectance / exp(L_sigma): albedo divided out, shading
                        kept LINEAR (shadows look as in the raw crop). Default.
      albedo="subtract" reflectance - exp(L_sigma): the WAC sigma100 recipe
                        (subtractive Gaussian high-pass) on Kaguya - removes the
                        regional brightness offset, keeps absolute shading size.
      albedo="logratio" log(reflectance) - L_sigma. Same albedo removal, but the
                        log stretches the 0.01-0.1 range of deep shadow as much
                        as the whole sunlit range, so shadow interiors become
                        the dominant contrast (seen in the smoke sheets).
      L_sigma is the SUNLIT (DN>1) weighted Gaussian low-pass of log reflectance from
      the overview, sigma = max(sigma_min_km, k * D) on the ground.
    A crop touching no-data (DN 0) comes back with NaN there; build_catalog's
    existing non-finite check then skips and records the crater.
    """

    def __init__(self, tile_dir: str = TILE_DIR, albedo: str = "none",
                 overview_dir: str | None = None, sigma_min_km: float = 10.0,
                 albedo_k: float = 2.0, offsets_csv: str | None = None,
                 m_per_px: float = M_PER_PX, max_frac_lowsat: float = 1.0):
        if albedo not in ("none", "ratio", "logratio", "subtract"):
            raise ValueError(albedo)
        if albedo != "none" and overview_dir is None:
            raise ValueError(f"albedo={albedo} needs overview_dir")
        self.tile_dir, self.albedo = tile_dir, albedo
        self.overview_dir, self.offsets_csv = overview_dir, offsets_csv
        self.sigma_min_km, self.albedo_k, self.m_per_px = sigma_min_km, albedo_k, m_per_px
        self.max_frac_lowsat = max_frac_lowsat
        self._mosaic = self._overview = self._field = None

    def __getstate__(self):                    # each worker reopens lazily
        s = self.__dict__.copy()
        s["_mosaic"] = s["_overview"] = s["_field"] = None
        return s

    @property
    def mosaic(self):
        if self._mosaic is None:
            self._mosaic = KaguyaMosaic(self.tile_dir)
        return self._mosaic

    @property
    def overview(self):
        if self._overview is None and self.overview_dir:
            self._overview = KaguyaOverview(self.overview_dir)
        return self._overview

    @property
    def field(self):
        if self._field is None and self.offsets_csv:
            self._field = OffsetField(self.offsets_csv)
        return self._field

    def sigma_km(self, diam_km: float) -> float:
        return max(self.sigma_min_km, self.albedo_k * diam_km)

    def crop(self, lat: float, lon: float, diam_km: float, offset: float):
        px = self.m_per_px
        # Odd side: the crater centre is a pixel centre, so a crater sitting on
        # a source pixel centre is reproduced without half-pixel interpolation.
        side = int(math.ceil(diam_km * 1000.0 * (1.0 + offset) / px)) | 1
        extra = {"m_per_px": px, "proj": "aeqd", "reg_dx_m": 0.0, "reg_dy_m": 0.0,
                 "sigma_km": np.nan, "frac_lowsat": np.nan, "frac_highsat": np.nan}

        # Registration: the field is where Kaguya puts a crater relative to
        # Robbins, so sample the tiles at (Robbins position + offset) to land
        # on the crater as Kaguya actually shows it.
        slat, slon = lat, lon
        if self.field is not None:
            dx, dy = self.field(lat, lon)
            extra["reg_dx_m"], extra["reg_dy_m"] = dx, dy
            slat = lat + math.degrees(dy / MOON_R)
            slon = lon + math.degrees(dx / (MOON_R * math.cos(math.radians(lat))))

        glon, glat = aeqd_lonlat_grid(slat, slon, side, px)
        col = glon * PX_PER_DEG                       # unwrapped, may be <0 or >N_COLS
        row = (LAT_TOP - glat) * PX_PER_DEG
        m = 4                                          # cubic support + slack
        r0, r1 = int(math.floor(row.min())) - m, int(math.ceil(row.max())) + m + 1
        c0, c1 = int(math.floor(col.min())) - m, int(math.ceil(col.max())) + m + 1
        dn = self.mosaic.read_window(r0, r1, c0, c1)  # raises OutOfCoverage

        mapx = (col - c0).astype(np.float32)
        mapy = (row - r0).astype(np.float32)
        refl = dn.astype(np.float32) * DN_SCALE
        bad = (dn == DUMMY).astype(np.float32)
        lowsat = (dn == LOW_SAT).astype(np.float32)
        highsat = (dn == HIGH_SAT).astype(np.float32)

        # Anti-alias x: source columns are 8.42*cos(lat) m apart, output
        # pixels 8.42 m, so area-average x by cos(lat) before the remap
        # (no-op near the equator).
        f = math.cos(math.radians(lat))
        if f < 0.95:
            w_new = max(1, int(round(refl.shape[1] * f)))
            fx = w_new / refl.shape[1]
            refl = cv2.resize(refl, (w_new, refl.shape[0]), interpolation=cv2.INTER_AREA)
            bad = cv2.resize(bad, (w_new, bad.shape[0]), interpolation=cv2.INTER_AREA)
            lowsat = cv2.resize(lowsat, (w_new, lowsat.shape[0]), interpolation=cv2.INTER_AREA)
            highsat = cv2.resize(highsat, (w_new, highsat.shape[0]), interpolation=cv2.INTER_AREA)
            mapx = ((mapx + 0.5) * fx - 0.5).astype(np.float32)

        out = cv2.remap(refl, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        badm = cv2.remap(bad, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        extra["frac_lowsat"] = float(cv2.remap(lowsat, mapx, mapy, cv2.INTER_LINEAR).mean())
        extra["frac_highsat"] = float(cv2.remap(highsat, mapx, mapy, cv2.INTER_LINEAR).mean())
        if extra["frac_lowsat"] > self.max_frac_lowsat:
            raise TooDark(f"frac_lowsat={extra['frac_lowsat']:.3f} > {self.max_frac_lowsat}")
        # Cubic overshoot next to clipped shadow can dip below the DN-1 floor;
        # clamp to the floor (needed anyway before the log).
        out = np.maximum(out, LOW_SAT * DN_SCALE)

        if self.albedo != "none":
            sig = self.sigma_km(diam_km)
            extra["sigma_km"] = sig
            low = self._lowpass(glon, glat, lat, sig)
            if self.albedo == "logratio":
                out = np.log(out) - low
            elif self.albedo == "subtract":
                # WAC-style high-pass: remove the brightness OFFSET, keep the
                # shading's absolute size (highland shading stays larger)
                out = out - np.exp(low)
            else:
                out = out / np.exp(low)

        out[badm > 1e-3] = np.nan
        return out.astype(np.float32), extra

    def _lowpass(self, glon, glat, lat0, sigma_km):
        """Valid-weighted Gaussian low-pass of overview log-reflectance,
        sampled at every crop pixel. sigma is isotropic on the GROUND, so it
        is wider in overview columns by 1/cos(lat)."""
        ov = self.overview
        b, opx = ov.block, ov.m_per_px
        sy = sigma_km * 1000.0 / opx
        sx = sy / max(math.cos(math.radians(lat0)), 0.05)
        orow = (LAT_TOP - glat) * PX_PER_DEG / b - 0.5 + 0.5 / b
        ocol = glon * PX_PER_DEG / b - 0.5 + 0.5 / b
        my, mx = int(math.ceil(3 * sy)) + 2, int(math.ceil(3 * sx)) + 2
        r0, r1 = int(math.floor(orow.min())) - my, int(math.ceil(orow.max())) + my + 1
        c0, c1 = int(math.floor(ocol.min())) - mx, int(math.ceil(ocol.max())) + mx + 1
        L, V = ov.window(r0, r1, c0, c1)
        num = gaussian_filter(L * V, (sy, sx), mode="constant", truncate=3.0)
        den = gaussian_filter(V, (sy, sx), mode="constant", truncate=3.0)
        low = num / np.maximum(den, 1e-6)
        return cv2.remap(low.astype(np.float32), (ocol - c0).astype(np.float32),
                         (orow - r0).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)


# ── crater-centre estimator (shared by registration and the smoke figure) ─────

def symmetry_center(img: np.ndarray, n: int = 128, max_lag: int = 24):
    """Centre of point symmetry of the rim-edge magnitude, as (dx, dy) in
    FRACTIONS of the image side (+x right, +y down) from the image centre.

    Autocorrelating the edge map with its 180-degree rotation peaks at twice
    the centre's displacement. Lighting changes which rim side is bright but
    not where the edges are, so the estimate is not dragged sunward. Noisy per
    crater (~0.2 D scatter, measured), unbiased in the median - use it for
    per-tile statistics only, never to re-centre a single crater.
    """
    from scipy.signal import fftconvolve
    a = cv2.resize(np.asarray(img, np.float32), (n, n), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    e = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.0
    e = e * np.exp(-(xx ** 2 + yy ** 2) / (2 * (0.45 * n) ** 2))
    e = (e - e.mean()) / (e.std() + 1e-9)
    c = fftconvolve(e, e, mode="same")
    m = n // 2
    w = c[m - max_lag:m + max_lag + 1, m - max_lag:m + max_lag + 1]
    iy, ix = np.unravel_index(np.argmax(w), w.shape)
    return (ix - max_lag) / 2.0 / n, (iy - max_lag) / 2.0 / n
