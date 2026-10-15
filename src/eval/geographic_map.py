"""
geographic_map.py — where on the Moon does a learned encoding put things?

Embeds a large crater pool with a frozen checkpoint, clusters the embeddings
into k groups, and plots every crater at its own lunar coordinates colored by
cluster, over the WAC mosaic itself as a basemap. The question it answers is
deliberately different from every other evaluation in this project: those all
ask "do the clusters agree with expert degradation labels" on a few hundred
labeled craters. This one asks "does whatever this encoder learned correspond
to anything SPATIAL" across ~100k craters and the whole mapped surface - a
question no labeled set can answer, because there aren't labels at that scale.

What to look for, and what NOT to conclude:
  - Coherent large-scale spatial structure (e.g. maria vs. highlands
    separating, or a latitude gradient) means the encoding tracks something
    real about lunar geology, since nothing in training ever saw a coordinate.
  - Structure that follows mosaic acquisition geometry instead (longitude
    stripes, discontinuities at seam boundaries, a latitude ramp mirroring
    illumination) is the warning sign - that's the encoder keying on imaging
    artifacts, the same class of confound confound_correlation() checks for
    per-cluster on the labeled sets.
  - Salt-and-pepper with no spatial organization is NOT automatically bad:
    crater degradation genuinely varies locally (a fresh crater sits next to
    an ancient one), so a good degradation encoder need not produce smooth
    regional blobs. Read this alongside the labeled-set metrics, never on
    its own.

Cluster colors are a fixed-order categorical palette (_MAP_CATEGORICAL),
NOT the _ORDINAL_RAMP the labeled-set plots use. Raw KMeans cluster ids are
arbitrary integers with no ordering, and visualize._cluster_color's own
docstring says why that matters: using an ordinal ramp here "would visually
imply an ordering that isn't there." (_cluster_color itself isn't reused
either - its nipy_spectral sampling puts black at one end and near-white at
the other, both effectively invisible on a grey mosaic basemap.)

EVERY k gets ONE COLOURED MAP. An earlier version switched to small
multiples above 8 clusters, on the correct observation that colour cannot
carry per-cluster identity past a handful of hues. But that answers the wrong
question: the reason to look at this figure is to see whether the encoding has
SPATIAL structure - above all whether it is banded by longitude, which would
be the mosaic artifact - and that reads perfectly well from a many-coloured
map. A coherent region of one colour is a localized cluster; vertical stripes
are the artifact. Per-cluster identity at k=100 is what the GeoJSON export is
for (load it in QGIS and style by the `cluster` field). For large k the
per-cluster panels are still written, as a supplementary figure rather than a
replacement.

The BASEMAP is always the plain morphology mosaic, whatever product the model
consumed. A high-pass residual has had exactly the maria/highland albedo
structure removed that makes a lunar map recognizable, so using it as a
backdrop produced a figure that looked like points floating on nothing.

Cluster ids are left EXACTLY as KMeans produced them - no reordering. An
earlier version of this script renumbered clusters by mean crater pixel
brightness so the two models' maps would use matching colors, which was a
bad idea and is deliberately not done: brightness is the very confound this
project has been trying to adjudicate, so ordering (and therefore coloring)
by it would make every map look brightness-organized BY CONSTRUCTION,
leaving a reader no way to tell whether spatial structure came from the
encoder or from the color assignment itself.

The consequence, stated plainly: colors are arbitrary WITHIN a model and
NOT comparable ACROSS models. Compare these maps on whether spatial
structure exists and what shape it takes - not cluster-color to
cluster-color. If a genuine cross-model correspondence is ever wanted, the
principled route is Hungarian matching on the two models' co-assignment
matrix over the same craters (which uses only the clusterings themselves,
no pixel statistics).
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd


def cluster_embeddings(latents: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    # n_init=10 rather than sklearn's newer default of 1: KMeans on 100k
    # points is cheap next to the embedding pass that produced them, and a
    # single init on a high-dimensional embedding is meaningfully more
    # likely to land in a poor local optimum.
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(latents)


def _basemap(basemap_tif: str, max_width: int = 2400):
    """Decimated read of the full mosaic for use as a plot background. The
    real thing is 109164x36388 - far past what a figure can show or memory
    wants - so this asks rasterio for an averaged, downsampled read rather
    than loading it and resizing afterward."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(basemap_tif) as src:
        scale = min(1.0, max_width / src.width)
        h, w = max(1, int(src.height * scale)), max(1, int(src.width * scale))
        img = src.read(1, out_shape=(h, w), resampling=Resampling.average)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    return img, extent


# The basemap is ALWAYS the plain morphology mosaic, whatever product the
# model was trained on. A high-pass residual has had the maria/highland
# albedo structure stripped out of it by construction, which is precisely
# the structure that makes a lunar map readable as a map.
_DEFAULT_BASEMAP = "data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif"

# Above this many clusters a LEGEND stops being usable (100 entries is not a
# legend), so it is dropped and the colours are documented as arbitrary ids.
# The MAP itself stays a single coloured map at every k - see _cluster_colors.
_MAX_LEGEND_CLUSTERS = 12

# Fixed-order categorical palette for the single-map case, Okabe-Ito derived
# and ordered so the closest colour-vision-deficiency pair is non-adjacent.
# Validated with the dataviz validator (light surface): lightness band PASS,
# chroma floor PASS, CVD separation PASS (worst adjacent deutan dE 11.0),
# normal-vision floor PASS; the one sub-3:1 contrast warning (#cc79a7) is
# relieved by the legend's visible text labels, which is the documented
# relief for that check. Deliberately NOT visualize._cluster_color's
# nipy_spectral sampling, whose endpoints are black and near-white - both
# effectively invisible against a grey mosaic basemap.
_MAP_CATEGORICAL = ["#0072b2", "#009e73", "#d55e00", "#cc79a7",
                    "#e69f00", "#56b4e9", "#000000", "#8b4513"]
# Single hue for the small-multiple panels: there, identity is carried by the
# panel title, so colour carries no information and one strong hue is correct.
_MAP_SINGLE = "#0072b2"

_COORD_NOTE = {
    "projected": "projected easting/northing in METRES, in the source mosaic's own "
                 "Equirectangular CRS - overlay directly on that mosaic with no "
                 "reprojection",
    "geographic": "geographic lon/lat in DEGREES (GCS_Moon / D_Moon_2000)",
}
_CRS_NOTE = {
    "projected": "USE THIS ONE to overlay on the WAC mosaic. Coordinates are already "
                 "in the mosaic's own CRS, so no datum transformation is needed - "
                 "which matters because the WAC GeoTIFFs declare DATUM[\"unknown\"], "
                 "and no GIS can transform between an unnamed datum and a named one. "
                 "Load with the .prj sidecar shipped beside this file.",
    "geographic": "Lon/lat degrees on D_Moon_2000. Correct in itself, but overlaying "
                  "it on the WAC mosaic requires a datum transformation the mosaic "
                  "cannot support (it declares DATUM[\"unknown\"]), so points may "
                  "appear displaced. Use the *_projected export against the mosaic; "
                  "use this one with other lunar layers that share D_Moon_2000. "
                  "GeoJSON RFC 7946 nominally mandates WGS84, which does not exist "
                  "for the Moon - hence the .prj sidecar.",
}


def _cluster_colors(k: int) -> list:
    """One colour per cluster, for ANY k.

    Up to 8 clusters this returns the validated Okabe-Ito-derived palette
    above, where colour genuinely carries identity and a legend works.

    Beyond that it samples matplotlib's `hsv` cycle. Be honest about what
    that is: at k=50 or k=100 no palette can make every cluster individually
    identifiable, and this one does not claim to. What it DOES do is let
    spatial structure show - a cluster occupying a coherent region reads as a
    coherent patch of colour, and longitude-banded structure reads as
    vertical stripes, which is the artifact this map exists to reveal. That
    is a real and different question from "which cluster is this one point
    in", and it is answerable at high k where per-cluster identification is
    not.

    `hsv` is chosen over a sequential ramp deliberately: cluster ids carry no
    order, and a light-to-dark ramp would imply one. It is cyclic and
    fully-saturated, so adjacent ids differ visibly without any implied
    ranking. Anyone needing per-cluster identity at high k should use the
    GeoJSON/CSV export and colour by the `cluster` field in QGIS."""
    if k <= len(_MAP_CATEGORICAL):
        return list(_MAP_CATEGORICAL[:k])
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("hsv")
    # Offset each successive colour by a large stride around the wheel so
    # NUMERICALLY adjacent cluster ids are also visually far apart; k-means
    # ids are arbitrary, but neighbouring ids landing on neighbouring hues
    # would still read as a false gradient.
    # The stride MUST be coprime with k, or (i*stride) % k does not visit every
    # id and colours repeat: at k=10 a stride of 4 produced only 5 distinct
    # colours, since gcd(4, 10) = 2. Start near the golden-ratio hop and walk
    # up to the first value coprime with k.
    from math import gcd
    stride = max(1, int(round(k * 0.381966)))      # 1 - 1/phi
    while gcd(stride, k) != 1:
        stride += 1
    order = [(i * stride) % k for i in range(k)]
    # o / k, NOT o / (k-1). `hsv` is CYCLIC: hue 0 and hue 1 are both red, so
    # spanning the closed interval [0, 1] makes the first and last ids
    # near-identical - measured #ff0000 and #ff0018 at k=10, indistinguishable
    # on a map. Dividing by k spans the half-open [0, 1) instead, so every id
    # gets its own arc of the wheel.
    colors = [cmap(o / k) for o in order]
    assert len(set(colors)) == k, f"stride {stride} did not cover all {k} ids"
    return colors


def spatial_concentration(df: pd.DataFrame) -> pd.Series:
    """Per-cluster RMS distance of its craters from their own centroid, in
    projected metres, divided by the same quantity for the whole pool. ~1.0
    means "spread exactly like the pool as a whole" (i.e. no spatial
    preference at all); well below 1.0 means the cluster is genuinely
    localized. This is the number that makes a k=100 map interpretable -
    with 100 arbitrary colours nobody can read a map, but "which clusters
    are spatially concentrated, and where" is still a real question."""
    def _rms(g: pd.DataFrame) -> float:
        return float(np.sqrt(((g["x"] - g["x"].mean()) ** 2
                              + (g["y"] - g["y"].mean()) ** 2).mean()))
    pool = _rms(df)
    return df.groupby("cluster")[["x", "y"]].apply(_rms) / (pool or 1.0)


def _draw_basemap(ax, img, extent, alpha: float = 1.0,
                  pct: tuple = (2.0, 98.0)) -> None:
    """The mosaic as a VISIBLE background.

    Two things went wrong in an earlier version, and both are fixed here.

    1. It defaulted to the SIGMA-100 HIGH-PASS mosaic, because that is what
       the model consumes. But a high-pass residual has had exactly the
       large-scale albedo structure removed that makes a lunar basemap
       recognizable - no maria/highland contrast, no basins, just local
       texture. The correct basemap is the plain morphology mosaic
       (_DEFAULT_BASEMAP), regardless of which product the model was
       trained on: its job is to tell the reader WHERE they are looking, not
       to reproduce the model's input.
    2. It used alpha=0.40 with 0.5/99.5 percentiles to keep the terrain
       recessive. Combined with (1) that rendered the basemap effectively
       invisible - a low-contrast grey at 40% opacity over a white axes
       background is indistinguishable from blank, so the figure looked like
       points floating on nothing.

    So: full opacity and a 2/98 percentile stretch. Legibility of the points
    is recovered on their side instead - larger markers with a thin white
    edge (see plot_map)."""
    lo, hi = np.nanpercentile(img, list(pct))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    ax.imshow(img, extent=extent, cmap="gray", vmin=lo, vmax=hi, alpha=alpha,
              interpolation="bilinear", aspect="equal", zorder=0)


def _style_axes(ax, xlabel: bool = True, ylabel: bool = True) -> None:
    ax.set_xlabel("projected easting (m)" if xlabel else "", fontsize=9, color="#555555")
    ax.set_ylabel("projected northing (m)" if ylabel else "", fontsize=9, color="#555555")
    ax.tick_params(labelsize=8, colors="#555555")
    for s in ax.spines.values():
        s.set_color("#cccccc")


def plot_map(df: pd.DataFrame, k: int, out_png: str, basemap_tif: str,
             title: str, basemap_cache: tuple | None = None,
             colors: list | None = None, legend_labels: list | None = None,
             legend_title: str | None = None) -> None:
    """A single coloured map, one colour per cluster, at EVERY k.

    This used to bail out to small multiples above 8 clusters on the grounds
    that colour cannot carry identity past that. True as far as it goes - but
    it answers the wrong question. The reason to look at this map is to see
    whether the encoding has spatial structure (in particular whether it is
    banded by longitude, which would be the mosaic artifact), and that reads
    perfectly well from a many-coloured map: a coherent region of one colour
    is a localized cluster, and vertical stripes are the artifact. Per-cluster
    identification at k=100 is what the GeoJSON export is for.

    Marker size and a thin white edge are load-bearing: the basemap now
    renders at full opacity (see _draw_basemap), so points need to hold their
    own against real terrain rather than against near-white."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    img, extent = basemap_cache if basemap_cache is not None else _basemap(basemap_tif)
    fig, ax = plt.subplots(figsize=(19, 8))
    _draw_basemap(ax, img, extent)

    ids = sorted(df["cluster"].unique())
    # colors/legend_labels are overridable because not every pool labelling is
    # a KMeans partition. pool_labeling.py paints degradation CLASSES, which
    # are ordinal - the categorical palette and the "colours arbitrary, not
    # ordered" legend below would both be actively wrong there.
    colors = colors or _cluster_colors(max(len(ids), k))
    # Fewer, larger points at small k; smaller at high k where 100k points
    # would otherwise merge into a solid sheet.
    size = 5.0 if k <= 12 else 3.0
    edge = 0.25 if k <= 12 else 0.0
    for i, c in enumerate(ids):
        sel = df["cluster"] == c
        ax.scatter(df.loc[sel, "x"], df.loc[sel, "y"], s=size,
                   color=colors[i % len(colors)],
                   linewidths=edge, edgecolors="white",
                   alpha=0.9, zorder=2, rasterized=True)

    # A legend past ~12 entries is not a legend. Above that the colours are
    # explicitly documented as arbitrary ids and the reader is pointed at the
    # export for identity.
    if len(ids) <= _MAX_LEGEND_CLUSTERS:
        handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=colors[i % len(colors)],
                   markeredgecolor="white", markeredgewidth=0.5,
                   label=(f"{legend_labels[int(c)]}" if legend_labels
                          else f"cluster {int(c)}")
                         + f"  (n={int((df['cluster'] == c).sum()):,})")
            for i, c in enumerate(ids)
        ]
        leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                        framealpha=0.92, fontsize=9,
                        title=legend_title or
                              f"k={k} — colours arbitrary, not ordered")
        leg.get_title().set_fontsize(9)
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f"{title}\ncolours are arbitrary cluster ids (no order); "
                     f"{len(ids)} clusters — use the GeoJSON export for "
                     f"per-cluster identity", fontsize=11)

    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def craters_crs_wkt() -> str:
    """The WKT of the CRS the crater lon/lat coordinates are actually in,
    taken from the pipeline itself rather than assumed.

    This is `preprocess_2.get_craters_crs()` - GCS_Moon, datum D_Moon_2000,
    spheroid Moon_2000_IAU_IAG with R = 1,737,400 m - and it is the right one
    because `crop_crater()` transforms FROM this CRS into each mosaic's own
    projected CRS. So the lat/lon in every metadata.csv and index.csv is
    expressed in it by construction.

    Two things worth knowing:
      - It is GEOGRAPHIC (degrees), despite its WKT carrying a stray
        PROJECTION["Equirectangular"] clause; pyproj reads the GEOGCS wrapper
        and reports is_geographic=True. The projected equirectangular CRS -
        which is what the x/y columns are in - belongs to the source mosaic
        (PROJCS, metres, central_meridian=0, same sphere).
      - It has NO authority code (`to_authority()` returns None), so no EPSG
        or ESRI code can honestly be written into the output. The full WKT is
        emitted instead."""
    from src.data.preprocess_2 import get_craters_crs
    return get_craters_crs().to_wkt()


def craters_crs_wkt1() -> str:
    """ESRI-flavoured WKT1 for the `.prj` sidecar, with the ANGULAR UNIT
    CORRECTED.

    `.prj` files conventionally carry WKT1, so that flavour is used. But the
    round-trip cannot be verbatim, because of a defect in the project's own
    CRS definition:

        preprocess_2.get_craters_crs()'s WKT declares  UNIT["metre",1]
        INSIDE its GEOGCS.

    A geographic CRS's axis unit must be angular. The stored lat/lon really
    are DEGREES, so emitting that WKT unchanged produces a .prj claiming
    degrees are metres - and QGIS would then place every crater at a
    fractionally-small offset from the origin instead of on the Moon. pyproj
    round-trips the mistake faithfully (`UNIT["Meter",1.0]`), so it has to be
    corrected here.

    Everything that describes the SHAPE of the body is still taken from the
    pipeline's own CRS - datum name, spheroid name, radius, inverse
    flattening - so this remains the CRS the data is in, with only the unit
    declaration repaired. The stray PROJECTION["Equirectangular"] clause in
    that same WKT is likewise dropped: the CRS is geographic (pyproj reports
    is_geographic=True), and the projected equirectangular CRS belongs to the
    source mosaic, not to these coordinates."""
    from src.data.preprocess_2 import get_craters_crs
    crs = get_craters_crs()
    e = crs.ellipsoid
    datum_name = crs.datum.name if crs.datum is not None else "D_Moon_2000"
    a = e.semi_major_metre if e is not None else 1737400.0
    invf = 0.0 if (e is None or e.is_semi_minor_computed or not e.inverse_flattening) \
        else e.inverse_flattening
    pm = crs.prime_meridian.longitude if crs.prime_meridian is not None else 0.0
    return (f'GEOGCS["{crs.name}",'
            f'DATUM["{datum_name}",'
            f'SPHEROID["{e.name if e is not None else "Moon_2000_IAU_IAG"}",'
            f'{a:.1f},{invf:.1f}]],'
            f'PRIMEM["Reference_Meridian",{float(pm):.1f}],'
            f'UNIT["Degree",0.017453292519943295]]')


def raster_crs_wkt(raster_path: str) -> tuple[str, str]:
    """(WKT1, WKT2) of a raster's own CRS, read from the raster.

    Used for the PROJECTED export, whose whole purpose is to carry exactly the
    CRS of the mosaic the points will be displayed over - so it is read from
    that file rather than reconstructed."""
    import rasterio
    import pyproj
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{raster_path} declares no CRS")
        crs = pyproj.CRS.from_user_input(src.crs)
    for flavour in ("WKT1_GDAL", "WKT1_ESRI"):
        try:
            w1 = crs.to_wkt(flavour)
            if w1:
                return w1, crs.to_wkt()
        except Exception:
            continue
    return crs.to_wkt(), crs.to_wkt()


def write_gpkg(df: pd.DataFrame, out_path: str, *, label: str, k: int,
               crs_wkt: str, layer: str | None = None) -> str:
    """A GeoPackage - THE export to use in QGIS.

    Why this and not GeoJSON: a GeoPackage is a single file that carries its
    own CRS internally, so there is no `.prj` sidecar to get separated from
    the data and no reliance on GeoJSON's legacy `crs` member (which RFC 7946
    removed, having mandated WGS84 - a projection that does not exist for the
    Moon). Load it and QGIS already knows where the points go.

    Geometry is the PROJECTED easting/northing in the source mosaic's own CRS,
    so overlaying it on that mosaic involves no reprojection at all. This
    matters specifically here: the WAC GeoTIFFs declare DATUM["unknown"], and
    no GIS can compute a datum transformation between an unnamed datum and a
    named one - which is why a lon/lat layer appears displaced over them.
    Longitude and latitude are still carried as ordinary attribute columns, so
    nothing is lost.

    Requires the x/y columns. Every geographic_clusters_k*.csv has them."""
    import geopandas as gpd
    from shapely.geometry import Point

    missing = {"x", "y", "cluster", "id"} - set(df.columns)
    if missing:
        raise ValueError(f"need columns {sorted(missing)} to write {out_path}")

    attrs = pd.DataFrame({
        "crater_id": df["id"].astype(str),
        "cluster": df["cluster"].astype("int32"),
        "lon_deg": df["lon"].astype(float) if "lon" in df else np.nan,
        "lat_deg": df["lat"].astype(float) if "lat" in df else np.nan,
        "diam_km": (df["diam"].astype(float) / 1000.0
                    if "diam" in df else np.nan),
    })
    gdf = gpd.GeoDataFrame(
        attrs,
        geometry=[Point(xy) for xy in zip(df["x"].astype(float),
                                          df["y"].astype(float))],
        crs=crs_wkt)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)          # GPKG append would stack duplicate layers
    gdf.to_file(out_path, layer=layer or f"clusters_k{k}", driver="GPKG")
    return out_path


def write_geojson(df: pd.DataFrame, out_path: str, *, label: str, k: int,
                  crs_wkt: str | None = None, mode: str = "geographic",
                  crs_wkt1: str | None = None) -> str:
    """One Point feature per crater, in lon/lat, carrying its cluster id -
    for loading into QGIS and styling by the `cluster` field.

    THE CRS COMES FROM THE DATA. GeoJSON (RFC 7946) mandates WGS84 and dropped
    the `crs` member, and WGS84 is simply wrong for the Moon - so a bare
    .geojson cannot state its own CRS in a standards-compliant way. Rather
    than leave the reader to guess, this writes all three of:

      1. a `.prj` SIDECAR next to the file, holding the WKT. GDAL/OGR reads
         `<name>.prj` for a GeoJSON that carries no CRS, so QGIS picks the
         Moon CRS up automatically as long as the two files travel together.
      2. a legacy `crs` member carrying the same WKT. Removed from the modern
         spec, but still honoured by some readers; harmless where it is not.
      3. the WKT inside the FeatureCollection's own `properties`, so the file
         remains self-describing even if the sidecar is lost.

    A `.zip` containing the geojson + prj is written alongside, because a
    single-file download would otherwise separate the data from its CRS.

    Coordinates are GEOGRAPHIC (lon, lat degrees), not the projected x/y the
    PNG plots, since that is what a GIS can reproject at will.

    Written one feature per line so a 100k-crater file stays streamable rather
    than becoming one enormous JSON line."""
    if mode not in ("geographic", "projected"):
        raise ValueError(f"mode must be 'geographic' or 'projected', got {mode!r}")
    if mode == "projected":
        # Projected easting/northing in the SOURCE MOSAIC's own CRS, so a GIS
        # overlays them on that mosaic with NO reprojection at all.
        #
        # THIS IS THE VARIANT THAT ACTUALLY ALIGNS, and the reason is a datum
        # mismatch that cannot be resolved from our side: the WAC GeoTIFFs
        # declare DATUM["unknown"], while the crater coordinates are in
        # D_Moon_2000. QGIS has no datum transformation between an unnamed
        # datum and a named one, so asking it to place GCS_Moon lon/lat points
        # over an "unknown"-datum raster misplaces them - which is exactly the
        # symptom reported. Handing it coordinates already in the raster's own
        # CRS removes the transformation step entirely.
        lon = df["x"].to_numpy(dtype=float)
        lat = df["y"].to_numpy(dtype=float)
    else:
        lon = df["lon"].to_numpy(dtype=float)
        lat = df["lat"].to_numpy(dtype=float)
    cid = df["cluster"].to_numpy()
    ids = df["id"].astype(str).to_numpy()
    diam_m = (df["diam"].to_numpy(dtype=float) if "diam" in df.columns
              else np.full(len(df), np.nan))

    wkt = crs_wkt or craters_crs_wkt()
    prj_wkt = crs_wkt1 or (craters_crs_wkt1() if crs_wkt is None else crs_wkt)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # The .prj sidecar is the mechanism that actually makes QGIS get this
    # right, so it is written first and unconditionally.
    prj_path = os.path.splitext(out_path)[0] + ".prj"
    with open(prj_path, "w") as f:
        f.write(prj_wkt)

    wkt_json = json.dumps(wkt)          # safely escaped for embedding
    with open(out_path, "w") as f:
        f.write('{"type": "FeatureCollection",\n')
        f.write(f' "name": {json.dumps(f"{label} k={k} clusters")},\n')
        # Legacy (pre-RFC-7946) crs member. Dropped from the modern spec, but
        # still read by several tools, and the only in-file way to state a
        # non-WGS84 CRS at all.
        f.write(' "crs": {"type": "name", "properties": {"name": '
                f'{wkt_json}}}}},\n')
        f.write(' "properties": {'
                f'"source": "src/eval/geographic_map.py", "k": {int(k)}, '
                f'"n_craters": {len(df)}, '
                f'"crs_wkt": {wkt_json}, '
                '"crs_origin": "preprocess_2.get_craters_crs() - the CRS the crater '
                'lat/lon are expressed in, since crop_crater transforms FROM it into '
                'each mosaic projected CRS", '
                f'"coordinates": {json.dumps(_COORD_NOTE[mode])}, '
                f'"crs_note": {json.dumps(_CRS_NOTE[mode])}, '
                '"cluster_note": "cluster ids are arbitrary k-means labels with no '
                'ordering; style as a categorical field"},\n')
        f.write(' "features": [\n')
        for i in range(len(df)):
            d = "null" if not np.isfinite(diam_m[i]) else f"{diam_m[i] / 1000.0:.5f}"
            f.write('  {"type": "Feature", "geometry": {"type": "Point", '
                    f'"coordinates": [{lon[i]:.6f}, {lat[i]:.6f}]}}, '
                    f'"properties": {{"crater_id": "{ids[i]}", '
                    f'"cluster": {int(cid[i])}, "diam_km": {d}}}}}')
            f.write(",\n" if i < len(df) - 1 else "\n")
        f.write(" ]\n}\n")

    # Zip the pair together. A lone .geojson download would arrive divorced
    # from its .prj, which is exactly how a layer ends up silently plotted on
    # a WGS84 Earth.
    import zipfile
    zip_path = os.path.splitext(out_path)[0] + "_geojson.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_path, os.path.basename(out_path))
        z.write(prj_path, os.path.basename(prj_path))
    return out_path


def plot_small_multiples(df: pd.DataFrame, k: int, out_png: str, basemap_tif: str,
                         title: str, concentration: pd.Series,
                         basemap_cache: tuple | None = None,
                         max_panels: int = 12) -> None:
    """One panel per cluster for large k, where a single map cannot work: no
    categorical palette stays distinguishable past ~8 hues, so a k=10 (let
    alone k=100) colour map is unreadable by construction. Here colour
    carries no information at all - each panel's title names its cluster -
    which is the standard resolution for "too many categories".

    Panels show the MOST SPATIALLY CONCENTRATED clusters first (see
    spatial_concentration), because those are the ones where the answer to
    "does this encoding correspond to anything geographic" is actually
    legible; a cluster spread exactly like the pool has nothing to show. All
    clusters, panelled or not, are still in the summary CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, extent = basemap_cache if basemap_cache is not None else _basemap(basemap_tif)
    chosen = list(concentration.sort_values().index[:max_panels])
    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 2.9 * nrows),
                             squeeze=False)

    for pos, cid in enumerate(chosen):
        ax = axes[pos // ncols][pos % ncols]
        _draw_basemap(ax, img, extent)
        sel = df["cluster"] == cid
        ax.scatter(df.loc[sel, "x"], df.loc[sel, "y"], s=2.0, linewidths=0,
                   color=_MAP_SINGLE, alpha=0.85, zorder=2, rasterized=True)
        ax.set_title(f"cluster {int(cid)} — n={int(sel.sum()):,}, "
                     f"spread {concentration.loc[cid]:.2f}× pool", fontsize=9)
        last_row = (pos // ncols) == (nrows - 1)
        _style_axes(ax, xlabel=last_row, ylabel=(pos % ncols == 0))
        ax.set_xticks([]) if not last_row else None

    for pos in range(len(chosen), nrows * ncols):
        axes[pos // ncols][pos % ncols].axis("off")

    fig.suptitle(f"{title}\nmost spatially concentrated {len(chosen)} of {k} clusters "
                 f"(spread < 1.00× = more localized than the pool as a whole)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--autoencoder-model", required=True,
                    choices=["mae", "cae", "dino", "dino_pretrained", "mae_pretrained"])
    ap.add_argument("--dat_path",
                    default="data/processed_wac_100m_new/sigma/100/geomap_100k_d1-12/craters_wide.dat")
    ap.add_argument("--metadata_csv",
                    default="data/processed_wac_100m_new/sigma/100/geomap_100k_d1-12/metadata.csv")
    ap.add_argument("--basemap", default=_DEFAULT_BASEMAP,
                    help="ALWAYS the plain morphology mosaic by default, even when the "
                         "model was trained on a high-pass product - a residual has no "
                         "maria/highland contrast and is unreadable as a map.")
    ap.add_argument("--k", type=int, nargs="+", default=[4, 10, 100],
                    help="One or more cluster counts. The embedding pass (the "
                         "expensive part) runs ONCE and every k reuses it.")
    ap.add_argument("--label", required=True,
                    help="Short model label used in the figure title and filenames.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crs_from", default=None,
                    help="Raster whose CRS the PROJECTED GeoJSON is written in "
                         "(default: --basemap). Must be the mosaic the points will be "
                         "displayed over, so no reprojection is needed.")
    ap.add_argument("--no_small_multiples", action="store_true",
                    help="Skip the supplementary per-cluster panel figure for large k.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from src.eval.holdout import embed_holdout

    meta = pd.read_csv(args.metadata_csv, dtype={"id": str})
    print(f"pool: {len(meta):,} craters from {args.metadata_csv}")

    # row_idx=None deliberately: embed_holdout() only keeps the memmap lazy
    # when it isn't asked for a subset, and this pool is the whole point.
    latents = embed_holdout(args.dat_path, args.checkpoint, args.autoencoder_model)
    print(f"embedded: {latents.shape}")
    if len(latents) != len(meta):
        raise ValueError(
            f"metadata.csv has {len(meta)} rows but the memmap embedded to "
            f"{len(latents)} - these are positionally joined (see "
            f"pipeline.labels_from_memmap_csv's docstring), so a mismatch "
            f"means they were not written by the same preprocess_2.py run.")

    # Read the basemap once too - it's a decimated read of a 109164x36388
    # raster, so re-reading it per k would dominate the runtime of every
    # plot after the first.
    basemap_cache = _basemap(args.basemap)

    for k in args.k:
        print(f"\n=== k={k} ===")
        clusters = cluster_embeddings(latents, k, seed=args.seed)
        df = meta.assign(cluster=clusters)

        csv_path = os.path.join(args.out_dir, f"geographic_clusters_k{k}.csv")
        df.to_csv(csv_path, index=False)

        concentration = spatial_concentration(df)
        summary = (df.groupby("cluster")
                     .agg(n=("id", "size"), mean_diam_km=("diam", lambda s: s.mean() / 1000.0),
                          mean_lat=("lat", "mean"), mean_abs_lat=("lat", lambda s: s.abs().mean()),
                          mean_lon=("lon", "mean"))
                     .reset_index())
        summary["spatial_spread_vs_pool"] = summary["cluster"].map(concentration)
        summary = summary.sort_values("spatial_spread_vs_pool")
        summary_path = os.path.join(args.out_dir, f"geographic_clusters_k{k}_summary.csv")
        summary.to_csv(summary_path, index=False)
        if k <= _MAX_LEGEND_CLUSTERS:
            print(summary.to_string(index=False))
        else:
            print(f"{len(summary)} clusters; sizes min/median/max "
                  f"{summary['n'].min()}/{int(summary['n'].median())}/{summary['n'].max()}; "
                  f"most concentrated: "
                  f"{summary.head(3)[['cluster', 'spatial_spread_vs_pool']].to_dict('records')}")

        title = (f"{args.label} — {len(df):,} craters (1-12 km), embeddings clustered into "
                 f"k={k}, plotted at their lunar coordinates")
        png_path = os.path.join(args.out_dir, f"geographic_map_k{k}.png")
        # A single COLOURED map at every k. Spatial structure - and above all
        # longitude banding - reads fine from many colours; per-cluster
        # identity at high k is what the GeoJSON export is for.
        plot_map(df, k, png_path, args.basemap, title=title,
                 basemap_cache=basemap_cache)

        # Small multiples are still produced for large k, as a SUPPLEMENT
        # rather than a replacement: "which clusters are spatially
        # concentrated, and where" stays a real question that one crowded map
        # cannot answer.
        if k > _MAX_LEGEND_CLUSTERS and not args.no_small_multiples:
            plot_small_multiples(
                df, k, os.path.join(args.out_dir, f"geographic_map_k{k}_panels.png"),
                args.basemap, title=title, concentration=concentration,
                basemap_cache=basemap_cache)

        # TWO exports, because they serve different jobs and only one of them
        # aligns on the mosaic. See _CRS_NOTE.
        pw1, pw2 = raster_crs_wkt(args.crs_from or args.basemap)
        proj_path = os.path.join(
            args.out_dir, f"geographic_clusters_k{k}_projected.geojson")
        write_geojson(df, proj_path, label=args.label, k=k, mode="projected",
                      crs_wkt=pw2, crs_wkt1=pw1)
        geo_path = os.path.join(
            args.out_dir, f"geographic_clusters_k{k}_geographic.geojson")
        write_geojson(df, geo_path, label=args.label, k=k, mode="geographic")
        print(f"wrote {proj_path}  <- use this one on the mosaic")
        print(f"wrote {geo_path}")

        json.dump({"label": args.label, "checkpoint": args.checkpoint, "k": k,
                   "n_craters": int(len(df)), "dat_path": args.dat_path,
                   "cluster_sizes": {int(r.cluster): int(r.n) for r in summary.itertuples()}},
                  open(os.path.join(args.out_dir, f"geographic_map_k{k}.json"), "w"),
                  indent=2, default=float)
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
