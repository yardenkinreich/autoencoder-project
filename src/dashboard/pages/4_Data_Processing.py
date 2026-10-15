"""Data Processing: every (source mosaic x high-pass filter x normalization)
combination, and what each one does to the pixels - broken down by degradation
class, because that is the only breakdown in which the two clipping bugs this
page documents were visible at all.

Reads logs/data_profiles/processing_compare.json, written by
`PYTHONPATH=. python -m src.eval.processing_compare`. Never recomputed live,
same convention as the rest of this dashboard.

Distinct from the "Data Sources" page, which profiles the crop sets that
trained models actually consumed. This page is upstream of that: it is about
choosing what those crop sets should be."""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import methodology as M

PROFILE = "logs/data_profiles/processing_compare.json"

st.set_page_config(page_title="Data Processing", layout="wide")
st.title("Data Processing")
st.caption(
    "What each source mosaic, high-pass filter and normalization does to the pixels, "
    "measured on the 642 expert-labeled craters. Every table is broken down by "
    "degradation class - a preprocessing step that is lossy *uniformly* is a "
    "trade-off, but one that is lossy *in proportion to the label* writes the answer "
    "into the input.")

if not os.path.exists(PROFILE):
    st.info(f"{PROFILE} not found - run "
            "`PYTHONPATH=. python -m src.eval.processing_compare` first.")
    st.stop()

P = json.load(open(PROFILE))
V = P["variants"]
CLASSES = P["class_names"]
CLASS_LONG = P["class_long"]
NORM_ORDER = P["norm_order"]
NORM_LABEL = P["norm_label"]
FIGS = P.get("figures_dir", "logs/data_profiles/processing_compare")

if P.get("skipped"):
    st.warning("Not yet available (mosaic missing on disk): " +
               ", ".join(f"`{s['name']}`" for s in P["skipped"]))

# ── 0. what each thing actually is ───────────────────────────────────────────
st.header("0. What each of these is")
st.caption(
    "A variant name like `emp_sigma200` is a source mosaic plus a filter. This section "
    "says what each half means and where it came from, so the tables below can be read "
    "without having to reconstruct the provenance.")

t_src, t_filt, t_norm = st.tabs(["Source mosaics", "The Gaussian high-pass filter",
                                 "The normalizations"])

with t_src:
    st.markdown("""
Both mosaics come from the **same instrument** — the Wide Angle Camera (WAC) on NASA's
Lunar Reconnaissance Orbiter — at the **same resolution** (~100 m/pixel), covering
**±60° latitude and all longitudes**. They differ entirely in how the brightness of each
pixel was decided, and that difference is the point of comparing them.

#### `morph_*` — WAC_GLOBAL morphology mosaic
`data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif`

Assembled from **three observation windows chosen to hold the sun low** — incidence angle
between roughly 55° and 75°. Low sun means long shadows and strong topographic shading,
which is exactly what a geologist wants for reading *shape*: this product was built for
morphologic interpretation, and it deliberately **keeps** the shading.

- **Why that matters here:** degradation is largely a bright-rim/deep-shadow signal, and
  this mosaic preserves it. It is what every model in this project has trained on.
- **The cost:** a pixel's brightness depends on *when and where* it was imaged, so
  contrast jumps at the boundaries between source tiles. That is the "seam" problem, and
  it is inherent to the product, not to anything we did to it.

  Measured on 100,000 craters (`seam_check.py`, 15° edge width), mean local contrast by
  longitude zone is 0.00926 (left seam) / 0.01152 (middle) / 0.01294 (right seam) —
  **−19.6% and +12.4% relative to the middle**. Note that the *effect size* is small:
  η² = 0.0098, i.e. zone membership explains ~1% of contrast variance, because
  within-zone spread is large. The `p = 1.5e-213` reflects n = 100,000, not importance.
  Quote the relative step or the η², and say which — see **Methods**.

#### `emp_*` — WAC_EMP_643 empirically normalized mosaic
`data/raw/wac_uniform/WAC_EMP_643_100m_180.tif`

Every pixel **photometrically corrected to one fixed illumination geometry** — incidence
30°, emission 0°, phase 30° — using GLD100 topography, then median-stacked over ~142
observations (Boyd et al. 2012). The 643 nm band, at 304 pixels/degree ≈ 100 m/pixel.

- **Why we want it:** correcting to a single geometry largely removes the "imaged at a
  different time" effect. Measured with the frozen encoder over all 30 catalog ×
  normalization cells, the longitude confound η² averages **0.154–0.168 for the EMP
  variants against 0.265–0.276 for the morphology variants — roughly a 40% reduction**,
  consistent across all three filter settings. (An earlier note quoted "8% / 7% / 0.7%"
  seam steps for EMP; that figure was never actually measured on this data and has been
  withdrawn in favour of the η² above.)
- **The cost, and the real tension:** incidence 30° is near-overhead sun, which
  *minimizes* topographic shading. The normalization that removes the seam also removes
  some of the shading the degradation cue rides on. No summary statistic settles that —
  hence the example-crater grids further down, and the probe.

#### Why not WAC_HAPKE
A third product exists, Hapke-model corrected (Sato et al. 2014), and is more principled
photometrically — but it is published **only at 400 m/pixel**. At 4× coarser than this
project's native resolution, a 1 km crater spans about 2 pixels. Rejected on resolution
grounds. An earlier seam figure of 0.2% derived from an upsampled Hapke file has been
retracted; it was not a like-for-like measurement.
""")

with t_filt:
    st.markdown(r"""
#### What `sigma100` / `sigma200` mean
A **Gaussian high-pass filter**. It blurs a copy of the mosaic and subtracts it:

$$\text{output} = \text{image} - \text{blur}_\sigma(\text{image})$$

σ is in **pixels**, and at 100 m/pixel that makes the scale physical:

| variant | σ | ground scale | removes variation slower than |
|---|---|---|---|
| `*_raw` | — | — | nothing (unfiltered) |
| `*_sigma100` | 100 px | **10 km** | ~10 km |
| `*_sigma200` | 200 px | **20 km** | ~20 km |

**What it removes:** anything varying more slowly than that scale — regional albedo
differences, broad illumination gradients, the large-scale brightness drift that makes
one mosaic tile sit at a different level from its neighbour.

**What it keeps:** structure smaller than the scale — crater rims, walls, terraces,
ejecta. That is the morphology we care about.

**What the output is:** a *residual*, not a brightness. It is centred on ~0 (mean ≈ 1e−8
by construction) and is freely **negative** — a pixel darker than its surroundings. This
is why a `[0,1]` mapping is a strange fit for the filtered variants: the data is already
zero-centred, and forcing it into `[0,1]` then standardizing back to zero is a round trip
that only costs information.

**Why two values of σ:** at 100 m/pixel, σ=100 px is 10 km — comparable to the diameter of
the craters under study (1–12 km). So σ=100 risks subtracting away part of the crater
itself, treating the crater's own bowl as "slow variation". σ=200 (20 km) is the more
conservative choice. Which is better is an empirical question, which is why both exist.

#### Two things the filter does *not* do
- **It does not fix the seams.** The tile-boundary contrast steps are the same magnitude
  before and after filtering. The filter was investigated as a possible *cause* of the
  seams and exonerated; it is also not a cure.
- **It no longer clips.** It used to — see section 1. That clip is now off, and the stored
  residuals are unclipped.
""")

with t_norm:
    st.markdown("""
Normalization is what turns stored pixels into numbers a network can use. Every option
below is some version of the same two steps: **subtract a middle, divide by a spread.**

That much is an *affine* transform — a change of units, like °C to °F. Every pixel keeps
its exact relationship to every other pixel and the whole thing inverts perfectly.
**Nothing is destroyed.** The options differ in how the middle and spread are chosen, and
in whether anything is thrown away afterwards.

| name | formula | middle / spread from | throws data away? |
|---|---|---|---|
| `none` | *x* | — | no (but unusable by a model) |
| `global` | (*x*−p1)/(p99−p1), then clip to [0,1] | 1st/99th percentile | **yes — the clip** |
| `global_wide` | (*x*−p0.01)/(p99.99−p0.01), clipped | 0.01/99.99 percentile | technically, but ~nothing |
| `zscore` | (*x*−mean)/std | mean, standard deviation | no |
| `robust` | (*x*−median)/(1.4826·MAD) | median, MAD | no |
| `asinh` | asinh of the `robust` value | median, MAD | no |

**Why `robust` rather than `zscore`.** The mean and standard deviation are themselves
dragged upward by a few very bright pixels. The median and the MAD (median absolute
deviation — the typical distance from the median) are not: they describe the *typical*
pixel. The 1.4826 factor is a convention that makes MAD comparable to a standard
deviation, so the two agree on well-behaved data and diverge only where the tails are
heavy. The `std ÷ MAD·1.4826` column in section 2 measures exactly that divergence.

**What `asinh` is for.** After `robust`, a fresh crater's rim can land at +33 — not an
error, just a true statement that it is 33 typical variations brighter than average
ground. But networks do badly with inputs that large. `asinh` squashes: it does
essentially nothing to small values and grows only logarithmically for big ones.

| input | asinh output |
|---|---|
| 0.5 | 0.48 |
| 1 | 0.88 |
| 3 | 1.82 |
| 10 | 3.00 |
| 33 | **4.19** |

**The crucial property is that it is strictly monotone.** The brightest pixel stays the
brightest, the second-brightest stays second. Clipping made the top few percent all
*identical*; `asinh` keeps them ordered, just closer together. If clipping is an
overexposed photograph, `asinh` is turning down the highlights — dimmer, but the detail
survives. It is the standard treatment for this in astronomy (Lupton et al. 1999, the
SDSS "asinh magnitude"), where the same problem appears as bright stars against faint sky.

**Which is used where.** `asinh` for the raw mosaics, whose reflectance is right-skewed
because albedo is bounded below by zero and unbounded above. `robust` for the high-pass
variants, where the filter has already subtracted the slow bright component away, so
there is little tail left to squash. Both are recorded per catalog in `loader.json`.
""")


def _eta_badge(eta: float) -> str:
    """seam_check.py's effect-size convention, so numbers are comparable
    across this project's analyses: 0.01 small / 0.06 medium / 0.14 large."""
    if eta is None or not np.isfinite(eta):
        return "n/a"
    tag = ("negligible" if eta < 0.01 else "small" if eta < 0.06
           else "medium" if eta < 0.14 else "**LARGE**")
    return f"{eta:.4f} ({tag})"


# ── 1. the two clipping findings ─────────────────────────────────────────────
st.header("1. Two clips were destroying fresh craters")
st.markdown(
    "Both were found only by grouping a pixel statistic by class, and neither raised "
    "anything. They are independent and they compound.")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.subheader("Clip A — inside the high-pass filter")
        st.markdown(
            "`local_highpass_filter_gpu` clipped its residual to the **whole mosaic's** "
            "0.1/99.9 percentiles. Those bounds are set by smooth mare and highland, "
            "which is nearly all of the Moon; craters are the high-contrast minority, "
            "so crater rims and shadows were flattened onto the bounds and **stored "
            "that way** in a 17 GB product.\n\n"
            "Now off by default (`clip_pct=None`), with a saturation gate in "
            "`highpass_mosaic.py` that refuses to save a product whose pixels pile up "
            "at its extremes. The old products are kept under "
            "`data/raw/**/sigma_legacyclip/` so historical runs stay explicable.")
with c2:
    with st.container(border=True):
        st.subheader("Clip B — inside the normalization")
        st.markdown(
            "Mapping to `[0,1]` with a 1st/99th-percentile window requires a clip, and "
            "that window is fitted on a corpus that is ~63% degraded craters. Fresh "
            "craters *are* the high-dynamic-range ones, so they are the ones it "
            "saturates.\n\n"
            "Still available as `norm=\"global\"` for backward comparability with every "
            "historical run, but no longer the default - see section 4.")

st.subheader("Clip A: pixels stored on exactly the product's min or max")
rows = []
for name, s in V.items():
    sat = s["stored_saturation"]
    r = {"variant": name, "overall %": sat["overall_pct"]}
    for c in CLASSES:
        r[c] = sat["per_class"].get(c, {}).get("pct_at_extremes", float("nan"))
    r["eta² vs class"] = _eta_badge(sat["vs_class"]["eta_sq"])
    rows.append(r)
sat_df = pd.DataFrame(rows)
st.dataframe(
    sat_df.style.format({c: "{:.4f}%" for c in ["overall %"] + CLASSES})
          .background_gradient(cmap="Oranges", subset=CLASSES),
    width="stretch", hide_index=True)
st.caption(
    "A high-pass residual is a difference of two smooth fields, so it should have no "
    "value spike at its extremes at all - any nonzero number here is an upstream clip. "
    "The raw (unfiltered) mosaics measure ~0.0003%, which is the clean baseline; the "
    "legacy sigma products measured 0.99% (morphology, and monotonic in degradation "
    "with **exactly 0.00%** for `v-dr`) and 11.87% (EMP).")
M.explain("highpass")

# ── 2. the variants themselves ───────────────────────────────────────────────
st.header("2. The data variants")
rows = []
for name, s in V.items():
    r = s["raw"]
    ns = s["native_size"]
    rows.append({
        "variant": name, "source": os.path.basename(s.get("source_mosaic", "")),
        "n craters": s["n_craters"], "n pixels": s["n_pixels"],
        "native px (min-max)": f"{ns['min']}-{ns['max']}",
        "min": r["min"], "max": r["max"], "mean": r["mean"], "std": r["std"],
        "median": r["median"], "MAD·1.4826": r["mad_std"],
        "std ÷ MAD·1.4826": r.get("tail_ratio_std_over_madstd", float("nan")),
        "skew": r["skew"], "excess kurtosis": r["kurtosis_excess"],
    })
st.dataframe(
    pd.DataFrame(rows).style.format({
        "min": "{:.5g}", "max": "{:.5g}", "mean": "{:.5g}", "std": "{:.5g}",
        "median": "{:.5g}", "MAD·1.4826": "{:.5g}", "std ÷ MAD·1.4826": "{:.2f}",
        "skew": "{:.2f}", "excess kurtosis": "{:.1f}", "n pixels": "{:,}"}),
    width="stretch", hide_index=True)
st.caption(
    "**Skew is the number that decides the normalization.** The raw mosaics are strongly "
    "right-skewed - reflectance is bounded below by zero and unbounded above, so bright "
    "ejecta and fresh rims form a long upper tail - and no centre-and-scale mapping makes "
    "them symmetric; it only slides the tail around. The high-pass variants are far closer "
    "to symmetric, because subtracting a local mean is exactly what removes the albedo "
    "tail.\n\n"
    "**`std ÷ MAD·1.4826` is the companion diagnostic:** it compares a tail-sensitive "
    "scale estimate against a tail-immune one, so it is 1.00 for Gaussian data and rises "
    "with tail weight. At 1.86 (morphology) and 2.45 (EMP) the raw mosaics' `std` is "
    "roughly double what the bulk of the data warrants - which is why `zscore`'s nominal "
    "\"std = 1\" is misleading there, and why a monotone compression rather than an affine "
    "map is the right tool. Excess kurtosis says the same thing a third way.")

dist_png = os.path.join(FIGS, "raw_distributions.png")
if os.path.exists(dist_png):
    st.image(dist_png, width="stretch")
    st.caption(
        "Small multiples with a per-panel x-axis, deliberately: raw reflectance "
        "(~[0, 0.55]) and a high-pass residual (~[-0.05, 0.09]) are different physical "
        "quantities on different scales, and a shared axis would compress most panels "
        "into a spike. Dotted lines are the p1/p99 clip bounds - note they fall inside "
        "the fresh classes' tails, which is section 3.")
M.explain("terrain_shortcut")

# ── 3. what each normalization destroys ──────────────────────────────────────
st.header("3. What each normalization destroys, by class")
clip_png = os.path.join(FIGS, "clip_by_class.png")
if os.path.exists(clip_png):
    st.image(clip_png, width="stretch")

st.subheader("Clipped-pixel share and post-normalization range")
sel_variant = st.selectbox("Variant", list(V.keys()), key="norm_variant")
s = V[sel_variant]
rows = []
for norm in NORM_ORDER:
    nb = s["normalizations"][norm]
    ov = nb["overall"]
    r = {"normalization": NORM_LABEL[norm],
         "range": f"[{ov['min']:.2f}, {ov['max']:.2f}]",
         "mean": ov["mean"], "std": ov["std"],
         ">|3|": ov.get("pct_beyond_3", float("nan")),
         ">|5|": ov.get("pct_beyond_5", float("nan"))}
    for c in CLASSES:
        r[f"{c} clip%"] = nb["per_class"].get(c, {}).get("clip_pct", float("nan"))
    r["eta² clip vs class"] = _eta_badge(nb["clip_vs_class"]["eta_sq"])
    rows.append(r)
clip_cols = [f"{c} clip%" for c in CLASSES]
tail_cols = [">|3|", ">|5|"]
st.dataframe(
    pd.DataFrame(rows).style.format(
        {**{c: "{:.3f}%" for c in clip_cols + tail_cols},
         "mean": "{:.4f}", "std": "{:.4f}"})
        .background_gradient(cmap="Oranges", subset=clip_cols)
        .background_gradient(cmap="Blues", subset=tail_cols),
    width="stretch", hide_index=True)
st.caption(
    "**Two different costs, two different sets of columns.** The orange `clip%` columns "
    "are pixels *destroyed* - irreversible, and on the bounded normalizations they land "
    "overwhelmingly on the fresh classes. The blue `>|3|` / `>|5|` columns are pixels "
    "merely *far from centre* - nothing is lost, but a network fed inputs at 33σ will "
    "struggle. `zscore` and `robust` clip nothing by construction (their clip columns and "
    "eta² are exactly 0, being affine and fully invertible), and pay instead in tail "
    "extent. `asinh` is the only candidate that is low on both: it clips nothing *and* "
    "keeps everything inside ~±4, because it compresses the tail monotonically rather "
    "than cutting it off.")
M.explain("clip_by_class")
M.explain("intensity_features")

st.subheader("The same craters under every normalization")
crop_png = os.path.join(FIGS, f"crops_{sel_variant}.png")
if os.path.exists(crop_png):
    st.image(crop_png, width="stretch")
    st.caption(
        "Rows are craters grouped by class; columns are normalizations; the last column "
        "flags in orange **which** pixels the p1/p99 clip destroys, with the percentage "
        "beneath. The pixels lost are the bright rim and the ejecta - the morphology the "
        "label is based on - not a scattering of noise. Every panel is autoscaled "
        "individually (the scales are genuinely incomparable) with its value range "
        "printed beneath, so nothing is hidden by the autoscale.")
else:
    st.info(f"No crop grid for {sel_variant} yet.")

st.subheader("One crater, every dataset × every normalization")
st.caption(
    "The transpose of the grid above. That one holds the dataset fixed and varies the "
    "normalization; this one varies both at once on a single crater, which is what "
    "separates \"that difference is the mosaic\" from \"that difference is the filter\" "
    "from \"that difference is the normalization\". The same example craters appear "
    "here and in every per-variant grid — they are chosen once, stratified by class, "
    "from the craters present in *all* variants.")
examples = P.get("example_craters") or []
if examples:
    opts = {f"{e['crater_id']}  ({e['class']} — {CLASS_LONG.get(e['class'], '')})": e
            for e in examples}
    pick = st.selectbox("Example crater", list(opts.keys()), key="example_crater")
    e = opts[pick]
    if os.path.exists(e["png"]):
        st.image(e["png"], width="stretch")
        st.caption(f"Crater `{e['crater_id']}` — class **{e['class']}** "
                   f"({CLASS_LONG.get(e['class'], '')}). Present in "
                   f"{len(e['variants'])} variant(s): {', '.join(e['variants'])}.")
    else:
        st.info(f"{e['png']} not found - re-run processing_compare.")
else:
    st.info("No cross-dataset example grids yet - re-run "
            "`PYTHONPATH=. python -m src.eval.processing_compare`.")

# ── 4. the recommendation ────────────────────────────────────────────────────
st.header("4. What the loader does now")
st.markdown(
    "Catalogs store **raw, native-resolution** pixels, so none of this is baked in - "
    "normalization is a loader policy (`loader.json` / `norm_stats.json`), changeable "
    "without rebuilding a multi-GB store. That is precisely why raw storage was worth "
    "the disk.")
st.markdown(
    "| norm | what it does | verdict |\n"
    "|---|---|---|\n"
    "| `asinh` | `asinh((x - median) / (1.4826·MAD))`, no clip | **Default for the raw "
    "mosaics.** Linear near zero, logarithmic in the tail (Lupton et al. 1999, the SDSS "
    "asinh magnitude). Clips nothing, lands in ~±4, and is *strictly monotone* - fresh "
    "craters stay the brightest things in the image instead of collapsing onto one "
    "saturated value. |\n"
    "| `robust` | `(x - median) / (1.4826·MAD)`, no clip | **Default for the high-pass "
    "variants**, where the albedo tail is already gone and the extra compression buys "
    "little. Zero-centred, scale set by the bulk of the data, nothing discarded. |\n"
    "| `zscore` | `(x - mean) / std`, no clip | Acceptable, and equivalent to `robust` "
    "for symmetric data. Its centre and scale are themselves dragged by the tails they "
    "are meant to describe - see the `std ÷ MAD` column in section 2. |\n"
    "| `global_wide` | `[0,1]` clip at p0.01/p99.99 | Use when a bounded range is "
    "genuinely required (e.g. an augmentation that assumes it, like solarize). The clip "
    "is near-inert. |\n"
    "| `global` | `[0,1]` clip at p1/p99 | **Historical only.** Kept so past runs remain "
    "reproducible and comparable. Not for new work. |\n"
    "| `none` | raw | Artifact analysis only. Feeding it to a model gives near-zero "
    "input variance and degenerate features, silently. |\n")
st.markdown(
    "The split between `asinh` and `robust` is not a hedge - it follows the skew column. "
    "Raw reflectance is skew ~2.6-3.5 and needs the compression; a high-pass residual has "
    "already had its albedo tail removed by construction and does not. Both are set per "
    "catalog in `loader.json`, and the intended comparison is empirical: the same frozen "
    "encoder over each option on the 642 labeled craters.")
st.info(
    "**One implementation, two consumers.** The arithmetic lives in "
    "`src/data/norm_stats.apply()`, and both the tensors a model trains on "
    "(`crater_catalog_dataset.CraterCatalog`) and the figures on this page call it. "
    "They cannot drift apart.")
M.explain("normalizations")
M.explain("catalog_storage")

with st.expander("Fitted constants per variant"):
    for name, s in V.items():
        st.markdown(f"**{name}**")
        st.json({n: s["normalizations"][n]["constants"] for n in NORM_ORDER
                 if s["normalizations"][n]["constants"]}, expanded=False)

with st.expander("Per-class detail (all variants, all normalizations)"):
    rows = []
    for name, s in V.items():
        for norm in NORM_ORDER:
            nb = s["normalizations"][norm]
            for c in CLASSES:
                d = nb["per_class"].get(c)
                if not d:
                    continue
                rows.append({"variant": name, "norm": norm, "class": c,
                             "class (long)": CLASS_LONG[c],
                             "n craters": d["n_craters"],
                             "clip low %": d["clip_low_pct"],
                             "clip high %": d["clip_high_pct"],
                             "clip %": d["clip_pct"],
                             "mean": d["mean"], "std": d["std"],
                             "min": d["min"], "max": d["max"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
