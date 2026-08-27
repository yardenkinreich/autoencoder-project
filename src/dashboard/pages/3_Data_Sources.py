"""Data Sources: distribution profiles for the underlying crop sets
themselves (brightness/sobel-texture/saturation/diameter, and their
correlation with lat/lon), not model results. Reads
logs/data_profiles/profile.json - written by
`PYTHONPATH=src python -m eval.profile_data_sources` - never recomputed
live, same convention as the rest of this dashboard.

Scoped to configs/data_sources.yaml: only crop sets whose data_tag is
actually used by a trained model, not every raw mosaic variant that ever
existed. Sources sharing a data_tag can still be genuinely different crop
conventions (diameter range, FOV) - see each source's own "note"."""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import streamlit as st

PROFILE_PATH = "logs/data_profiles/profile.json"

st.set_page_config(page_title="Data Sources", layout="wide")
st.title("Data Sources")

if not os.path.exists(PROFILE_PATH):
    st.info(f"{PROFILE_PATH} not found - run "
           "`PYTHONPATH=src python -m eval.profile_data_sources` first.")
    st.stop()

profile = json.load(open(PROFILE_PATH))
sources = profile["sources"]
source_names = list(sources.keys())

st.caption(
    f"{len(source_names)} data sources, {profile['sample_size']} craters sampled per source "
    "(or all of them, if fewer exist). Brightness/sobel/saturation are computed directly from "
    "the crop pixels (same features the per-run confound check uses); diameter/lat/lon come "
    "from each source's own metadata.csv.")

st.warning(
    "**Raw-mosaic sources (`wac100m_raw__*`) and sigma-filtered sources (`wac100m_sigma100__*`) "
    "are on different absolute pixel scales entirely** - different processing pipelines (no "
    "highpass filter vs. sigma-100 highpass), not a bug. They're also **not resolution- or "
    "channel-matched** - see the Resolution column below (some are older-convention crops, e.g. "
    "224x224/3-channel or 64x64, not this project's current 128x128/1-channel). Treat brightness/"
    "sobel/saturation as comparable *within* a matching pipeline+resolution, not directly across it.")

# --- Source overview ---------------------------------------------------------
with st.container(border=True):
    st.subheader("Sources")
    rows = []
    for name, s in sources.items():
        r = s["resolution"]
        resolution = f"{r['height']}x{r['width']}, {r['channels']}ch" if r["channels"] else "unknown"
        row = {"source": name, "data_tag": s["data_tag"], "fov": s["fov"],
              "resolution": resolution, "n_sampled": s["n_sampled"], "note": s["note"]}
        if "geo_extent" in s:
            ge = s["geo_extent"]
            row["lat_range"] = f"{ge['lat_min']:.1f} to {ge['lat_max']:.1f}"
            row["lon_range"] = f"{ge['lon_min']:.1f} to {ge['lon_max']:.1f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --- Per-feature distribution -------------------------------------------------
FEATURE_LABELS = {
    "brightness": "Brightness (mean pixel value, 0-255)",
    "sobel_mean": "Sobel texture/sharpness (mean gradient magnitude)",
    "frac_saturated": "Fraction of pixels saturated (>=254/255)",
    "diam_km": "Crater diameter (km)",
}
feature_cols = [c for c in FEATURE_LABELS if any(c in s["features"] for s in sources.values())]


def _feature_summary_table(feat: str) -> pd.DataFrame:
    rows = []
    for name, s in sources.items():
        f = s["features"].get(feat)
        if not f:
            continue
        rows.append({
            "source": name, "n": s["n_sampled"], "load_rate": f["load_rate"],
            "mean": f["mean"], "std": f["std"],
            "p5": f["p5"], "p25": f["p25"], "p50": f["p50"], "p75": f["p75"], "p95": f["p95"],
        })
    return pd.DataFrame(rows)


def _render_feature(feat: str, show_view_toggle: bool = True) -> None:
    if show_view_toggle:
        vc1, vc2 = st.columns([3, 1])
        view = vc1.radio("View", ["Box plot (summary)", "Histogram (actual shape, per source)"],
                         horizontal=True, key=f"view_{feat}")
        if "Histogram" in view:
            shared_y = vc2.checkbox(
                "Shared y-axis", key=f"shared_y_{feat}",
                help="Sources have very different sample sizes (Julie n=150 vs. 2000 elsewhere) - "
                     "shared y-axis switches to density (area=1) so heights are genuinely "
                     "comparable, instead of raw counts making the small-N source look flat.")
            hist_key = "histograms_shared_y" if shared_y else "histograms"
            plot_path = profile.get(hist_key, {}).get(feat)
        else:
            plot_path = profile.get("plots", {}).get(feat)
    else:
        plot_path = profile.get("plots", {}).get(feat)
    if plot_path and os.path.exists(plot_path):
        st.image(plot_path, width="stretch")

    summary_df = _feature_summary_table(feat)
    st.dataframe(
        summary_df, width="stretch", hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ["load_rate", "mean", "std", "p5", "p25", "p50", "p75", "p95"]
        })
    if not summary_df.empty and (summary_df["load_rate"] < 1.0).any():
        st.caption(
            "load_rate < 1.0 means some crop files for that source couldn't be read "
            "(missing on disk) - excluded from the stats above, not silently counted as 0.")


st.divider()
with st.container(border=True):
    st.subheader("Diameter distribution")
    st.caption("Physical crater size, from each source's own metadata - not a pixel feature, kept "
              "separate. Julie's set has none (no lat/lon/diam metadata for it, see the Sources "
              "table's note).")
    _render_feature("diam_km", show_view_toggle=False)

st.divider()
with st.container(border=True):
    st.subheader("Pixel feature distribution by source")
    pixel_feats = ["brightness", "sobel_mean", "frac_saturated"]
    feat = st.selectbox("Feature", pixel_feats, format_func=lambda c: FEATURE_LABELS.get(c, c))
    _render_feature(feat)

# --- Cross-source significance (omnibus) --------------------------------------
st.divider()
with st.container(border=True):
    st.subheader("Does this feature differ across sources? (Kruskal-Wallis)")
    st.caption(
        "Omnibus test across all sources at once - a significant, large-effect result means "
        "at least one source's distribution differs, not which one. See the pairwise tests below "
        "for that.")
    cst = profile["cross_source_test"]
    cst_df = pd.DataFrame(cst).T.rename_axis("feature").reset_index()
    cst_df = cst_df.rename(columns={
        "h_stat": "H statistic", "p_value": "p-value", "eta_sq": "effect size (eta-sq)", "n": "n"})
    st.dataframe(cst_df, width="stretch", hide_index=True)

# --- Pairwise tests ------------------------------------------------------------
st.divider()
with st.container(border=True):
    st.subheader("Pairwise comparisons")
    st.caption(
        "Dunn's post-hoc (Holm-corrected) tells you which specific pairs of sources differ in "
        "location, following up the omnibus test above. KS tests whether the whole distribution "
        "shape differs (not just the median); Levene tests whether the variance differs - either "
        "can flag a pair that Kruskal-Wallis/Dunn's under-report if the medians happen to be "
        "close but the spread or shape isn't (e.g. Julie's set: similar mean brightness to "
        "training, about half the standard deviation).")
    pw_feat = st.selectbox("Feature ", feature_cols, format_func=lambda c: FEATURE_LABELS.get(c, c),
                           key="pairwise_feature")
    pw = profile["pairwise_tests"].get(pw_feat, {})
    dunn = pw.get("dunn", {})
    ks_lev = pw.get("ks_levene", {})
    pair_rows = []
    for key in dunn:
        a, b = key.split("__vs__")
        d, k = dunn[key], ks_lev.get(key, {})
        pair_rows.append({
            "source A": a, "source B": b,
            "dunn z": d["z"], "dunn p (Holm)": d["p_holm"],
            "KS stat": k.get("ks_stat"), "KS p": k.get("ks_p"),
            "Levene stat": k.get("levene_stat"), "Levene p": k.get("levene_p"),
        })
    pair_df = pd.DataFrame(pair_rows).sort_values("dunn p (Holm)")

    _FLAG_BG = "background-color: rgba(211, 59, 59, 0.35)"

    def _flag_significant(row):
        flagged = row["dunn p (Holm)"] is not None and row["dunn p (Holm)"] < 0.05
        return [_FLAG_BG if flagged else "" for _ in row]

    st.dataframe(
        pair_df.style.apply(_flag_significant, axis=1).format(
            {c: "{:.4f}" for c in pair_df.columns if c not in ("source A", "source B")}),
        width="stretch", hide_index=True)

# --- Geo/diameter correlation ---------------------------------------------------
st.divider()
with st.container(border=True):
    st.subheader("Correlation with lat / lon / diameter (within a source)")
    st.caption(
        "Spearman rank correlation of each pixel feature against that source's own lat/lon/"
        "diameter - a real signal here (e.g. brightness varying smoothly with latitude) would "
        "mean the crop pipeline itself introduces a geographic or size-dependent artifact, "
        "independent of any model. Julie's set has no lat/lon/diameter metadata, so it's absent "
        "here.")
    geo_feat = st.selectbox("Feature  ", feature_cols, format_func=lambda c: FEATURE_LABELS.get(c, c),
                            key="geo_feature")
    geo_rows = []
    for name, s in sources.items():
        gc = s.get("geo_correlation", {})
        for geo in ["lat", "lon", "diam_km"]:
            key = f"{geo_feat}_vs_{geo}"
            if key not in gc or geo_feat == geo:
                continue
            v = gc[key]
            geo_rows.append({"source": name, "vs": geo, "r": v["r"], "p_value": v["p_value"], "n": v["n"]})
    if geo_rows:
        geo_df = pd.DataFrame(geo_rows)

        def _flag_geo(row):
            flagged = row["p_value"] is not None and row["p_value"] < 0.05 and abs(row["r"] or 0) >= 0.3
            return [_FLAG_BG if flagged else "" for _ in row]

        st.dataframe(
            geo_df.style.apply(_flag_geo, axis=1).format({"r": "{:.3f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)
    else:
        st.info("No source has lat/lon/diameter metadata for this feature.")
