"""Methods: exactly how every number in this dashboard is computed.

All content lives in src/dashboard/methodology.py, so each metric is described
in one place and the same text is reused by the inline "How this is
calculated" expanders on the other pages - a description cannot drift from the
page that shows the number.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from dashboard import methodology as M

st.set_page_config(page_title="Methods", layout="wide")
st.title("Methods")
st.caption(
    "Every metric, with the sample it is computed on, what it is grouped or correlated "
    "against, and which statistic is used. The same entries appear as inline "
    "\"How this is calculated\" expanders next to the numbers themselves.")

st.warning(
    "**Several numbers in this project are all loosely called \"the longitude effect\", "
    "and they are not the same quantity.** They differ in sample (642 labeled craters vs "
    "100,000), in grouping (k-means clusters vs three longitude zones vs a continuous "
    "principal component), and in statistic (epsilon-squared vs Spearman ρ vs a relative "
    "difference in means). Reading one as another has already produced incorrect "
    "statements in this project's own notes. The first three entries below are the ones "
    "most often confused.")

# Grouped so the easily-confused ones sit together and can be read against
# each other, rather than alphabetically.
GROUPS = [
    ("The three \"geography\" measurements — read these together",
     ["confound_correlation", "seam_zone_comparison", "latent_geo_correlation"]),
    ("Representation quality",
     ["silhouette", "map", "clustering_agreement", "alignment"]),
    ("Supervised metrics and their caveats",
     ["qwk_ordinal", "ece", "bootstrap_ci", "supervised_probe_caveat"]),
    ("How to read separation against confounds",
     ["separation_tradeoff"]),
    ("The data pipeline",
     ["sources", "highpass", "normalizations", "catalog_storage"]),
    ("Data-side measurements",
     ["clip_by_class", "intensity_features", "terrain_shortcut"]),
]

with st.container(border=True):
    st.subheader("Contents")
    for title, keys in GROUPS:
        st.markdown(f"**{title}**")
        st.markdown("\n".join(
            f"- {M.METHODS[k]['title']}" for k in keys if k in M.METHODS))

st.divider()

for title, keys in GROUPS:
    st.header(title)
    for k in keys:
        m = M.METHODS.get(k)
        if m is None:
            continue
        with st.container(border=True):
            st.subheader(m["title"])
            st.caption(f"Source: `{m['source']}`")
            st.markdown(M.text(k))
    st.divider()

st.caption(
    f"Effect-size convention used throughout (Kruskal-Wallis epsilon-squared): "
    f"{M.EFFECT_SIZE_BARS}.")
