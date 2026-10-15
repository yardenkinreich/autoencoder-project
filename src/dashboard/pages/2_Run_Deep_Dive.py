"""Single-run deep dive: browse one checkpoint's full results. Checkpoints
come from the full registry (every trained model under logs/, evaluated or
not), not just ones with an eval_results/ entry - nothing here is
recomputed, it only reads what evaluate.py/train.py already wrote."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import streamlit as st

from dashboard import data as D
from dashboard import model_meta as MM
from dashboard import methodology as M

st.set_page_config(page_title="Run Deep Dive", layout="wide")
st.title("Run Deep Dive")

def _cluster_exports_and_profile(entry, key_prefix: str, k_sel: int,
                                 heading_suffix: str = "") -> None:
    """Downloads + per-cluster profile for one geographic map at one k.

    Shared by the per-checkpoint map section and the data-comparison section
    so the two cannot present the same artifacts differently."""
    import json as _json
    c1, c2 = st.columns(2)
    gp, cs = entry.get("gpkg"), entry.get("clusters_csv")
    if gp and os.path.exists(gp):
        with open(gp, "rb") as f:
            c1.download_button(
                f"⬇ GeoPackage for QGIS (k={k_sel}){heading_suffix}",
                data=f.read(), file_name=os.path.basename(gp),
                mime="application/geopackage+sqlite3",
                key=f"{key_prefix}_gpkg")
        note = ("Single file, CRS embedded, already in the mosaic's own projection — "
                "drop it on the WAC mosaic and it lands correctly with no "
                "reprojection.")
        if entry.get("qml"):
            note += (" Put the matching `.qml` beside it (download below) and QGIS "
                     "applies **the same per-cluster colours** you see here.")
        c1.caption(note)
    else:
        c1.caption("No GeoPackage for this k — run "
                   "`python -m src.eval.backfill_geojson --scan logs`.")
    qml = entry.get("qml")
    if qml and os.path.exists(qml):
        with open(qml, "rb") as f:
            c2.download_button(f"⬇ QGIS style (.qml, k={k_sel}){heading_suffix}",
                               data=f.read(), file_name=os.path.basename(qml),
                               mime="application/xml", key=f"{key_prefix}_qml")
        c2.caption("Categorized renderer on `cluster`. Save it next to the .gpkg with "
                   "the same basename and QGIS loads it automatically.")
    if cs and os.path.exists(cs):
        with open(cs, "rb") as f:
            c2.download_button(f"⬇ Per-crater CSV (k={k_sel}){heading_suffix}",
                               data=f.read(), file_name=os.path.basename(cs),
                               mime="text/csv", key=f"{key_prefix}_csv")

    # ── per-cluster profile ─────────────────────────────────────────────────
    prof_csv = entry.get("profile_csv")
    if not prof_csv or not os.path.exists(prof_csv):
        return
    prof = pd.read_csv(prof_csv)
    ordinal = {}
    if entry.get("ordinal_json") and os.path.exists(entry["ordinal_json"]):
        ordinal = _json.load(open(entry["ordinal_json"]))

    st.markdown("**What is each cluster?**")
    if entry.get("pca_kmeans_png") or entry.get("umap_kmeans_png"):
        st.markdown(f"**All pool craters in the embedding, coloured by these k={k_sel} clusters**")
        pc1, pc2 = st.columns(2)
        if entry.get("pca_kmeans_png"):
            pc1.image(entry["pca_kmeans_png"], width="stretch")
        if entry.get("umap_kmeans_png"):
            pc2.image(entry["umap_kmeans_png"], width="stretch")
    if entry.get("mean_craters_png"):
        st.image(entry["mean_craters_png"], width="stretch")
    if ordinal:
        v1 = ordinal.get("verdict_linearity")
        v2 = ordinal.get("verdict_agreement")
        pc1 = ordinal.get("pc1_explained_variance")
        rho = ordinal.get("spearman_brightness_vs_sobel")
        st.info(
            f"**Is there a real ordinal?** Clusters are laid out by "
            f"`{ordinal.get('ordered_by', '?')}` — an *imposed* view, computed from "
            f"pixel statistics; cluster ids and colours are untouched by it.\n\n"
            + (f"- **Linearity:** PC1 of the cluster means explains "
               f"**{pc1:.1%}** of between-cluster variance — {v1}\n"
               if pc1 is not None else "")
            + (f"- **Agreement:** ρ(brightness, texture) = **{rho:+.3f}** — {v2}\n"
               if rho is not None else "")
            + "\nNeither test says the axis means *degradation*: both are pixel "
              "statistics this project has repeatedly found confounded with "
              "acquisition geometry. Read `lon_ks_vs_pool` in the table alongside — "
              "a clean 1-D axis that turns out to be longitude is a warning, not a "
              "result.")
    if entry.get("feature_profile_png"):
        st.image(entry["feature_profile_png"], width="stretch")

    # ── per-band clustering ─────────────────────────────────────────────────
    pb = entry.get("per_band_json")
    # Gated behind a checkbox rather than an expander, deliberately: Streamlit
    # EXECUTES an expander's body and ships its images even while the expander
    # is collapsed, so wrapping this in one would look like lazy loading
    # without being it. Only skipping the code actually skips the work - and
    # this section is the heaviest on the page (3 band tabs x profile + PCA +
    # map, plus the two comparison figures).
    if pb and os.path.exists(pb):
        st.markdown("**Clustering within each acquisition band**")
        _pb_on = st.checkbox(
            "Load the per-band analysis", key=f"{key_prefix}_pbopen",
            help="Off by default — this section renders a dozen large figures "
                 "and is what makes the page slow to load. Nothing here is "
                 "computed on the fly; it is all read from disk when ticked.")
    else:
        _pb_on = False
    if _pb_on:
        j = _json.load(open(pb))
        st.caption(
            "The WAC mosaic is assembled from three observation windows, and the three "
            "longitude bands differ in illumination geometry — measured, diameter-"
            "matched, against the middle band: contrast west 1.23 / mid 1.00 / east "
            "0.82 (solar elevation), and the left-right asymmetry **flips sign** at "
            "−90° (solar azimuth). Neither the subtractive high-pass nor a divisive "
            "local-contrast normalization removes that; both were built and measured. "
            "So instead: cluster *inside* each band, where illumination is constant and "
            "band identity is therefore unavailable as a feature, then match clusters "
            "across bands by their mean-crater profile.")
        c1, c2, c3 = st.columns(3)
        c1.metric("one global clustering", f"{j.get('v_global', float('nan')):.4f}")
        c2.metric("per-band, matched", f"{j.get('v_per_band_matched', float('nan')):.4f}",
                  delta=f"{100*(j['v_per_band_matched']/j['v_global']-1):+.0f}%"
                  if j.get('v_global') else None, delta_color="inverse")
        mir = j.get("mirror_used") or {}
        c3.metric("bands mirrored to match",
                  ", ".join(k for k, v in mir.items() if v) or "none")
        st.caption(
            "Cramér's V between the cluster assignment and the acquisition band — "
            "**lower is better**, 0 would mean the band is no longer recoverable from "
            "the clustering. A band listed as *mirrored* only matched once its mean "
            "crater images were flipped left-right, which is independent confirmation "
            "that its illumination azimuth is reversed.\n\n"
            "This measures only that band is less predictable. It does **not** show the "
            "clusters are degradation-meaningful — that needs the labelled set, where "
            "the west band has just 30 craters.")

        # These figures are matplotlib grids with k rows, so at full container
        # width every panel is blown up past the point of being readable - the
        # eye has to scroll to compare rows that the figure was laid out to be
        # compared side by side. One control for the whole section rather than
        # per figure, because the comparison being made is ACROSS them.
        pb_w = st.slider(
            "Figure width for this section (px)", min_value=500, max_value=1600,
            value=900, step=50, key=f"{key_prefix}_pbw",
            help="Applies to every figure in the per-band section. Lower it "
                 "until whole grids fit on screen — these are meant to be read "
                 "as a grid, not zoomed into one panel at a time.")
        # Order follows the question: what is each cluster, in each band ->
        # do the bands correspond -> where are they on the Moon.
        if entry.get("per_band_matches_png"):
            st.markdown("**The matches themselves**")
            st.image(entry["per_band_matches_png"], width=pb_w)
            st.caption(
                "Each row is one matched group; the three panels are that group's mean "
                "crater in each band. **If the matching is real, a row shows the same "
                "kind of crater three times.** A band flagged *mirrored* only matched "
                "after its images were flipped left-right — look for the west panels "
                "being mirror images of the others, which is the illumination-azimuth "
                "reversal made visible.")
        if entry.get("per_band_examples_png"):
            st.markdown("**Example craters per group, per band**")
            st.image(entry["per_band_examples_png"], width=pb_w)
            st.caption(
                "Three real members per group per band. A mean image can look tidy "
                "while its members are heterogeneous, so this is the check the mean "
                "cannot provide.")

        if any(entry.get(f"per_band_profile_{b}") for b in ("west", "mid", "east")):
            st.markdown("**Each band's own clusters — average then real members**")
            st.caption(
                "One tab per acquisition band, BEFORE any cross-band merging. Rows are "
                "ordered by texture, the axis these clusters actually separate on "
                "(4.2× sobel range against 1.05× brightness). A band that tracks "
                "degradation should read as a smooth run from washed-out to "
                "sharp-rimmed; one that does not will look shuffled.\n\n"
                "Example craters are the members **nearest that cluster's centroid** — "
                "the centroid being the mean embedding k-means actually minimised "
                "distance to, not the mean image.")
            # Tabs are right here: the section gate above already defers the
            # cost, and west/mid/east are meant to be flipped between quickly
            # to compare - which tabs do client-side, with no rerun.
            tabs = st.tabs(["west", "mid", "east"])
            for t, b in zip(tabs, ("west", "mid", "east")):
                with t:
                    if entry.get(f"per_band_profile_{b}"):
                        st.image(entry[f"per_band_profile_{b}"], width=pb_w)
                    # The PCA sits directly under the clusters it describes:
                    # the rows above say what each cluster looks like, this
                    # says whether they are actually apart in the embedding.
                    if entry.get(f"per_band_ownbasis_{b}"):
                        st.markdown(f"**Latent separation — {b} band, its own PCA**")
                        st.image(entry[f"per_band_ownbasis_{b}"], width=pb_w)
                        st.caption(
                            f"PCA fitted on the {b} band alone and coloured by the "
                            f"clusters shown above. **Silhouette is printed in the "
                            "title and is computed in the full 384-dim embedding, "
                            "not in these 2 PCs** — a 2-D projection can smear "
                            "apart clusters together, so the number must come from "
                            "the space the clustering happened in. Values near 0 "
                            "mean the clusters overlap almost completely, which is "
                            "the honest range for this data.")
                    if entry.get(f"per_band_map_{b}"):
                        st.image(entry[f"per_band_map_{b}"], width=pb_w)
                        st.caption(
                            f"Geographic spread of the {b} band's own clusters. "
                            "Colours are this band's ids and do not correspond "
                            "to the other bands until matched.")

        with st.expander("All three bands side by side — shared basis vs own basis"):
            if entry.get("per_band_latent_png"):
                st.markdown("**One shared PCA basis**")
                st.image(entry["per_band_latent_png"], width=pb_w)
                st.caption(
                    "One PCA basis fitted on all embeddings, so the four panels sit "
                    "in the same coordinates and are directly comparable. First three "
                    "panels are each band's **own** clusters; the last is after "
                    "matching.")
            if entry.get("per_band_ownbasis_png"):
                st.markdown("**A separate PCA per band**")
                st.image(entry["per_band_ownbasis_png"], width=pb_w)
                st.caption(
                    "Axes are deliberately **not** comparable between panels — this "
                    "shows each band's intrinsic structure rather than its position "
                    "in a shared frame.\n\n"
                    "**The decisive number is not either picture but the principal "
                    "angles between the bands' top-10 subspaces**, printed in the run "
                    "log and stored in the JSON: measured at 7–15° out of 90°, the "
                    "three bands span essentially the *same* directions. So separating "
                    "them does not give each its own representation — it only moves "
                    "the cluster boundaries inside one shared space, which is why "
                    "per-band silhouette (0.083–0.093) matches the global one (0.093).")

        st.markdown("**Geographic spread, full mosaic**")
        m1, m2 = st.columns(2)
        if entry.get("per_band_map_global"):
            m1.image(entry["per_band_map_global"], width="stretch")
            m1.caption("ONE clustering over all bands — the current approach.")
        if entry.get("per_band_map_matched"):
            m2.image(entry["per_band_map_matched"], width="stretch")
            m2.caption("Clustered within each band, then matched across bands. "
                       "If the acquisition bands have stopped driving the clustering, "
                       "the right-hand band should no longer be its own patch of "
                       "colour.")
        st.markdown("**QGIS downloads**")
        st.caption(
            "Every version as its own GeoPackage, all in the mosaic's projected CRS so "
            "they overlay it without reprojection. The per-band layers carry that "
            "band's **own** cluster ids, which do not correspond between bands — the "
            "shared `.qml` is only meaningful within one band, or on the matched layer.")
        dl = [("per_band_gpkg", "band-matched (all craters)"),
              ("per_band_gpkg_global", "one global clustering"),
              ("per_band_gpkg_west", "west band only"),
              ("per_band_gpkg_mid", "mid band only"),
              ("per_band_gpkg_east", "east band only"),
              ("per_band_qml", "QGIS style (.qml)"),
              ("per_band_csv", "per-crater CSV (all labelings)")]
        cols = st.columns(4)
        for i, (key, lbl) in enumerate(dl):
            f = entry.get(key)
            if f and os.path.exists(f):
                with open(f, "rb") as fh:
                    cols[i % 4].download_button(
                        f"⬇ {lbl}", data=fh.read(),
                        file_name=os.path.basename(f),
                        key=f"{key_prefix}_{key}")

    show = [c for c in ["cluster", "order_rank", "n", "pct_of_pool",
                        "brightness_mean", "sobel_mean", "diam_km_mean",
                        "spatial_spread_vs_pool", "lon_ks_vs_pool", "color_hex"]
            if c in prof.columns]
    with st.expander(f"Per-cluster table (k={k_sel})"):
        st.dataframe(
            prof.sort_values("order_rank" if "order_rank" in prof else "cluster")[show]
                .style.format({c: "{:.4f}" for c in show
                               if c not in ("cluster", "order_rank", "n", "color_hex")}),
            width="stretch", hide_index=True)
        st.caption(
            "`order_rank` is the imposed ordering; `lon_ks_vs_pool` is how unlike the "
            "whole pool this cluster's longitude distribution is (high = the cluster "
            "is a longitude artifact); `spatial_spread_vs_pool` < 1 means more "
            "localized than the pool. `color_hex` is the colour used here, in the map "
            "and in QGIS.")


registry = D.build_registry()
if registry.empty:
    st.info("No trained checkpoints found under logs/, and logs/eval_history.csv is empty.")
    st.stop()

registry = D.enrich_registry_with_params(registry)
registry = registry.sort_values(["has_eval", "checkpoint"], ascending=[False, True]).reset_index(drop=True)

# param_source (provenance/confidence of the extracted training params, not
# to be confused with "scratch/finetune/pretrained") - color by how much to
# trust it.
PROVENANCE_COLOR = {
    "run_manifest": "green", "dino_config": "green", "snakefile_snapshot": "blue",
    "dirname_regex": "orange", "stock": "gray", "unknown": "gray",
}

FAMILY_COLOR = {"mae": "blue", "cae": "violet", "dino": "orange"}
SOURCE_COLOR = {"scratch": "gray", "finetune": "green", "stock_pretrained": "blue", "unknown": "gray"}

_FLAG_BG = "background-color: rgba(211, 59, 59, 0.35)"


def _flag_confound_rows(row):
    # NaN-safe: comparisons against a missing p_value/eta_sq are always False.
    flagged = row["p_value"] < 0.05 and row["eta_sq"] >= 0.14
    return [_FLAG_BG if flagged else "" for _ in row]


def _flag_geo_rows(row):
    # |r| >= 0.5 matches eta_sq>=0.14's "large effect" spirit (Cohen's
    # thresholds for correlation vs. ANOVA-style effect size respectively) -
    # same bar for how alarming a flag is, even though the two tables test
    # different things (discrete-group difference vs. continuous correlation).
    # NaN-safe: comparisons against a missing p_value/r are always False.
    flagged = row["p_value"] < 0.05 and abs(row["r"]) >= 0.5
    return [_FLAG_BG if flagged else "" for _ in row]


def _label(i: int) -> str:
    row = registry.iloc[i]
    if row["training_run_dir"]:
        # Canonical runs live at logs/{family}/{source}/{structure}/{frozen}/
        # {data_tag}/{d}/{n}/{other}/{epN}[/...] (run_layout.py), and many share
        # the last folder (e.g. four Kaguya finetunes are all ".../ep30") - so
        # label by data_tag / other-metric (objective, mask, latent) / epochs.
        parts = os.path.normpath(row["training_run_dir"]).split(os.sep)
        if len(parts) >= 10 and parts[0] == "logs":
            tag = "/".join([row["architecture"] or parts[1], parts[5]]
                           + ([parts[8]] if parts[8] != "na" else []) + parts[9:])
        else:
            tag = f"{row['architecture'] or '?'}/{os.path.basename(row['training_run_dir'].rstrip(os.sep))}"
    else:
        tag = row["run_id"] or os.path.basename(row["checkpoint"])
    bits = [b for b in [row.get("param_source"),
                        f"ep{row['param_epochs']}" if row.get("param_epochs") is not None else None] if b]
    if bits:
        tag = f"{tag}  [{' / '.join(str(b) for b in bits)}]"
    return f"{tag}  {'✓ evaluated' if row['has_eval'] else '(not yet evaluated)'}"


# --- Searchable, filterable picker ------------------------------------------
col_search, col_unevaluated = st.columns([3, 1])
search = col_search.text_input(
    "Filter checkpoints", placeholder="family, run dir, data tag, ...")
only_unevaluated = col_unevaluated.checkbox("Only unevaluated", value=False)

filtered = registry
if only_unevaluated:
    filtered = filtered[~filtered["has_eval"]]
if search:
    s = search.lower()

    def _matches(row) -> bool:
        haystack = " ".join(str(v) for v in [
            row.get("architecture"), row.get("training_run_dir"), row.get("param_data_tag"),
        ] if v is not None).lower()
        return s in haystack

    filtered = filtered[filtered.apply(_matches, axis=1)]

if filtered.empty:
    st.warning("No checkpoints match this filter.")
    st.stop()
filtered_idx = filtered.index

if "deep_dive_idx" not in st.session_state or st.session_state.deep_dive_idx not in filtered_idx:
    st.session_state.deep_dive_idx = filtered_idx[0]


def _step(delta: int) -> None:
    # Must run as an on_click callback, not inline in the script body -
    # Streamlit forbids writing to a widget-bound session_state key (here,
    # the "Checkpoint" selectbox's "deep_dive_idx") after that widget has
    # been instantiated; callbacks run before the next rerun's script body,
    # which is the one place that write is still allowed.
    pos = filtered_idx.get_loc(st.session_state.deep_dive_idx)
    st.session_state.deep_dive_idx = filtered_idx[(pos + delta) % len(filtered_idx)]


nav_prev, nav_pick, nav_next = st.columns([1, 8, 1])
nav_prev.button("◀", help="Previous in filtered set", on_click=_step, args=(-1,))
idx = nav_pick.selectbox("Checkpoint", filtered_idx, format_func=_label, key="deep_dive_idx")
nav_next.button("▶", help="Next in filtered set", on_click=_step, args=(1,))

row = registry.iloc[idx]

st.subheader(_label(idx))
st.caption(f"checkpoint: `{row['checkpoint']}`")

params_arch = row["architecture"]
if row["has_eval"]:
    summary = D.load_run_summary(row["eval_run_dir"])
    params_arch = summary["autoencoder_model"]  # authoritative (e.g. distinguishes *_pretrained)
else:
    summary = None

params = MM.extract_model_params(row["checkpoint"], params_arch)

# --- Gaps checklist: what's available for this run, at a glance ------------
gaps = {"trained checkpoint": True, "evaluated": bool(row["has_eval"])}
loss_curve_available = False
if row["training_run_dir"]:
    artifacts = D.training_artifacts(row["training_run_dir"])
    loss_curve_available = bool(artifacts["loss_curve_png"])
    gaps["loss curve"] = loss_curve_available
labeled_sets = (summary or {}).get("labeled_sets", {})
# Primary-source (Kaguya) labeled results FIRST - the first tab is the default.
# A checkpoint evaluated before the Kaguya sets joined eval_suite.yaml (e.g.
# the stock DINOv2) still has them in its data-comparison section, evaluated on
# the full baseline labels (configs/liorr_labels.csv); lift those into the tabs.
set_dirs = {n: row["eval_run_dir"] for n in labeled_sets}
_primary = {}
if row["eval_run_dir"]:
    _dc = D.data_comparison_artifacts(row["eval_run_dir"])
    for (_cat, _norm), _c in sorted(_dc.get("cells", {}).items(),
                                    key=lambda kv: ("ratio" not in kv[0][0], kv[0])):
        if D.is_primary_variant(_cat) and _c["set_name"] not in labeled_sets:
            _primary[_c["set_name"]] = _c["summary"]
            set_dirs[_c["set_name"]] = os.path.join(_dc["root"], "eval", _cat)
labeled_sets = {**_primary,
                **dict(D.primary_first(labeled_sets.items(), key=lambda kv: kv[0]))}


def _set_label(name: str) -> str:
    """Readable tab title for a labeled set."""
    if name.startswith("kaguya_tco_"):
        src, _, norm = name[len("kaguya_tco_"):].partition("__")
        n = (labeled_sets.get(name) or {}).get("n_samples") or ""
        part = "final_test half" if norm.endswith("_final_test") else "liorr labels"
        return f"Kaguya {src} · {part}{f' (n={n})' if n else ''}"
    return name


eval_artifacts_by_set = {}
for set_name in labeled_sets:
    ea = D.load_labeled_set_artifacts(set_dirs[set_name], set_name)
    eval_artifacts_by_set[set_name] = ea
    gaps[f"{set_name}: plots"] = all(ea[k] for k in (
        "confusion_matrix_png", "clusters_by_sample_png",
        "latent_separation_png", "agreement_examples_png"))
    gaps[f"{set_name}: tables"] = ea["per_class"] is not None and ea["confusion"] is not None
if row["has_eval"]:
    gaps["holdout"] = bool((summary or {}).get("holdout"))

st.markdown(" ".join(
    f":{'green' if ok else 'gray'}-badge[{'✓' if ok else '○'} {label}]"
    for label, ok in gaps.items()
))

# --- Params band -------------------------------------------------------------
st.badge(f"params source: {params['param_source']}",
        color=PROVENANCE_COLOR.get(params["param_source"], "gray"))
with st.container(border=True):
    st.caption("Training parameters")
    cols = st.columns(6)
    cols[0].caption("family")
    cols[0].badge(params["family"] or "—", color=FAMILY_COLOR.get(params["family"], "gray"))
    cols[1].caption("source")
    cols[1].badge(params["source"], color=SOURCE_COLOR.get(params["source"], "gray"))
    epochs = params["epochs"]
    if epochs is not None and params.get("iterations") is not None:
        # DINO: "epochs" isn't a literal dataset pass, and OFFICIAL_EPOCH_LENGTH
        # (iterations per "epoch") varies run to run - two runs both saying
        # "epochs: 2" can mean very different training volume. Show both.
        cols[2].metric("epochs (iters)", f"{epochs:g}", f"{params['iterations']:,} it",
                      delta_color="off")
    else:
        cols[2].metric("epochs", epochs if epochs is not None else "—")
    cols[3].metric("mask_ratio", params["mask_ratio"] if params["mask_ratio"] is not None else "—")
    cols[4].metric("num_samples", params["num_samples"] if params["num_samples"] is not None else "—")
    cols[5].metric("diameter_range", str(params["diameter_range"]) if params["diameter_range"] else "—")

    extra = [
        ("pretrained_weights", params["pretrained_weights"]),
        ("data_tag", params["data_tag"]),
        # crop radius as a multiple of crater diameter - MAE/CAE (0.5) and
        # DINO (1.0) can share a data_tag (same underlying mosaic/sigma
        # source) while training on differently-framed crops, so this is
        # what actually distinguishes them.
        ("fov", params["fov"]),
        ("freeze_until", params["freeze_until"]),
        ("proto_loss_enabled", params["proto_loss_enabled"] or None),
        ("proto_loss_weight", params["proto_loss_weight"]),
        ("proto_n_support_per_class", params["proto_n_support_per_class"]),
        ("proto_split_csv", params["proto_split_csv"]),
        ("augmentation_recipe", params["augmentation_recipe"]),
        ("training_method", params["training_method"]),
        ("diet_cp_unfreeze_last_n_blocks", params["diet_cp_unfreeze_last_n_blocks"]),
        ("diet_cp_freeze_epochs", params["diet_cp_freeze_epochs"]),
    ]
    extra = [(k, v) for k, v in extra if v is not None]
    if extra:
        with st.expander("More details"):
            for k, v in extra:
                st.caption(f"{k}: `{v}`")

# --- Training artifacts (available even without an eval run) --------------
loss_component_pngs = artifacts.get("loss_component_pngs", {}) if row["training_run_dir"] else {}
if row["training_run_dir"] and (loss_curve_available or loss_component_pngs
                                or artifacts["legacy_result_images"]):
    st.divider()
    st.subheader("Training")
    if loss_curve_available:
        st.image(artifacts["loss_curve_png"], caption="Loss curve (total)", width=600)
    if loss_component_pngs:
        # DINO only - each loss TERM's own trend (very different scales, e.g.
        # koleo_loss hovers near 0 while dino_local_crops_loss is ~7 - see
        # train_dino.py's plot_dino_loss() docstring for why these are kept
        # separate from the total/from each other, and from Compare Runs'
        # cross-run view, which only ever shows the total).
        st.caption("Loss components")
        cols = st.columns(min(3, len(loss_component_pngs)))
        for i, (name, img) in enumerate(loss_component_pngs.items()):
            cols[i % len(cols)].image(img, caption=name, width='stretch')
    if artifacts["legacy_result_images"]:
        st.caption("Legacy clustering plots (pre-eval-harness Snakefile rules)")
        cols = st.columns(min(4, len(artifacts["legacy_result_images"])))
        for i, img in enumerate(artifacts["legacy_result_images"]):
            cols[i % len(cols)].image(img, caption=os.path.basename(img), width='stretch')

# --- Eval results -----------------------------------------------------------
if not row["has_eval"]:
    st.divider()
    st.info("No eval run yet for this checkpoint.")
    st.code(
        f"PYTHONPATH=src python -m eval.evaluate --checkpoint {row['checkpoint']} "
        f"--autoencoder-model {params_arch or '<mae|cae|dino|mae_pretrained|dino_pretrained>'}",
        language="bash",
    )
    st.stop()

if labeled_sets:
    st.divider()
    st.subheader("Eval results")
    tabs = st.tabs([_set_label(n) for n in labeled_sets])
    for tab, (set_name, results) in zip(tabs, labeled_sets.items()):
        with tab:
            m1, m2, m3, m4, m5 = st.columns(5)
            for col, key, label in [
                (m1, "accuracy", "Accuracy"), (m2, "macro_f1", "Macro F1"),
                (m3, "quadratic_weighted_kappa", "QWK"), (m4, "ordinal_mae", "Ordinal MAE"),
                (m5, "map", "mAP"),
            ]:
                v = results.get(key)
                if v:
                    col.metric(label, f"{v['point']:.3f}",
                              f"95% CI [{v['lo']:.3f}, {v['hi']:.3f}]", delta_color="off")

            m6, m7, m8 = st.columns(3)
            if results.get("ece") is not None:
                m6.metric("ECE", f"{results['ece']:.3f}")
            agree = results.get("agreement", {})
            m7.metric("ARI / NMI / V-measure",
                     f"{agree.get('ari', 0):.2f} / {agree.get('nmi', 0):.2f} / {agree.get('v_measure', 0):.2f}")
            m8.metric("n_samples", results.get("n_samples"))
            st.caption("The numbers above are **unsupervised**: k-means with k = #classes, "
                       "clusters Hungarian-aligned to the classes - no labels used.")

            eval_artifacts = eval_artifacts_by_set[set_name]
            hc = eval_artifacts.get("head_comparison")
            if hc is not None and len(hc):
                st.markdown("**k-NN and linear probe on this set** (5-fold CV - these DO use "
                            "the labels; the ceiling the embedding allows)")
                names = {"raw_cv": "k-means (CV folds)", "lda_head_cv": "LDA linear probe",
                         "knn_cv": "k-NN (k=5)"}
                tb = hc.copy()
                tb.index = [names.get(i, i) for i in tb.index]
                show = {}
                for m, lab in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
                               ("quadratic_weighted_kappa", "QWK"), ("ordinal_mae", "Ordinal MAE")]:
                    if m in tb:
                        show[lab] = [f"{p:.3f}  [{lo:.3f}, {hi:.3f}]" for p, lo, hi in
                                     zip(tb[m], tb.get(f"{m}_lo", tb[m]), tb.get(f"{m}_hi", tb[m]))]
                st.dataframe(pd.DataFrame(show, index=tb.index), width="stretch")
            plot_specs = [
                ("confusion_matrix_png", "Confusion matrix"),
                ("clusters_by_sample_png", "Clusters by sample"),
                ("latent_separation_png", "Latent separation (PCA)"),
                ("latent_separation_umap_png", "Latent separation (UMAP)"),
                ("agreement_examples_png", "Agreement examples"),
            ]
            available_plots = [(k, c) for k, c in plot_specs if eval_artifacts[k]]
            if available_plots:
                img_cols = st.columns(len(available_plots))
                for col, (key, caption) in zip(img_cols, available_plots):
                    col.image(eval_artifacts[key], caption=caption, width='stretch')

            if eval_artifacts["per_class"] is not None:
                st.markdown("**Per-class**")
                st.dataframe(eval_artifacts["per_class"])
            if eval_artifacts["confusion"] is not None:
                st.markdown("**Confusion matrix (rows=true, cols=pred)**")
                st.dataframe(eval_artifacts["confusion"])
            if eval_artifacts["confound_correlation"] is not None:
                st.markdown("**Confound check** (highlighted rows: p<0.05 and eta_sq>=0.14 — "
                            "the raw cluster grouping likely tracks this feature, not just degradation)")
                st.dataframe(eval_artifacts["confound_correlation"]
                            .style.apply(_flag_confound_rows, axis=1))
            if eval_artifacts["geo_correlation"] is not None:
                st.markdown("**Latent space vs. geography** (highlighted rows: p<0.05 and |r|>=0.5 — "
                            "the embedding itself correlates with lat/lon, Robbins-ID craters only)")
                st.dataframe(eval_artifacts["geo_correlation"]
                            .style.apply(_flag_geo_rows, axis=1))
            if eval_artifacts["reliability"] is not None:
                st.markdown("**Reliability bins**")
                st.dataframe(eval_artifacts["reliability"])

            unsupervised_pngs = eval_artifacts.get("unsupervised_cluster_pngs", {})
            if unsupervised_pngs:
                with st.expander("Unsupervised over-clustering exploration"):
                    st.caption("KMeans at k far larger than this set's own classes, colored "
                              "purely by raw cluster id — no true labels involved anywhere "
                              "here (see visualize.py's plot_unsupervised_cluster_dots/"
                              "_images docstrings). For looking at finer latent structure, "
                              "not for validating against the known degradation classes.")
                    available_k = sorted({k for (k, _, _) in unsupervised_pngs})
                    k_sel = st.radio("k", available_k, horizontal=True,
                                     key=f"unsup_k_{set_name}")
                    dot_cols = [(tech, unsupervised_pngs.get((k_sel, tech, "dots")))
                               for tech in ("pca", "umap")]
                    dot_cols = [(t, p) for t, p in dot_cols if p]
                    if dot_cols:
                        cols = st.columns(len(dot_cols))
                        for col, (tech, p) in zip(cols, dot_cols):
                            col.image(p, caption=f"k={k_sel} clusters ({tech.upper()})",
                                     width='stretch')
                    img_cols = [(tech, unsupervised_pngs.get((k_sel, tech, "images")))
                               for tech in ("pca", "umap")]
                    img_cols = [(t, p) for t, p in img_cols if p]
                    if img_cols:
                        cols = st.columns(len(img_cols))
                        for col, (tech, p) in zip(cols, img_cols):
                            col.image(p, caption=f"k={k_sel} clusters, crater crops ({tech.upper()})",
                                     width='stretch')

            attention_pngs = eval_artifacts.get("attention_map_pngs", [])
            if attention_pngs:
                with st.expander("Attention maps (DINO only)"):
                    st.caption("Last transformer block's CLS-token attention, overlaid on a "
                              "handful of sample craters per class (see eval/attention_maps.py's "
                              "module docstring). Check whether attention stays on the crater "
                              "itself vs. tracking a nuisance feature like raw brightness — a "
                              "faster, visual read on shortcut learning than the PCA/confound "
                              "checks above, and an independent one: the two can disagree.")
                    cols = st.columns(min(3, len(attention_pngs)))
                    for i, p in enumerate(attention_pngs):
                        cols[i % len(cols)].image(p, caption=os.path.basename(p), width='stretch')
else:
    st.divider()
    st.info("This run has no labeled-set results.")

# --- Global geographic cluster map (eval/geographic_map.py) ----------------
# Outside the labeled_sets block on purpose: this one isn't scored against
# any labeled set - it clusters a ~100k-crater unlabeled pool and plots it
# at lunar coordinates, so it exists (or doesn't) per CHECKPOINT rather than
# per labeled set. Only generated for selected models, so absent for most.
geo_variants = D.all_geographic_variants(row["eval_run_dir"])
if geo_variants:
    st.divider()
    st.subheader("Geographic cluster maps & cluster profiles")
    st.caption(
        "~100k craters (1-12 km) embedded with this checkpoint, clustered with KMeans, "
        "and plotted at their lunar coordinates over the plain morphology mosaic — see "
        "`eval/geographic_map.py` for what this does and doesn't show. Read these for "
        "**spatial structure**: a coherent region of one colour is a localized cluster, "
        "and vertical banding is the mosaic longitude artifact.\n\n"
        "Cluster ids come straight from KMeans with no reordering, so colours carry no "
        "ranking and **do not correspond between variants or between models**. The "
        "basemap is always the plain morphology mosaic whatever product the model "
        "consumed, since a high-pass residual has no maria/highland contrast and is "
        "unreadable as a map.")

    names = list(geo_variants)
    # Legacy last: a catalog-derived map is the one to look at when both exist.
    names.sort(key=lambda n: (geo_variants[n].get("legacy", False), n))
    names = D.primary_first(names)          # Kaguya first = the default selection
    v_sel = st.selectbox("Data variant", names,
                         key=f"geovar_{row['checkpoint']}")
    gv = geo_variants[v_sel]
    if gv.get("legacy"):
        st.warning(
            "This map comes from the **older `preprocess_2` pipeline, which excluded a "
            "90° longitude band** around the mosaic seam — hence the gap. Catalog-derived "
            "maps (`data/catalogs/*`) cover the full longitude range, seam band included, "
            "because that band is itself under test. Don't read the two as equivalent.")

    # What "morph_sigma100_div500__global_wide" actually MEANS. The variant name
    # is the label on every figure and download below it, and it is opaque to
    # anyone who did not build the catalogs - so the mosaic, the filter and the
    # normalization are spelled out here, read from the catalog's own manifests
    # rather than from a lookup table that could drift away from the data.
    prov = D.variant_provenance(v_sel)
    if prov:
        with st.expander(f"What is `{v_sel}`?  —  source mosaic, filter, normalization"):
            if prov.get("source_name"):
                st.markdown(f"**Source · {prov['source_name']}**")
                st.caption(prov.get("source_note", ""))
            if prov.get("mosaic"):
                st.markdown(f"**Mosaic file**  \n`{prov['mosaic']}`")
            if prov.get("filter"):
                st.markdown(f"**Filter applied to the mosaic**  \n{prov['filter']}")
            if prov.get("norm_does"):
                extra = (f"  \nConstants fitted on this catalog: "
                         f"`{prov['norm_constants']}`") if prov.get("norm_constants") else ""
                st.markdown(f"**Normalization · `{prov['norm']}`**  \n"
                            f"{prov['norm_does']}{extra}")
                # How much a norm clips is the whole question, and the answer is
                # not "clips / does not clip" - `global` (1/99) removes 6-9% of
                # the FRESHEST craters' pixels and 0% of the most degraded, i.e.
                # damage correlated with the label, while `global_wide`
                # (0.01/99.99) removes 0.02% and was chosen precisely as the fix
                # for that. Warning about both identically would teach the reader
                # to distrust the wrong one.
                clipped = prov.get("clipped_pct")
                if clipped is not None and clipped >= 1.0:
                    st.warning(
                        f"**This variant clips ~{clipped:.1f}% of pixels.** The "
                        "p1/p99 window removes 6–9% of the *freshest* craters' "
                        "pixels and 0% of the most degraded — damage correlated "
                        "with the degradation label itself, which is what makes "
                        "it dangerous rather than merely lossy. Prefer "
                        "`global_wide` or `robust` here.")
                elif clipped is not None:
                    st.caption(
                        f"Clips ~{clipped:.2f}% of pixels — negligible, and "
                        "deliberately so. `global_wide` was chosen over both "
                        "`global` (which damages fresh craters specifically) and "
                        "`robust`: on the 30-cell labelled comparison it gave the "
                        "best class separation of any normalization (silhouette "
                        "0.131 vs 0.082 for `robust`). Note the catalog *loader* "
                        "default is `robust` — a separate decision, for embedding "
                        "rather than for this evaluation.")
            c1, c2, c3 = st.columns(3)
            if prov.get("n_craters"):
                c1.metric("craters", f"{prov['n_craters']:,}")
            if prov.get("diameter_km"):
                d = prov["diameter_km"]
                c2.metric("diameter", f"{d[0]:g}–{d[1]:g} km")
            if prov.get("lat_bounds"):
                lb = prov["lat_bounds"]
                c3.metric("latitude", f"{lb[0]:g}° to {lb[1]:g}°")
            if prov.get("raw_range"):
                lo, hi = prov["raw_range"]
                st.caption(f"Raw mosaic values in this catalog span "
                           f"`{lo:.4g} … {hi:.4g}` before normalization.")
            if prov.get("longitude"):
                st.caption(f"**Longitude:** {prov['longitude']}")
            if prov.get("crater_selection"):
                st.caption(f"**Crater selection:** {prov['crater_selection']}")
            if prov.get("built"):
                st.caption(f"Catalog built {prov['built']} · `{prov['catalog_dir']}`")

    available_k = sorted(gv["by_k"])
    k_sel = st.radio("k", available_k, horizontal=True,
                     key=f"geo_k_{row['checkpoint']}_{v_sel}")
    entry = gv["by_k"][k_sel]
    if entry.get("png"):
        st.image(entry["png"], width='stretch')
    if k_sel > 12:
        st.caption(
            f"At k={k_sel} the colours are arbitrary cluster ids and no legend is shown "
            "— 100 legend entries is not a legend. For per-cluster identity, download "
            "the GeoPackage below and style by `cluster` in QGIS.")

    _cluster_exports_and_profile(
        entry, f"geo_{row['checkpoint']}_{v_sel}_{k_sel}", k_sel)

    # ── pool labelled by something OTHER than KMeans ────────────────────────
    pl = D.pool_label_variants(gv.get("dir"))
    if pl:
        st.divider()
        st.subheader("k-NN classes on the full pool — map · PCA · UMAP")
        # Gated at the SECTION, not per figure: the point of a section toggle
        # is that opening it gives you everything in it. A checkbox rather than
        # an expander because Streamlit executes and ships an expander's body
        # even while it is collapsed, so an expander would not actually defer
        # the ~10 MB of figures below.
        _pl_on = st.checkbox(
            "Load this section", value=True, key=f"plopen_{row['checkpoint']}_{v_sel}",
            help="On by default; untick to skip loading the map, the per-class "
                 "examples and all three PCA views at once.")
    if pl and _pl_on:
        st.caption(
            "Everything above partitions the pool by **KMeans**, which gives "
            "arbitrary cluster ids with no relation to degradation. These maps "
            "instead propagate the **human labels** (the eval set's labelled craters - "
            "1,030 of configs/liorr_labels.csv for Kaguya) out to every pool crater, "
            "so the colours are degradation classes and carry an order "
            "(fresh → degraded), not arbitrary hues.\n\n"
            "The labelled craters are excluded from the pool, verified at run "
            "time — a non-empty overlap is a hard error, not a warning, since "
            "otherwise a crater could be its own nearest neighbour.")
        tags = sorted(pl)
        m_sel = st.selectbox(
            "Labelling method", tags,
            format_func=lambda t: D.POOL_LABEL_BLURB.get(t, (t, ""))[0],
            key=f"plm_{row['checkpoint']}_{v_sel}")
        pe = pl[m_sel]
        blurb = D.POOL_LABEL_BLURB.get(m_sel)
        if blurb:
            st.caption(blurb[1])

        ev = pe.get("eval") or {}
        if ev:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CV accuracy", f"{ev.get('accuracy', float('nan')):.3f}")
            if ev.get("balanced_accuracy") is not None:
                c2.metric("balanced accuracy",
                          f"{ev['balanced_accuracy']:.3f}")
            c3.metric("macro F1", f"{ev.get('macro_f1', float('nan')):.3f}")
            c4.metric("QWK", f"{ev.get('qwk', float('nan')):.3f}")
            st.caption(
                f"**Measured on the {ev.get('n_labelled', '?')} labelled craters "
                f"by {ev.get('n_folds', '?')}-fold cross-validation — NOT on the "
                "pool, which has no ground truth.** With 44/88/103/407 labels, "
                "always answering *v-dr* scores 0.63 accuracy while being "
                "useless, so balanced accuracy is the number that separates a "
                "real classifier from that. QWK is quadratic-weighted: these "
                "classes are ordinal, so confusing v-fr with v-dr costs far "
                "more than confusing neighbours.")
            pc = ev.get("per_class_f1") or {}
            if pc:
                st.dataframe(
                    pd.DataFrame({
                        "class": list(pc),
                        "F1": [pc[c] for c in pc],
                        "n labelled": [(ev.get("class_support") or {}).get(c)
                                       for c in pc],
                        "n in pool": [(ev.get("pool_distribution") or {}).get(c)
                                      for c in pc],
                    }), width="stretch", hide_index=True)

            # The caveat that governs every number above it.
            cs = ev.get("covariate_shift") or {}
            warns = cs.get("warnings") or []
            if warns:
                st.warning(
                    "**The labelled craters are not a random sample of the "
                    "pool**, so the accuracy above transfers to it only as far "
                    "as the two resemble each other:\n\n"
                    + "\n".join(f"- {w}" for w in warns)
                    + "\n\nThe true accuracy on the pool is **not measurable** "
                      "without labelling a random sample of it. The same "
                      "caveat applies to the class proportions: the labelled "
                      "set's fresh fraction is not an estimate of the pool's.")

        # ONE figure at a time, chosen by a radio rather than shown in tabs.
        # st.tabs renders every tab's body, so a 5-tab layout would ship all
        # five images (the map alone is ~7 MB at 3022x1096) on every rerun; a
        # radio means only the selected branch runs. Same reason the downloads
        # below are gated: st.download_button needs its bytes up front, so
        # four unconditional buttons read ~31 MB from disk into RAM EVERY time
        # anything on the page changes, including moving an unrelated slider.
        # The whole section is gated above, so inside it every figure renders -
        # the cost was paid once at the gate, and a reader comparing the
        # class-coloured PCA against the band-coloured one should not have to
        # click back and forth between them.
        if pe.get("map_png"):
            st.image(pe["map_png"], width="stretch")
        if pe.get("examples_png"):
            st.markdown("**Average crater and real members, per class**")
            st.image(pe["examples_png"], width="stretch")

        if pe.get("pca_png"):
            st.markdown("**Latent separation, coloured by class**")
            st.image(pe["pca_png"], width="stretch")
            st.caption("All 100k pool craters, coloured by the propagated "
                       "class. Silhouette in the title is computed in the "
                       "full 384-dim embedding, not in these 2 PCs.")
        if pe.get("umap_png"):
            st.markdown("**UMAP of the full pool, coloured by class**")
            st.image(pe["umap_png"], width="stretch")
            st.caption("Non-linear 2-D view of the same embedding (cosine, 30 "
                       "neighbours) - local neighbourhoods are kept, global "
                       "distances between islands are not meaningful.")
        if pe.get("pca_band_png"):
            st.markdown("**The same projection, coloured by acquisition band**")
            st.image(pe["pca_band_png"], width="stretch")
            ba = ev.get("band_predictable_balanced_acc")
            st.caption(
                "Identical coordinates to the figure above — compare them "
                "directly and a blob that looked like a class may turn out to "
                "be a band.\n\n"
                + (f"These 2 PCs cannot settle it either way, so the decisive "
                   f"number is printed instead: band is recoverable from the "
                   f"**full** embedding at balanced accuracy **{ba:.3f}** "
                   f"against a chance level of 0.333. Illumination is encoded "
                   f"whatever the picture suggests." if ba else ""))
        if pe.get("pca_images_png"):
            st.markdown("**The same projection, with real craters**")
            st.image(pe["pca_images_png"], width="stretch")
            st.caption("A random sample of pool craters drawn at their PCA "
                       "coordinates (border = assigned class), over all 100k "
                       "as grey dots. 100k thumbnails would overplot into a "
                       "grey mat, hence the sample.")
        missing = [lbl for lbl, key in
                   [("map", "map_png"), ("examples", "examples_png"),
                    ("PCA by class", "pca_png"),
                    ("PCA by band", "pca_band_png"),
                    ("PCA with images", "pca_images_png")]
                   if not pe.get(key)]
        if missing:
            st.caption(f"Not generated for this method yet: {', '.join(missing)}.")

        if st.checkbox("Prepare QGIS downloads", key=f"pldl_{v_sel}_{m_sel}",
                       help="Reads ~31 MB from disk, so it is off by default."):
            dc = st.columns(4)
            for i, (key, lbl) in enumerate([
                    ("zip", "QGIS bundle (.zip)"),
                    ("gpkg", "GeoPackage"),
                    ("qml", "QGIS style (.qml)"),
                    ("clusters_csv", "Per-crater CSV")]):
                f = pe.get(key)
                if f and os.path.exists(f):
                    with open(f, "rb") as fh:
                        dc[i % 4].download_button(
                            f"⬇ {lbl}", data=fh.read(),
                            file_name=f"{m_sel}_{os.path.basename(f)}",
                            key=f"pldlb_{m_sel}_{key}_{v_sel}")
            st.caption("The zip holds the GeoPackage, its `.qml` style, the "
                       "per-crater CSV and `eval.json` together — a .gpkg "
                       "opened without its .qml renders in random colours that "
                       "contradict every figure here.")

    if entry.get("summary") is not None and k_sel <= 12:
        with st.expander(f"Per-cluster spatial summary (k={k_sel})"):
            st.dataframe(entry["summary"], width="stretch", hide_index=True)
    if entry.get("panels_png"):
        with st.expander(f"Per-cluster panels (k={k_sel})"):
            st.image(entry["panels_png"], width="stretch")
            st.caption(
                "Supplementary to the coloured map above. Shows the most spatially "
                "concentrated clusters individually — spread < 1.00× = more localized "
                "than the pool as a whole.")

dc = D.data_comparison_artifacts(row.get("eval_run_dir"))
if dc.get("cells") or dc.get("geomap"):
    st.divider()
    st.header("Same model, different data")
    st.caption(
        "This checkpoint evaluated on every (source mosaic × high-pass filter × "
        "normalization) combination — the same 642 labeled craters throughout, so any "
        "difference between rows is a difference in *data presentation* and nothing "
        "else. Shown as a section rather than separate runs because the registry is "
        "one row per checkpoint: these all share this one checkpoint file.")

    cells = dc.get("cells") or {}
    if cells:
        tbl = D.data_comparison_table(cells)
        qual = ["silhouette_true", "map", "ari", "nmi", "qwk", "accuracy", "purity"]
        conf = ["lon_eta", "lat_eta", "brightness_eta", "sobel_eta", "diam_eta"]

        st.subheader("Metrics vs confounds")
        tbl = tbl.sort_values("silhouette_true", ascending=False)
        tbl = tbl.iloc[D.primary_first(range(len(tbl)),
                                       key=lambda i: (tbl.iloc[i]["catalog"], ""))]
        st.dataframe(
            tbl
               .style.format({c: "{:.3f}" for c in qual + conf
                              + ["pc1_vs_lon", "pc2_vs_lat"]})
               .background_gradient(cmap="Blues", subset=[c for c in qual if c in tbl])
               .background_gradient(cmap="Oranges", subset=[c for c in conf if c in tbl]),
            width="stretch", hide_index=True)
        st.caption(
            "**Blue = better higher. Orange = better LOWER** (these are confounds: how "
            "much the clustering is explained by geography, brightness, texture or "
            "crater size). Read the two halves together — across 67 recorded "
            "evaluations, corr(lon_eta, silhouette_true) = +0.248 (p = 0.043), so part "
            "of the apparent separation rides on the geographic artifact and a genuine "
            "fix is expected to *lower* separation somewhat.")
        M.explain("confound_correlation")
        M.explain("silhouette")
        M.explain("map")

        # Best-on-both, since the two objectives genuinely conflict and a
        # single blended score would hide which axis a cell won on.
        if tbl["silhouette_true"].notna().any() and tbl["lon_eta"].notna().any():
            ms = tbl["silhouette_true"].median()
            ml = tbl["lon_eta"].median()
            both = tbl[(tbl["silhouette_true"] > ms) & (tbl["lon_eta"] < ml)]
            st.markdown(f"**Above-median separation *and* below-median longitude "
                        f"artifact** (silhouette > {ms:.3f}, lon η² < {ml:.3f}): "
                        f"{len(both)} of {len(tbl)}")
            if len(both):
                st.dataframe(
                    both.sort_values("silhouette_true", ascending=False)
                        [["catalog", "norm"] + qual[:4] + conf[:3]]
                        .style.format({c: "{:.3f}" for c in qual[:4] + conf[:3]}),
                    width="stretch", hide_index=True)

        # ── metric charts, same idiom as Compare Runs ────────────────────────
        # There, one bar per RUN for a chosen metric. Here the runs are all the
        # same checkpoint, so one bar per DATA VARIANT instead - the identical
        # question ("which of these is better on this metric") asked along the
        # other axis.
        st.subheader("Metric charts by data variant")
        DC_METRICS = {
            "silhouette_true": "Latent separation (silhouette, true labels)",
            "map": "mAP (mean average precision)",
            "ari": "ARI (cluster agreement)",
            "nmi": "NMI",
            "qwk": "Quadratic weighted kappa",
            "accuracy": "Accuracy (Hungarian-aligned)",
            "purity": "Purity",
            "lon_eta": "LONGITUDE confound η²  (lower is better)",
            "lat_eta": "Latitude confound η²  (lower is better)",
            "brightness_eta": "Brightness confound η²  (lower is better)",
            "sobel_eta": "Sobel/texture confound η²  (lower is better)",
            "diam_eta": "Diameter confound η²  (lower is better)",
        }
        avail = [m for m in DC_METRICS if m in tbl.columns and tbl[m].notna().any()]
        chosen = st.multiselect(
            "Metrics to chart", avail,
            default=[m for m in ("silhouette_true", "lon_eta") if m in avail],
            format_func=lambda m: DC_METRICS[m],
            key=f"dc_metrics_{row['checkpoint']}")
        norm_filter = st.multiselect(
            "Normalizations to include", sorted({n for _, n in cells}),
            default=sorted({n for _, n in cells}),
            key=f"dc_normfilter_{row['checkpoint']}")
        chart_df = tbl[tbl["norm"].isin(norm_filter)].copy()
        chart_df["variant"] = chart_df["catalog"] + " / " + chart_df["norm"]
        if chosen and not chart_df.empty:
            ccols = st.columns(min(2, len(chosen)))
            for i, m in enumerate(chosen):
                t = ccols[i % len(ccols)]
                t.caption(DC_METRICS[m])
                d = chart_df.set_index("variant")[[m]].dropna().sort_values(m)
                if d.empty:
                    t.info("No variant has a value for this metric.")
                else:
                    t.bar_chart(d, horizontal=True)
            st.caption(
                "Bars sorted ascending, so for the **η² confound** metrics the best "
                "variants are at the TOP and for the quality metrics at the bottom. "
                "Charting one of each together is the intended use — that pairing is "
                "the whole decision.")

        # ── side-by-side figure gallery, same idiom as Compare Runs ─────────
        st.subheader("Compare figures across data variants")
        FIG_CHOICES = {
            "Latent separation (PCA)": "latent_separation",
            "Latent separation (UMAP)": "latent_separation_umap",
            "Latent separation (LDA head)": "latent_separation_lda_head",
            "Confusion matrix": "confusion_matrix",
            "Clusters by sample": "clusters_by_sample",
            "Agreement examples": "agreement_examples",
        }
        # Offer whatever actually exists, plus anything else the suite wrote.
        present = {n for c in cells.values() for n in c["figures"]}
        extra = sorted(present - set(FIG_CHOICES.values()))
        choices = {k: v for k, v in FIG_CHOICES.items() if v in present}
        choices.update({n.replace("_", " "): n for n in extra})
        if not choices:
            st.info("No per-cell figures found.")
        else:
            f1, f2, f3 = st.columns(3)
            fig_label = f1.selectbox("Figure", list(choices),
                                     key=f"dc_fig_{row['checkpoint']}")
            fig_key = choices[fig_label]
            gal_cols = f2.radio("Columns", [2, 3, 4], index=1, horizontal=True,
                                key=f"dc_cols_{row['checkpoint']}")
            cap_metric = f3.selectbox(
                "Caption metric", avail,
                index=avail.index("silhouette_true") if "silhouette_true" in avail else 0,
                format_func=lambda m: DC_METRICS[m],
                key=f"dc_capm_{row['checkpoint']}")

            variant_keys = D.primary_first([k for k in sorted(cells) if k[1] in norm_filter])
            picked = st.multiselect(
                "Variants shown", variant_keys,
                default=variant_keys,
                format_func=lambda kk: f"{kk[0]} / {kk[1]}",
                key=f"dc_gal_{row['checkpoint']}")
            if not picked:
                st.info("No variants selected.")
            else:
                metric_by_variant = {
                    (r["catalog"], r["norm"]): r.get(cap_metric)
                    for _, r in tbl.iterrows()}
                icols = st.columns(min(gal_cols, len(picked)))
                any_shown = False
                for i, kk in enumerate(picked):
                    col = icols[i % len(icols)]
                    path = cells[kk]["figures"].get(fig_key)
                    v = metric_by_variant.get(kk)
                    cap = f"{kk[0]} / {kk[1]}"
                    if v is not None and pd.notna(v):
                        cap += f"  ({cap_metric}: {v:.3f})"
                    if path:
                        col.image(path, caption=cap, width="stretch")
                        any_shown = True
                    else:
                        col.caption(f"{cap}: not available")
                if not any_shown:
                    st.info("None of the selected variants have this figure.")

holdout = summary.get("holdout")
if holdout:
    st.divider()
    st.subheader("Held-out unlabeled set")
    h1, h2, h3, h4 = st.columns(4)
    rl = holdout.get("reconstruction_loss")
    if rl:
        h1.metric("Reconstruction mean loss", f"{rl['mean_loss']:.4f}",
                 f"±{rl['std_loss']:.4f}", delta_color="off")
    q = holdout.get("cluster_quality", {})
    if q:
        h2.metric("Silhouette", f"{q.get('silhouette', 0):.3f}")
        h3.metric("Davies-Bouldin", f"{q.get('davies_bouldin', 0):.3f}")
        h4.metric("Calinski-Harabasz", f"{q.get('calinski_harabasz', 0):.1f}")

st.divider()
with st.expander("Raw summary.json"):
    st.json(summary)
