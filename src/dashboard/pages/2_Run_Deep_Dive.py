"""Single-run deep dive: browse one checkpoint's full results. Checkpoints
come from the full registry (every trained model under logs/, evaluated or
not), not just ones with an eval_results/ entry - nothing here is
recomputed, it only reads what evaluate.py/train.py already wrote."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from dashboard import data as D
from dashboard import model_meta as MM

st.set_page_config(page_title="Run Deep Dive", layout="wide")
st.title("Run Deep Dive")

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
eval_artifacts_by_set = {}
for set_name in labeled_sets:
    ea = D.load_labeled_set_artifacts(row["eval_run_dir"], set_name)
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
    tabs = st.tabs(list(labeled_sets.keys()))
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

            eval_artifacts = eval_artifacts_by_set[set_name]
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
else:
    st.divider()
    st.info("This run has no labeled-set results.")

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
