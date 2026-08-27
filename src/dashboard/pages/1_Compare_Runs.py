"""Cross-run comparison: filter/group runs by eval metrics AND training
parameters (family, scratch-vs-pretrained, epochs, mask ratio, ...), then
compare them by metric (one chart per metric - never mixes scales) and by
the actual result plots side by side (confusion matrix, latent separation,
agreement examples, loss curve). Built off eval_history.csv + the checkpoint
registry - no live recomputation.

Two tabs: "Compare" (sidebar-filtered, side-by-side) and "Leaderboard"
(top-N across the FULL history, ignoring the Compare tab's filters)."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import streamlit as st

from dashboard import data as D
from eval import history as HIST

st.set_page_config(page_title="Compare Runs", layout="wide")
st.title("Compare Runs")

history_df = D.load_history()
if history_df.empty:
    st.info("logs/eval_history.csv is empty or missing - run eval.evaluate at least once.")
    st.stop()

df = D.enrich_history_with_params(history_df)
df = D.join_history_to_dirs(df, D.list_run_dirs())
registry = D.build_registry()[["checkpoint", "training_run_dir"]]
df = df.merge(registry, on="checkpoint", how="left")


def _is_incomplete(row) -> bool:
    """A run's training-param metadata has a real gap (unknown source, no
    family, no data_tag) - not a dashboard bug, just something worth
    flagging so a blank field doesn't get silently read as "nothing to
    know" rather than "not recorded". See model_meta.py's param_source
    (confidence) vs. the actual param values here - this checks the latter.

    Stock-pretrained checkpoints are exempt: they were never trained on our
    data at all, so having no data_tag is correct, not missing."""
    family = row.get("param_family")
    source = row.get("param_source")
    data_tag = row.get("param_data_tag")
    if source == "stock_pretrained":
        return False
    return (pd.isna(family) or not family
           or pd.isna(source) or source == "unknown"
           or pd.isna(data_tag) or not data_tag)


def _run_label(row) -> str:
    """Friendly display label for a run - the raw run_id (e.g.
    'mae_20260820-131833_da98de5f') uniquely identifies a run but isn't
    something a person can tell runs apart by at a glance. run_id stays the
    underlying value everywhere (filtering, joins, chart indexing) - this
    only changes what's shown."""
    bits = [str(row.get("autoencoder_model") or "?")]
    source = row.get("param_source")
    if pd.notna(source):
        bits.append(str(source))
    epochs = row.get("param_epochs")
    if pd.notna(epochs):
        bits.append(f"ep{epochs:g}")
    data_tag = row.get("param_data_tag")
    if pd.notna(data_tag) and data_tag:
        bits.append(str(data_tag))
    fov = row.get("param_fov")
    if pd.notna(fov):
        # DINO (1.0) and MAE/CAE (0.5) can share a data_tag - same
        # underlying mosaic/sigma source, differently-framed crops - so this
        # is what actually tells two same-data_tag runs apart at a glance.
        bits.append(f"fov{fov:g}")
    ts = row.get("timestamp")
    if pd.notna(ts):
        bits.append(str(ts)[5:16].replace("T", " "))  # "2026-08-20T13:18:33" -> "08-20 13:18"
    label = " · ".join(bits)
    if row.get("metadata_incomplete"):
        label = f"⚠ {label}"
    tag = row.get("tag")
    if pd.notna(tag) and tag:
        label = f"[{tag}] {label}"
    return label


df["metadata_incomplete"] = df.apply(_is_incomplete, axis=1)
df["run_label"] = df.apply(_run_label, axis=1)

groups = D.metric_groups(history_df)
metric_options = [c for cols in groups.values() for c in cols]
if not metric_options:
    st.warning("No metric columns found in eval_history.csv.")
    st.stop()
label_map = D.metric_label_map(groups)

# Friendly labels for the non-metric columns shown in either table below -
# metric columns get theirs from label_map instead (formatted as numbers).
COLUMN_LABELS = {
    "run_id": "Run ID", "run_label": "Run", "checkpoint": "Checkpoint",
    "autoencoder_model": "Architecture", "training_run_dir": "Training run dir",
    "param_family": "Family", "param_source": "Source", "param_epochs": "Epochs",
    "param_mask_ratio": "Mask ratio", "param_num_samples": "Num samples",
    "param_data_tag": "Data tag", "provenance": "Params provenance",
    "param_pretrained_weights": "Pretrained weights", "param_latent_dim": "Latent dim",
    "param_diameter_range": "Diameter range (km)", "param_freeze_until": "Freeze until",
    "param_iterations": "Iterations", "param_param_source": "Params provenance",
    "param_fov": "FOV (crop radius ×diameter)",
    "param_proto_loss_enabled": "Proto loss", "param_proto_loss_weight": "Proto loss weight",
    "param_proto_n_support_per_class": "Proto n_support/class",
    "param_proto_split_csv": "Proto split csv",
}

# Every param_* field worth showing in the row detail panel, in display order -
# same underlying columns enrich_history_with_params() adds, just relabeled
# and filtered for presence per row rather than shown as a flat wide table.
DETAIL_PARAM_COLS = [
    "param_family", "param_source", "param_pretrained_weights", "param_epochs",
    "param_iterations", "param_mask_ratio", "param_latent_dim", "param_num_samples",
    "param_diameter_range", "param_fov", "param_freeze_until", "param_data_tag",
    "param_proto_loss_enabled", "param_proto_loss_weight", "param_proto_n_support_per_class",
    "param_param_source",
]


def _is_present(val) -> bool:
    # diameter_range is a 2-element list, not a scalar - pd.notna() on it
    # returns an elementwise array rather than a single bool.
    if isinstance(val, (list, tuple)):
        return len(val) > 0
    return pd.notna(val)


def _fmt_val(val):
    if isinstance(val, (list, tuple)):
        return " – ".join(str(v) for v in val)
    return val


def _render_row_detail(row) -> None:
    """Full context for one selected run - checkpoint path, every param
    field, provenance - inline in Compare Runs instead of requiring a
    switch to Run Deep Dive just to see what a row actually is."""
    with st.container(border=True):
        st.markdown(f"**{row.get('run_label', row.get('run_id'))}**")
        st.code(row.get("checkpoint", ""), language=None)

        meta_cols = st.columns(2)
        meta_cols[0].caption(f"Run ID: `{row.get('run_id')}`")
        if _is_present(row.get("tag")):
            meta_cols[0].caption(f"Tag: `{row['tag']}`")
        if _is_present(row.get("training_run_dir")):
            meta_cols[0].caption(f"Training dir: `{row['training_run_dir']}`")
        if _is_present(row.get("run_dir")):
            meta_cols[1].caption(f"Eval run dir: `{row['run_dir']}`")
        if _is_present(row.get("timestamp")):
            meta_cols[1].caption(f"Evaluated: {row['timestamp']}")

        detail = {COLUMN_LABELS.get(c, c): _fmt_val(row[c]) for c in DETAIL_PARAM_COLS
                  if c in row.index and _is_present(row[c])}
        if detail:
            st.table(pd.DataFrame(detail.items(), columns=["Field", "Value"]).set_index("Field"))
        st.caption("For plots and other diagnostics, see Run Deep Dive.")


def _column_config(cols: list[str]) -> dict:
    config = {}
    for c in cols:
        if c in COLUMN_LABELS:
            config[c] = COLUMN_LABELS[c]
        elif c in label_map:
            config[c] = st.column_config.NumberColumn(label_map[c], format="%.4f")
    return config


compare_tab, leaderboard_tab = st.tabs(["Compare", "Leaderboard"])

# ============================================================================
# Compare tab
# ============================================================================
with compare_tab:
    # --- Filters --------------------------------------------------------------
    st.sidebar.header("Filters")

    only_latest = st.sidebar.checkbox(
        "Only latest run per checkpoint", value=True,
        help="Re-evaluating the same checkpoint (e.g. after an eval-suite change) "
             "adds a new history row rather than replacing the old one - on by "
             "default so repeat evals don't clutter the table. Off shows every "
             "eval ever run, including superseded ones.")
    base_df = df
    if only_latest and "timestamp" in df.columns:
        base_df = df.sort_values("timestamp").drop_duplicates("checkpoint", keep="last")

    complete_only = st.sidebar.checkbox(
        "Only runs with complete metadata", value=False,
        help="Hides runs flagged ⚠ - missing family/source/data_tag, usually because "
             "the training run predates that field being recorded (e.g. an old "
             "Snakefile.snapshot with no DATA_TAG variable) rather than a dashboard "
             "bug. Off by default so incomplete runs stay visible instead of "
             "silently disappearing.")
    if complete_only:
        base_df = base_df[~base_df["metadata_incomplete"]]

    def _ms(label, col):
        if col not in base_df.columns:
            return None
        options = sorted(v for v in base_df[col].dropna().unique())
        if not options:
            return None
        return st.sidebar.multiselect(label, options)

    f_arch = _ms("Architecture (eval)", "autoencoder_model")
    f_family = _ms("Family", "param_family")
    f_source = _ms("Source (scratch/finetune/pretrained)", "param_source")
    f_epochs = _ms("Epochs", "param_epochs")
    f_mask = _ms("Mask ratio", "param_mask_ratio")
    f_nsamples = _ms("Num samples", "param_num_samples")
    f_fov = _ms("FOV (crop radius ×diameter)", "param_fov")
    f_checkpoint = st.sidebar.text_input("Checkpoint contains")

    filtered = base_df.copy()
    for col, values in [
        ("autoencoder_model", f_arch), ("param_family", f_family),
        ("param_source", f_source), ("param_epochs", f_epochs),
        ("param_mask_ratio", f_mask), ("param_num_samples", f_nsamples),
        ("param_fov", f_fov),
    ]:
        if values:
            filtered = filtered[filtered[col].isin(values)]
    if f_checkpoint:
        filtered = filtered[filtered["checkpoint"].str.contains(f_checkpoint, case=False, na=False)]

    st.caption(f"{len(filtered)} / {len(df)} runs match the current filters")

    # --- Display options: grouping, metrics, run selection - sidebar too, so
    # the main body opens straight into results instead of a wall of widgets.
    st.sidebar.divider()
    st.sidebar.header("Display")

    group_options = [c for c in ["training_run_dir", "param_family", "param_source",
                                 "param_epochs", "param_mask_ratio", "param_num_samples",
                                 "autoencoder_model"]
                     if c in filtered.columns]
    group_by = st.sidebar.multiselect(
        "Group by (bunches matching runs together in the table/chart - "
        "'Training run dir' clusters re-evals of the same checkpoint)",
        group_options, default=["training_run_dir"] if "training_run_dir" in group_options else [],
        format_func=lambda c: COLUMN_LABELS.get(c, c))
    if group_by:
        filtered = filtered.sort_values(group_by)

    primary_metric = st.sidebar.selectbox("Sort by metric", metric_options,
                                          format_func=lambda c: label_map.get(c, c))
    compare_metrics = st.sidebar.multiselect(
        "Metrics to compare (one chart each, so scales never mix)", metric_options,
        default=[primary_metric], format_func=lambda c: label_map.get(c, c))
    minimize = st.sidebar.checkbox("Lower is better for the sort metric",
                                   value=D.metric_is_lower_better(primary_metric))

    run_ids = filtered["run_id"].tolist()
    label_by_id = dict(zip(filtered["run_id"], filtered["run_label"]))
    selected_ids = st.sidebar.multiselect("Runs to compare", run_ids, default=run_ids,
                                          format_func=lambda rid: label_by_id.get(rid, rid))

    selected = filtered[filtered["run_id"].isin(selected_ids)]

    id_cols = ["run_label", "checkpoint", "autoencoder_model", "param_family",
              "param_source", "param_data_tag", "param_fov", "param_epochs",
              "param_mask_ratio", "param_num_samples"]
    id_cols = [c for c in id_cols if c in selected.columns]
    show_cols = id_cols + [c for c in compare_metrics if c not in id_cols]

    ordered = selected.sort_values(primary_metric, ascending=minimize, na_position="last")

    with st.container(border=True):
        st.subheader("Results table")
        st.caption("Select a row to see its full context (checkpoint path, every param field, provenance).")
        selection = st.dataframe(
            ordered[show_cols], width='stretch', column_config=_column_config(show_cols),
            on_select="rerun", selection_mode="single-row", key="results_table")
        selected_rows = selection.selection.rows if selection and selection.selection else []
        if selected_rows:
            _render_row_detail(ordered.iloc[selected_rows[0]])

    # --- One chart per metric: each has exactly one scale, by construction --
    if compare_metrics:
        st.divider()
        with st.container(border=True):
            st.subheader("Metric charts")
            chart_base = ordered.set_index("run_id")
            chart_cols = st.columns(min(2, len(compare_metrics)) or 1)
            shown_any = False
            for i, metric in enumerate(compare_metrics):
                col_data = chart_base[[metric]].dropna()
                target = chart_cols[i % len(chart_cols)]
                target.caption(label_map.get(metric, metric))
                if col_data.empty:
                    target.info("No selected run has a value for this metric.")
                else:
                    target.bar_chart(col_data)
                    shown_any = True
            if not shown_any:
                st.info("No selected run has a value for any chosen metric.")

    # --- Compare the actual result plots side by side --------------------------
    st.divider()
    with st.container(border=True):
        st.subheader("Compare visual results")
        labeled_set_names = [g for g in groups.keys() if g != "holdout"]
        plot_choices = {"Loss curve (training)": "loss_curve"}
        if labeled_set_names:
            plot_choices.update({
                "Confusion matrix": "confusion_matrix_png",
                "Clusters by sample": "clusters_by_sample_png",
                "Latent separation (PCA)": "latent_separation_png",
                "Latent separation (UMAP)": "latent_separation_umap_png",
                "Agreement examples": "agreement_examples_png",
            })
            # Unsupervised over-clustering exploration (evaluate.py's
            # evaluate_labeled_set(), see visualize.py's plot_unsupervised_
            # cluster_dots/_images) - keyed by a ("unsup", k, technique, kind)
            # tuple rather than a flat string like every other entry here,
            # since load_labeled_set_artifacts() nests these under
            # unsupervised_cluster_pngs[(k, technique, kind)] instead of one
            # top-level key per plot - branched on below at lookup time.
            for k in (10, 50):
                for tech in ("pca", "umap"):
                    plot_choices[f"Unsupervised clusters k={k} ({tech.upper()})"] = ("unsup", k, tech, "dots")
                    plot_choices[f"Unsupervised clusters k={k} ({tech.upper()}), crops"] = ("unsup", k, tech, "images")

        vc1, vc2, vc3 = st.columns(3)
        vplot_label = vc1.selectbox("Plot", list(plot_choices.keys()))
        vplot_key = plot_choices[vplot_label]
        vset = vc2.selectbox("Labeled set", labeled_set_names) if vplot_key != "loss_curve" else None
        gallery_cols = vc3.radio("Columns", [2, 3, 4], index=1, horizontal=True)

        runs_to_show = ordered
        if runs_to_show.empty:
            st.info("No runs selected.")
        else:
            gallery_label_by_id = dict(zip(runs_to_show["run_id"], runs_to_show["run_label"]))
            gallery_runs = st.multiselect(
                "Runs shown in gallery", runs_to_show["run_id"].tolist(),
                default=runs_to_show["run_id"].tolist(),
                format_func=lambda rid: gallery_label_by_id.get(rid, rid))
            gallery_df = runs_to_show[runs_to_show["run_id"].isin(gallery_runs)]

            if gallery_df.empty:
                st.info("No runs selected for the gallery.")
            else:
                img_cols = st.columns(min(gallery_cols, len(gallery_df)))
                any_shown = False
                for i, (_, r) in enumerate(gallery_df.iterrows()):
                    col = img_cols[i % len(img_cols)]
                    path = None
                    if vplot_key == "loss_curve":
                        if pd.notna(r.get("training_run_dir")):
                            path = D.training_artifacts(r["training_run_dir"])["loss_curve_png"]
                    elif pd.notna(r.get("run_dir")):
                        ea = D.load_labeled_set_artifacts(r["run_dir"], vset)
                        if isinstance(vplot_key, tuple) and vplot_key[0] == "unsup":
                            _, k, tech, kind = vplot_key
                            path = ea["unsupervised_cluster_pngs"].get((k, tech, kind))
                        else:
                            path = ea[vplot_key]
                    metric_val = r.get(primary_metric)
                    caption = r["run_label"]
                    if pd.notna(metric_val):
                        caption = f"{r['run_label']}  ({label_map.get(primary_metric, primary_metric)}: {metric_val:.3f})"
                    if path:
                        col.image(path, caption=caption, width='stretch')
                        any_shown = True
                    else:
                        col.caption(f"{r['run_label']}: not available")
                if not any_shown:
                    st.info("None of the selected runs have this plot available.")

# ============================================================================
# Leaderboard tab
# ============================================================================
with leaderboard_tab:
    st.subheader("Leaderboard")
    st.caption("Top runs across the full history, ignoring the Compare tab's filters.")

    c1, c2, c3 = st.columns(3)
    top_metric = c1.selectbox("Metric", metric_options,
                              format_func=lambda c: label_map.get(c, c), key="lb_metric")
    top_n = c2.slider("N", min_value=1, max_value=50, value=10, key="lb_n")
    top_minimize = c3.checkbox("Lower is better", value=D.metric_is_lower_better(top_metric),
                               key="lb_minimize")
    try:
        top_df = HIST.best_runs(top_metric, D.HISTORY_PATH, int(top_n), top_minimize)
        top_df = D.enrich_history_with_params(top_df)
        # param_param_source = provenance of the training-param columns
        # themselves (run_manifest / dino_config / snakefile_snapshot /
        # dirname_regex / stock / unknown) - not to be confused with
        # param_source (scratch/finetune/pretrained, an actual training param).
        top_df = top_df.rename(columns={"param_param_source": "provenance"})
        top_df["run_label"] = top_df.apply(_run_label, axis=1)
        lb_cols = [c for c in ["run_label", top_metric, "param_family", "param_source",
                               "param_epochs", "param_mask_ratio", "param_num_samples",
                               "provenance", "checkpoint"] if c in top_df.columns]
        cmap = "Blues_r" if top_minimize else "Blues"
        styled = top_df[lb_cols].style.background_gradient(subset=[top_metric], cmap=cmap)
        st.dataframe(styled, width='stretch', column_config=_column_config(lb_cols))
    except ValueError as e:
        st.warning(str(e))
