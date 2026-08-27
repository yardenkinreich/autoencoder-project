"""
visualize.py — plots for the eval report. Kept separate from metrics.py,
which stays pure-function/no-plotting (dicts/arrays only, testable).

From src/test/evaluate.py's plot_confusion_matrix / display_craters_by_cluster,
adapted to this package's df schema (true_label/pred_label/cluster, already
Hungarian/majority-vote aligned by align.py — no re-alignment here).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# class_names (e.g. Semi-New/Semi-Old/Old, or v-fr/r-fr/r-dr/v-dr) is an
# ORDINAL degradation scale - swapping the order changes the meaning - so
# classes get one ramp's monotone steps rather than unrelated categorical
# hues, and every plot below colors by class_idx (the ALIGNED index from
# eval/align.py, always the same index for the same class_names entry)
# rather than by raw cluster id or draw order - so "Old"/"v-dr" is always
# the same dark red, on every run, on every labeled set, even though which
# raw KMeans cluster maps to it varies run to run.
#
# gold -> orange -> red -> dark red ("fresh" to "most degraded"): a
# multi-hue warm ramp rather than one flat hue - a light single-hue blue
# ramp (the previous version here) reads as too washed-out/hard to tell
# apart at a glance. Interpolated from 4 anchor stops (#B8860B/#D2691E/
# #B22222/#4A0000), and unlike the pastel end of a stock colormap like
# ColorBrewer's YlOrRd, the "fresh" anchor is a saturated gold rather than
# pale yellow specifically so it's still legible as text (this ramp colors
# matplotlib title text directly, not just fill dots) - contrast ratio
# against white rises monotonically from 3.25 (gold) to 16.26 (darkest
# red), so every step clears the WCAG large-text floor (3.0), most clear
# the normal-text floor (4.5) too.
_ORDINAL_RAMP = ["#b8860b", "#c17c11", "#c97318", "#d2691e", "#c7521f",
                 "#bd3921", "#b22222", "#8f1616", "#6c0b0b", "#4a0000"]


def _class_color(class_idx: int | None, n_classes: int) -> str:
    if class_idx is None or not (0 <= class_idx < n_classes):
        return "#898781"  # muted gray - out-of-range/unknown, never a ramp step
    if n_classes <= 1:
        return _ORDINAL_RAMP[len(_ORDINAL_RAMP) // 2]
    positions = np.linspace(0, len(_ORDINAL_RAMP) - 1, n_classes)
    return _ORDINAL_RAMP[round(positions[class_idx])]


def _load_crop_for_display(path: str) -> np.ndarray:
    """A display crop is either a PNG (julie's set - labels_from_png_dir)
    or a .npy tensor (a memmap-based set's saved --save_raw_crops crop -
    labels_from_memmap_csv), saved as (num_channels, H, W) already in the
    model's own normalized range. Either way, returns a plain (H, W) array
    for ax.imshow(..., cmap="gray")."""
    if path.endswith(".npy"):
        return np.load(path)[0]
    return np.array(Image.open(path).convert("L"))


def _reduce_2d(latents: np.ndarray, technique: str):
    """Shared PCA/t-SNE/UMAP projection - factored out of plot_latent_separation
    so the unsupervised-cluster plots below (plot_unsupervised_cluster_dots/
    _images) can reuse the exact same projections instead of a second,
    potentially-drifting implementation."""
    if technique == "pca":
        proj = PCA(n_components=2, random_state=0)
        coords = proj.fit_transform(latents)
        ev = proj.explained_variance_ratio_
        xlabel, ylabel = f"PC1 ({ev[0]*100:.1f}%)", f"PC2 ({ev[1]*100:.1f}%)"
    elif technique == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=0).fit_transform(latents)
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    elif technique == "umap":
        import umap
        coords = umap.UMAP(n_components=2, random_state=0).fit_transform(latents)
        xlabel, ylabel = "UMAP 1", "UMAP 2"
    else:
        raise ValueError(f"unknown technique: {technique}")
    return coords, xlabel, ylabel


def _cluster_color(cluster_id: int, n_clusters: int) -> tuple:
    """Distinct hue per raw cluster id, for the UNSUPERVISED plots below only
    (plot_unsupervised_cluster_dots/_images) - deliberately NOT the ordinal
    _ORDINAL_RAMP every other plot in this file uses, since that ramp encodes
    a meaningful fresh->degraded ordering for the known 3-4 true classes,
    which a k=10/k=50 exploratory KMeans has no relationship to at all (its
    cluster ids are arbitrary integers, not an ordinal scale) - using the
    ordinal ramp here would visually imply an ordering that isn't there.
    nipy_spectral sampled evenly across its range stays distinguishable up to
    several dozen categories, matplotlib's own suggested approach for many
    discrete categories (no fixed palette goes that high while staying
    individually distinct)."""
    if n_clusters <= 1:
        return plt.cm.nipy_spectral(0.5)
    return plt.cm.nipy_spectral(cluster_id / (n_clusters - 1))


def plot_unsupervised_cluster_dots(latents: np.ndarray, cluster_labels: np.ndarray,
                                   out_path: str, technique: str = "pca",
                                   model_name: str = "") -> None:
    """Purely unsupervised exploration: project latents to 2D and color by a
    KMeans cluster id from an over-clustering k (e.g. 10 or 50) FAR larger
    than the known number of true degradation classes - deliberately no
    true_label anywhere in this function, unlike plot_latent_separation's
    true-vs-predicted pairing. The point is to look for finer structure
    within/across the known classes that a small, label-matched k can't
    reveal, not to validate against the labels at all."""
    n_clusters = len(np.unique(cluster_labels))
    coords, xlabel, ylabel = _reduce_2d(latents, technique)

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=20, alpha=0.75,
                  color=_cluster_color(c, n_clusters))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"Unsupervised k={n_clusters} clusters ({technique.upper()})"
                + (f" — {model_name}" if model_name else ""))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_unsupervised_cluster_images(latents: np.ndarray, cluster_labels: np.ndarray,
                                     df: pd.DataFrame, imgs_dir: str, out_path: str,
                                     technique: str = "pca", model_name: str = "",
                                     zoom: float = 0.18) -> None:
    """Same projection/coloring as plot_unsupervised_cluster_dots, but each
    point is the actual crater crop (from df['filename'] under imgs_dir - see
    _load_crop_for_display for the .npy-vs-.png convention) instead of a
    colored dot, with a thin border in that point's cluster color so
    membership stays visible even though the image itself carries no color
    coding. Same AnnotationBbox/OffsetImage mechanism as src/cluster/
    cluster.py's plot_imgs (kept here instead of imported from there so this
    module doesn't gain a dependency on that standalone CLI tool)."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    n_clusters = len(np.unique(cluster_labels))
    coords, xlabel, ylabel = _reduce_2d(latents, technique)

    fig, ax = plt.subplots(figsize=(14, 12))
    for (x, y), cluster_id, fname in zip(coords, cluster_labels, df["filename"]):
        img_path = os.path.join(imgs_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = _load_crop_for_display(img_path)
        color = _cluster_color(cluster_id, n_clusters)
        ab = AnnotationBbox(OffsetImage(img, zoom=zoom, cmap="gray"), (x, y),
                            frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor=color, linewidth=2))
        ax.add_artist(ab)

    ax.update_datalim(coords)
    ax.autoscale()
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"Unsupervised k={n_clusters} clusters, crater crops ({technique.upper()})"
                + (f" — {model_name}" if model_name else ""))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latent_separation(latents: np.ndarray, df: pd.DataFrame,
                           class_names: list[str], out_path: str,
                           technique: str = "pca"):
    """
    2D projection of the latent space, colored by TRUE label and by
    PREDICTED label side by side — the direct visual answer to "how
    separated are the classes, and does the model's own clustering agree?"
    Complements the scalar geometry_comparison/boundary_uncertainty metrics
    with something you can actually look at.
    """
    coords, xlabel, ylabel = _reduce_2d(latents, technique)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_classes = len(class_names)

    for ax, col, title in [(axes[0], "true_label", "True label"),
                           (axes[1], "pred_label", "Predicted (aligned)")]:
        for v in sorted(df[col].unique()):
            mask = (df[col] == v).to_numpy()
            ax.scatter(coords[mask, 0], coords[mask, 1], s=30, alpha=0.7,
                      color=_class_color(v, n_classes),
                      label=class_names[v] if v < n_classes else str(v))
        ax.set_title(title)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.suptitle(f"Latent space separation ({technique.upper()})", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lda_head_separation(latents: np.ndarray, df: pd.DataFrame,
                             class_names: list[str], out_path: str):
    """Same visual pairing as plot_latent_separation (true label vs.
    predicted, side by side), but through the LDA head (eval/head.py)
    instead of a raw-latent PCA/UMAP/t-SNE projection. No separate
    dimensionality-reduction step is needed - LDA's own output already IS a
    low-dimensional projection (top-2 of its up-to-(n_classes-1)
    components), fit specifically to separate these classes, unlike PCA/
    UMAP which are blind to the labels entirely.

    "Predicted" here is the LDA's own .predict() output directly - no
    Hungarian alignment needed the way the raw-latent panel requires (LDA
    is a supervised classifier; its predicted class index already IS the
    real class index, not an arbitrary cluster id).

    Fit on the FULL labeled set for this plot (a visual diagnostic, not a
    metric) - the actual head_comparison numbers in metrics.json are the
    strictly cross-validated, out-of-fold ones (see head.py's module
    docstring); this plot would look optimistically clean by comparison
    since it's seeing the same data it was fit on."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    y = df["true_label"].to_numpy()
    n_components = min(2, len(np.unique(y)) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components).fit(latents, y)
    coords = lda.transform(latents)
    if n_components == 1:
        # 2-class case: LDA only has one discriminant axis - fake a y-axis
        # (zero) so the same 2-panel scatter code below still works.
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    pred = lda.predict(latents)

    plot_df = df.copy()
    plot_df["pred_label"] = pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_classes = len(class_names)
    for ax, col, title in [(axes[0], "true_label", "True label"),
                           (axes[1], "pred_label", "Predicted (LDA head)")]:
        for v in sorted(plot_df[col].unique()):
            mask = (plot_df[col] == v).to_numpy()
            ax.scatter(coords[mask, 0], coords[mask, 1], s=30, alpha=0.7,
                      color=_class_color(v, n_classes),
                      label=class_names[v] if v < n_classes else str(v))
        ax.set_title(title)
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2" if n_components == 2 else "")
        ax.legend(fontsize=8)

    plt.suptitle("Latent space separation (LDA head, fit on full labeled set - "
                 "see head_comparison in metrics.json for the CV'd numbers)", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(df: pd.DataFrame, class_names: list[str], out_path: str):
    """Aligned confusion matrix, raw counts + row-normalized, side by side."""
    n = len(class_names)
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=list(range(n)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Aligned raw counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Aligned normalized (recall)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def display_craters_by_cluster(df: pd.DataFrame, imgs_dir: str, out_path: str,
                                class_names: list[str], cluster_to_class: dict[int, int],
                                samples_per_cluster: int = 5, seed: int = 0):
    """
    Grid of sample crater images per RAW cluster (pre-alignment - useful to
    eyeball what a cluster actually looks like before trusting the label
    alignment). Rows are still grouped by the raw cluster id (nothing about
    membership is re-derived here), but ORDERED and COLORED by the class
    each cluster was aligned to (cluster_to_class, from eval/align.py) - so
    row order/color is stable across runs even though which raw id lands
    where is arbitrary per run. df needs 'cluster' and 'filename' columns.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(class_names)
    clusters = sorted(df["cluster"].unique(),
                      key=lambda c: cluster_to_class.get(c, n_classes))
    n_clusters = len(clusters)

    fig, axes = plt.subplots(n_clusters, samples_per_cluster,
                             figsize=(3 * samples_per_cluster, 3 * n_clusters),
                             squeeze=False)

    for row, cl in enumerate(clusters):
        sub = df[df["cluster"] == cl]
        n_samples = min(samples_per_cluster, len(sub))
        sample = sub.sample(n=n_samples, random_state=int(rng.integers(0, 2**31)))
        class_idx = cluster_to_class.get(cl)
        class_label = class_names[class_idx] if class_idx is not None and class_idx < n_classes else "?"
        color = _class_color(class_idx, n_classes)

        for col in range(samples_per_cluster):
            ax = axes[row, col]
            ax.axis("off")
            if col >= n_samples:
                continue
            fname = sample.iloc[col]["filename"]
            img_path = os.path.join(imgs_dir, fname)
            if os.path.exists(img_path):
                ax.imshow(_load_crop_for_display(img_path), cmap="gray")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            if col == 0:
                ax.set_title(f"cluster {cl} → {class_label} (n={len(sub)})",
                             fontsize=10, loc="left", color=color, fontweight="bold")

    plt.suptitle("Sample craters by raw cluster (row order/color = aligned class)", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_agreement_examples(df: pd.DataFrame, imgs_dir: str, class_names: list[str],
                            out_path: str, n_per_group: int = 8, seed: int = 0):
    """
    Random craters where the ALIGNED prediction agreed vs. disagreed with the
    true label, titled "true -> predicted" - the direct visual answer to
    "where does the model actually make mistakes." df needs 'true_label',
    'pred_label' (post-alignment, from align.apply_mapping) and 'filename'.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(class_names)
    agree = df[df["true_label"] == df["pred_label"]]
    disagree = df[df["true_label"] != df["pred_label"]]

    def sample_rows(sub, n):
        n = min(n, len(sub))
        return sub.sample(n=n, random_state=int(rng.integers(0, 2**31))) if n > 0 else sub

    agree_sample = sample_rows(agree, n_per_group)
    disagree_sample = sample_rows(disagree, n_per_group)

    fig, axes = plt.subplots(2, n_per_group, figsize=(2.2 * n_per_group, 5.2), squeeze=False)
    for row, (sample, rtitle) in enumerate([(agree_sample, "Agreed"), (disagree_sample, "Disagreed")]):
        for col in range(n_per_group):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(sample):
                continue
            r = sample.iloc[col]
            img_path = os.path.join(imgs_dir, r["filename"])
            if os.path.exists(img_path):
                ax.imshow(_load_crop_for_display(img_path), cmap="gray")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                continue
            ax.set_title(f"{class_names[r['true_label']]} → {class_names[r['pred_label']]}", fontsize=7,
                        color=_class_color(r["true_label"], n_classes))
        fig.text(0.005, 0.75 - row * 0.5, rtitle, rotation=90, va="center", fontsize=12, fontweight="bold")

    plt.suptitle(f"Sample craters: {len(agree)} agreed / {len(disagree)} disagreed (true → predicted)")
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
