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
    if technique == "pca":
        proj = PCA(n_components=2, random_state=0)
        coords = proj.fit_transform(latents)
        ev = proj.explained_variance_ratio_
        xlabel, ylabel = f"PC1 ({ev[0]*100:.1f}%)", f"PC2 ({ev[1]*100:.1f}%)"
    elif technique == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=0).fit_transform(latents)
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    else:
        raise ValueError(f"unknown technique: {technique}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.get_cmap("tab10")

    for ax, col, title in [(axes[0], "true_label", "True label"),
                           (axes[1], "pred_label", "Predicted (aligned)")]:
        for v in sorted(df[col].unique()):
            mask = (df[col] == v).to_numpy()
            ax.scatter(coords[mask, 0], coords[mask, 1], s=30, alpha=0.7,
                      color=cmap(v % 10), label=class_names[v] if v < len(class_names) else str(v))
        ax.set_title(title)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.suptitle(f"Latent space separation ({technique.upper()})", fontsize=13)
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
                                samples_per_cluster: int = 5, seed: int = 0):
    """
    Grid of sample crater images per RAW cluster (pre-alignment - useful to
    eyeball what a cluster actually looks like before trusting the label
    alignment). df needs 'cluster' and 'filename' columns.
    """
    rng = np.random.default_rng(seed)
    clusters = sorted(df["cluster"].unique())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(n_clusters, samples_per_cluster,
                             figsize=(3 * samples_per_cluster, 3 * n_clusters),
                             squeeze=False)

    for row, cl in enumerate(clusters):
        sub = df[df["cluster"] == cl]
        n_samples = min(samples_per_cluster, len(sub))
        sample = sub.sample(n=n_samples, random_state=int(rng.integers(0, 2**31)))

        for col in range(samples_per_cluster):
            ax = axes[row, col]
            ax.axis("off")
            if col >= n_samples:
                continue
            fname = sample.iloc[col]["filename"]
            img_path = os.path.join(imgs_dir, fname)
            if os.path.exists(img_path):
                ax.imshow(Image.open(img_path).convert("L"), cmap="gray")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            if col == 0:
                ax.set_title(f"cluster {cl} (n={len(sub)})", fontsize=10, loc="left")

    plt.suptitle("Sample craters by raw cluster", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
