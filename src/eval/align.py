"""
Cluster -> class alignment.

An unsupervised pipeline gives clusters with arbitrary ids. Before ANY
supervised metric (accuracy, confusion, recall...) can be computed, each
cluster must be mapped to a ground-truth class. This is the step people most
often get subtly wrong, so it is isolated here and made explicit.

Two regimes:

1. n_clusters == n_classes  -> one-to-one Hungarian assignment (optimal
   bijection maximizing agreement). This is the honest "did the pipeline
   recover the classes" question.

2. n_clusters >  n_classes  -> many-to-one majority vote (each cluster takes
   the label of its most common true class). This flatters the pipeline (more
   clusters = more freedom to fit), so the harness WARNS and reports both the
   cluster count and a purity number, so reviewers can't be misled.

The mapping is FIT on the labeled set. That is legitimate here because Julie's
set is the evaluation set and the pipeline never saw the labels during training
— but it means these numbers describe "how alignable is the latent structure",
not out-of-sample classification. evaluate.py prints this caveat.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def contingency(df: pd.DataFrame, n_classes: int) -> np.ndarray:
    """rows = clusters (sorted), cols = true classes. M[i,j] = count."""
    clusters = np.sort(df["cluster"].unique())
    idx = {c: i for i, c in enumerate(clusters)}
    M = np.zeros((len(clusters), n_classes), dtype=int)
    for cl, tl in zip(df["cluster"], df["true_label"]):
        M[idx[cl], tl] += 1
    return M, clusters


def align(df: pd.DataFrame, n_classes: int) -> tuple[dict, dict]:
    """
    Returns (cluster_to_class, info).
      cluster_to_class : {raw_cluster_id -> class_idx}
      info             : diagnostics dict (mode, purity, n_clusters)
    """
    M, clusters = contingency(df, n_classes)
    n_clusters = len(clusters)
    total = M.sum()

    if n_clusters == n_classes:
        # optimal one-to-one: maximize agreement == minimize -agreement
        row, col = linear_sum_assignment(-M)
        mapping = {int(clusters[r]): int(c) for r, c in zip(row, col)}
        agree = M[row, col].sum()
        info = {"mode": "hungarian_1to1", "purity": agree / total,
                "n_clusters": n_clusters}
    else:
        # majority vote, many-to-one
        assigned = M.argmax(axis=1)
        mapping = {int(clusters[i]): int(assigned[i]) for i in range(n_clusters)}
        purity = M.max(axis=1).sum() / total
        if n_clusters > n_classes:
            warnings.warn(
                f"n_clusters ({n_clusters}) > n_classes ({n_classes}): using "
                f"majority-vote alignment. Purity ({purity:.3f}) is optimistic "
                f"— more clusters can only raise it. Report n_clusters alongside.",
                stacklevel=2,
            )
        info = {"mode": "majority_vote", "purity": purity,
                "n_clusters": n_clusters}

    return mapping, info


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = df.copy()
    out["pred_label"] = out["cluster"].map(mapping).astype(int)
    return out
