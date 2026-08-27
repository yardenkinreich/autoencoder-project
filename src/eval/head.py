"""
head.py — cross-validated "frozen backbone + lightweight head" evaluation.

Meta's own DINOv2 guidance (MODEL_CARD.md / README.md): the frozen backbone
already produces robust, linearly-separable features, and the intended way
to adapt it to a new domain is a small, separately-trained head on top -
not continued self-distillation training of the backbone itself (see
configs/dino_craters_finetune.yaml for that path, which this complements
rather than replaces). This module answers a narrower question: starting
from a checkpoint's raw latents (frozen, whatever architecture produced
them), does projecting through a simple supervised head before clustering
improve class separation, in a way that actually generalizes?

align.py's own docstring already flags why this needs its own CV harness
rather than reusing evaluate_labeled_set()'s standard path as-is: fitting
the cluster->class mapping on the same data being scored measures
"alignability", not out-of-sample classification. cv_cluster_eval() below
fits the head AND the alignment on each fold's train split only, then
scores the held-out split - a stricter, genuinely out-of-fold protocol,
applied identically whether transform_fn is given (head) or not (raw
latents), so a head-vs-no-head comparison isn't confounded by the two
being scored under different rigor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from eval.align import align


def fit_lda_head(latents_train: np.ndarray, y_train: np.ndarray, n_components: int | None = None):
    """The head Meta's own guidance describes: "a simple, untrained linear
    layer" - LDA is the closed-form version of that (no gradient training,
    no extra hyperparameters to tune, deterministic), which matters given
    how small these labeled sets are (~150-650 craters) - a gradient-trained
    MLP head risks overfitting exactly the same way continuing to train the
    backbone does. Capped at n_classes - 1 components (LDA's own ceiling)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    max_components = len(np.unique(y_train)) - 1
    n_components = min(n_components or max_components, max_components)
    return LinearDiscriminantAnalysis(n_components=n_components).fit(latents_train, y_train)


def cv_cluster_eval(latents: np.ndarray, df: pd.DataFrame, n_classes: int,
                    transform_fn=None, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold cluster predictions for every sample. Each fold: fit
    transform_fn (if given) and KMeans on the TRAIN split only, align
    clusters to classes using the TRAIN split's own labels only, then
    predict + apply that fold's mapping to the held-out split - the held-out
    predictions never influence how the transform/clustering/alignment
    were fit.

    transform_fn(latents_train, y_train) -> fitted object with .transform()
    - None evaluates raw latents under the same protocol (the fair "no head"
    baseline for comparison, not just the existing in-sample pipeline's
    numbers elsewhere in this project).

    Returns crater_id/true_label/cluster/pred_label - no dist_* columns
    (see module docstring: cluster ids aren't comparable across folds, so
    mAP/ECE - which need a single global cluster->class mapping - don't
    apply here; use accuracy/macro_f1/qwk/ordinal_mae/clustering_agreement
    instead, which only need true_label/pred_label)."""
    y = df["true_label"].to_numpy()
    crater_ids = df["crater_id"].to_numpy()
    min_class_count = np.min(np.unique(y, return_counts=True)[1])
    folds = min(n_folds, min_class_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    rows = []
    for train_idx, test_idx in skf.split(latents, y):
        lat_train, lat_test = latents[train_idx], latents[test_idx]
        y_train = y[train_idx]

        if transform_fn is not None:
            transformer = transform_fn(lat_train, y_train)
            lat_train_t = transformer.transform(lat_train)
            lat_test_t = transformer.transform(lat_test)
        else:
            lat_train_t, lat_test_t = lat_train, lat_test

        km = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit(lat_train_t)
        mapping, _ = align(pd.DataFrame({"cluster": km.labels_, "true_label": y_train}), n_classes)

        test_clusters = km.predict(lat_test_t)
        for i, idx in enumerate(test_idx):
            cluster = int(test_clusters[i])
            rows.append({
                "crater_id": crater_ids[idx],
                "true_label": int(y[idx]),
                "cluster": cluster,
                "pred_label": int(mapping.get(cluster, -1)),
            })
    return pd.DataFrame(rows)


def cv_metric_suite(oof_df: pd.DataFrame, n_classes: int, n_boot: int = 2000) -> dict:
    """The subset of evaluate_labeled_set()'s metric suite that only needs
    true_label/pred_label (see cv_cluster_eval's docstring for why mAP/ECE
    are excluded here)."""
    from eval import metrics as M

    return {
        "n_samples": len(oof_df),
        "accuracy": M.bootstrap_ci(oof_df, n_classes, M.acc_fn, n_boot),
        "macro_f1": M.bootstrap_ci(oof_df, n_classes, M.macrof1_fn, n_boot),
        "quadratic_weighted_kappa": M.bootstrap_ci(oof_df, n_classes, M.qwk_fn, n_boot),
        "ordinal_mae": M.bootstrap_ci(oof_df, n_classes, M.ordmae_fn, n_boot),
        "agreement": M.clustering_agreement(oof_df),
    }


def cv_knn_eval(latents: np.ndarray, df: pd.DataFrame, n_classes: int,
                k: int = 5, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold k-NN predictions - Meta's OTHER standard frozen-feature
    eval protocol (alongside the linear/LDA head above). Philosophically
    different from cv_cluster_eval, not just a different transform: k-NN is
    a directly supervised classifier (majority vote among a held-out
    crater's k nearest TRAIN-fold neighbors in the raw embedding space) -
    it never goes through KMeans + alignment, so it answers "how good are
    these frozen features for classification", not this project's usual
    "how naturally separated are the classes in this space" framing.
    No fitting step beyond storing the train fold's embeddings - unlike
    LDA, there's nothing to solve for."""
    from sklearn.neighbors import KNeighborsClassifier

    y = df["true_label"].to_numpy()
    crater_ids = df["crater_id"].to_numpy()
    min_class_count = np.min(np.unique(y, return_counts=True)[1])
    folds = min(n_folds, min_class_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    rows = []
    for train_idx, test_idx in skf.split(latents, y):
        k_fold = min(k, len(train_idx))
        # n_jobs=1: seen a garbage-index IndexError out of predict_proba's
        # compiled backend on new_set_final_test (321 samples, under a
        # multi-CPU SLURM allocation) - a classic symptom of a joblib/BLAS
        # threading race in sklearn's KDTree/BallTree query, not a logic bug
        # here (min_class_count/folds/k_fold bookkeeping above all check
        # out). Single-threaded CV over a few hundred samples costs nothing.
        knn = KNeighborsClassifier(n_neighbors=k_fold, n_jobs=1).fit(latents[train_idx], y[train_idx])
        preds = knn.predict(latents[test_idx])
        for i, idx in enumerate(test_idx):
            rows.append({
                "crater_id": crater_ids[idx],
                "true_label": int(y[idx]),
                "pred_label": int(preds[i]),
            })
    return pd.DataFrame(rows)


def evaluate_head_vs_raw(latents: np.ndarray, df: pd.DataFrame, n_classes: int,
                         head_name: str = "lda", knn_k: int = 5, n_folds: int = 5,
                         n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Three-way comparison under the identical held-out CV protocol: raw
    latents + KMeans (this project's usual "alignability" framing, but done
    properly out-of-fold here), an LDA head + KMeans (frozen features
    reprojected into a task-adapted subspace before clustering), and k-NN
    (a directly supervised classifier baseline, no clustering step at all).
    All three answer "does adapting frozen features help", just via
    different mechanisms - see cv_cluster_eval/cv_knn_eval's docstrings for
    what distinguishes them.

    None if the rarest class has fewer than 2 members - same precondition
    metrics.boundary_uncertainty() already uses for its own CV, since
    StratifiedKFold needs at least n_splits examples of every class."""
    y = df["true_label"].to_numpy()
    min_class_count = int(np.min(np.unique(y, return_counts=True)[1]))
    if min_class_count < 2:
        return None

    head_fns = {"lda": fit_lda_head}
    if head_name not in head_fns:
        raise ValueError(f"unknown head {head_name!r}, choose from {list(head_fns)}")

    raw_oof = cv_cluster_eval(latents, df, n_classes, transform_fn=None, n_folds=n_folds, seed=seed)
    head_oof = cv_cluster_eval(latents, df, n_classes, transform_fn=head_fns[head_name],
                               n_folds=n_folds, seed=seed)
    knn_oof = cv_knn_eval(latents, df, n_classes, k=knn_k, n_folds=n_folds, seed=seed)
    return {
        "head": head_name,
        "knn_k": knn_k,
        "n_folds": min(n_folds, int(np.min(np.unique(df["true_label"], return_counts=True)[1]))),
        "raw_cv": cv_metric_suite(raw_oof, n_classes, n_boot),
        f"{head_name}_head_cv": cv_metric_suite(head_oof, n_classes, n_boot),
        "knn_cv": cv_metric_suite(knn_oof, n_classes, n_boot),
    }


def summarize_to_frame(head_cmp: dict) -> pd.DataFrame:
    """Flatten evaluate_head_vs_raw()'s output to one row per method (raw/
    head/knn), point estimate + CI per metric - for the CSV artifact and
    dashboard table, mirroring per_class.csv/confound_correlation.csv's
    flat-table convention elsewhere in this project."""
    rows = []
    for method_key, suite in head_cmp.items():
        if not isinstance(suite, dict) or "accuracy" not in suite:
            continue
        row = {"method": method_key, "n_samples": suite["n_samples"]}
        for metric in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"]:
            ci = suite[metric]
            row[metric] = ci["point"]
            row[f"{metric}_lo"] = ci["lo"]
            row[f"{metric}_hi"] = ci["hi"]
        row.update({f"agreement_{k}": v for k, v in suite["agreement"].items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")
