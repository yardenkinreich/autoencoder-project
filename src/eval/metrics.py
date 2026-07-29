"""
Metrics — one small pure function per checklist item. Each takes aligned
predictions (with true_label + pred_label, optionally probabilities) and
returns plain dicts/arrays so they serialize cleanly and stay testable.

Checklist coverage:
  [x] Confusion matrix
  [x] Per-class precision / recall / F1
  [x] Ordinal: quadratic weighted kappa + ordinal MAE
  [x] Calibration: reliability diagram + ECE   (needs probabilities)
  [x] Bootstrap CIs on key metrics
  [x] Separation: silhouette in latent space + cluster purity
  [x] Generic (non-ordinal) agreement: ARI / NMI / V-measure
  [x] Latent-space structure: Procrustes / distance-correlation vs ground truth
  [x] Continuum evidence: SVM boundary uncertainty
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    cohen_kappa_score, silhouette_score,
)


# ---- confusion + per-class -------------------------------------------------

def confusion(df: pd.DataFrame, n_classes: int) -> np.ndarray:
    return confusion_matrix(df["true_label"], df["pred_label"],
                            labels=list(range(n_classes)))


def per_class(df: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        df["true_label"], df["pred_label"],
        labels=list(range(n_classes)), zero_division=0,
    )
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s})


# ---- ordinal ---------------------------------------------------------------

def quadratic_weighted_kappa(df: pd.DataFrame, n_classes: int) -> float:
    # quadratic weights penalize distant ordinal errors more
    return float(cohen_kappa_score(
        df["true_label"], df["pred_label"],
        labels=list(range(n_classes)), weights="quadratic",
    ))


def ordinal_mae(df: pd.DataFrame) -> float:
    """mean |true - pred| in ordinal-step units. fresh->very-eroded counts big."""
    return float(np.abs(df["true_label"] - df["pred_label"]).mean())


# ---- calibration (probabilities required) ----------------------------------

def _class_probabilities(df: pd.DataFrame, mapping: dict, n_classes: int):
    """
    Convert distance-to-cluster columns into class probabilities, given the
    cluster->class mapping. Softmax over negative distances, then sum the
    probability mass of clusters assigned to each class.
    Returns (P [N,n_classes], confidence [N], pred [N]) or None if no dist cols.
    """
    dist_cols = df.attrs.get("dist_cols", [])
    if not dist_cols:
        return None
    D = df[dist_cols].to_numpy(dtype=float)               # [N, n_clusters]
    # softmax over -distance -> soft cluster assignment
    Z = np.exp(-(D - D.min(axis=1, keepdims=True)))
    Z = Z / Z.sum(axis=1, keepdims=True)
    P = np.zeros((len(df), n_classes))
    for cl_idx, col in enumerate(dist_cols):
        cl_id = int(col.split("_")[1])
        cls = mapping.get(cl_id)
        if cls is not None:
            P[:, cls] += Z[:, cl_idx]
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    return P, conf, pred


def ece(df: pd.DataFrame, mapping: dict, n_classes: int, n_bins: int = 10):
    """Expected Calibration Error + reliability-diagram bins. None if no probs."""
    out = _class_probabilities(df, mapping, n_classes)
    if out is None:
        return None
    _, conf, pred = out
    correct = (pred == df["true_label"].to_numpy()).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows, ece_val = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                         "acc": np.nan, "conf": np.nan})
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        ece_val += (m.sum() / len(conf)) * abs(acc_b - conf_b)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": int(m.sum()),
                     "acc": acc_b, "conf": conf_b})
    return {"ece": float(ece_val), "bins": pd.DataFrame(rows)}


# ---- separation ------------------------------------------------------------

def separation(latents: np.ndarray | None, df: pd.DataFrame, info: dict) -> dict:
    """
    Class separation in feature space. Two complementary numbers:
      silhouette  : geometry of the latent space wrt TRUE labels (needs latents)
      purity      : alignment quality from align.info (always available)
    """
    out = {"purity": info.get("purity")}
    if latents is not None and len(np.unique(df["true_label"])) > 1:
        out["silhouette_true"] = float(silhouette_score(latents, df["true_label"]))
        out["silhouette_cluster"] = float(silhouette_score(latents, df["cluster"]))
    return out


# ---- bootstrap -------------------------------------------------------------

def bootstrap_ci(df: pd.DataFrame, n_classes: int, metric_fn,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """
    Percentile bootstrap CI for any metric_fn(sub_df, n_classes) -> float.
    Resamples craters with replacement, recomputing the metric each time.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    vals = np.empty(n_boot)
    arr = df.reset_index(drop=True)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = metric_fn(arr.iloc[idx], n_classes)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": float(metric_fn(df, n_classes)),
            "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


# convenience scalar wrappers for bootstrap
def acc_fn(df, n_classes):     return float((df["true_label"] == df["pred_label"]).mean())
def qwk_fn(df, n_classes):     return quadratic_weighted_kappa(df, n_classes)
def ordmae_fn(df, n_classes):  return ordinal_mae(df)
def macrof1_fn(df, n_classes): return float(per_class(df, n_classes)["f1"].mean())


# ---- generic (non-ordinal) agreement ----------------------------------------

def clustering_agreement(df: pd.DataFrame) -> dict:
    """
    ARI / NMI / V-measure between true_label and pred_label. These don't
    assume the classes are ordered — a sanity check independent of the
    ordinal framing QWK/ordinal_mae use, from src/test/evaluate.py.
    """
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score, v_measure_score,
    )
    return {
        "ari": float(adjusted_rand_score(df["true_label"], df["pred_label"])),
        "nmi": float(normalized_mutual_info_score(df["true_label"], df["pred_label"])),
        "v_measure": float(v_measure_score(df["true_label"], df["pred_label"])),
    }


# ---- latent-space structure vs ground truth ---------------------------------

def geometry_comparison(latents: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Structural comparison between predicted-class and true-class centroids in
    latent space: does the geometry the model learned actually mirror the
    ground-truth class structure, not just per-crater label agreement?
    From src/test/evaluate.py's compare_cluster_geometry, reworked to group
    by the already-aligned pred_label rather than re-deriving cluster->class
    from a reversed dict (which silently drops clusters in majority-vote
    mode, where multiple raw clusters map to the same class).

    procrustes_disparity : lower = predicted centroid layout is a closer
        (rotation/scale/translation-invariant) match to the true layout.
    distance_correlation : Spearman correlation between pairwise true-class
        distances and pairwise predicted-class distances - e.g. does "Old"
        sit as far from "Semi-New" in the model's space as it should?

    Needs >=2 classes with both true and predicted samples; NaNs otherwise.
    """
    from scipy.spatial import procrustes
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr

    classes = sorted(df["true_label"].unique())
    gt_centroids, pr_centroids = [], []
    for c in classes:
        gt_mask = (df["true_label"] == c).to_numpy()
        pr_mask = (df["pred_label"] == c).to_numpy()
        if gt_mask.sum() == 0 or pr_mask.sum() == 0:
            continue
        gt_centroids.append(latents[gt_mask].mean(axis=0))
        pr_centroids.append(latents[pr_mask].mean(axis=0))

    if len(gt_centroids) < 2:
        return {"procrustes_disparity": float("nan"), "distance_correlation": float("nan")}

    gt_centroids, pr_centroids = np.array(gt_centroids), np.array(pr_centroids)
    _, _, disparity = procrustes(gt_centroids, pr_centroids)

    gt_dists = cdist(gt_centroids, gt_centroids)[np.triu_indices(len(gt_centroids), k=1)]
    pr_dists = cdist(pr_centroids, pr_centroids)[np.triu_indices(len(pr_centroids), k=1)]
    if np.std(gt_dists) == 0 or np.std(pr_dists) == 0:
        corr = float("nan")
    else:
        corr, _ = spearmanr(gt_dists, pr_dists)

    return {"procrustes_disparity": float(disparity), "distance_correlation": float(corr)}


def boundary_uncertainty(latents: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Are the classes cleanly separated, or is this really a continuum? Fits a
    cross-validated SVM to predict true_label from latents, then measures the
    entropy of its class probabilities: low entropy = confident, cleanly
    separated classes; high entropy = classes blur together in latent space
    (evidence degradation state is a continuum, not discrete clusters - a
    real possibility for erosion state). From src/test/evaluate.py.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_predict
    from scipy.stats import entropy

    y = df["true_label"].to_numpy()
    min_class_count = np.min(np.unique(y, return_counts=True)[1])
    cv_folds = min(5, min_class_count)
    if cv_folds < 2:
        return {"boundary_uncertainty": float("nan")}

    clf = SVC(kernel="rbf", probability=True, random_state=42)
    proba = cross_val_predict(clf, latents, y, cv=cv_folds, method="predict_proba")
    return {"boundary_uncertainty": float(np.mean(entropy(proba, axis=1)))}
