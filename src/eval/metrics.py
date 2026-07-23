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
