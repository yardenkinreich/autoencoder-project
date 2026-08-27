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
import os
import re
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


def mean_average_precision(df: pd.DataFrame, mapping: dict, n_classes: int) -> float | None:
    """
    Mean Average Precision: per-class Average Precision (area under the
    precision-recall curve for "is this crater class c?" using that class's
    softmax score from _class_probabilities - the same distance-derived
    scores ece() uses) averaged across classes. Unlike accuracy/F1, this
    scores the RANKING quality of each class's confidence, not just the
    argmax decision - a model that ranks true "Old" craters above non-Old
    ones (even without a perfect threshold) scores well here. None if no
    distance columns are available (same precondition as ece()).
    """
    from sklearn.metrics import average_precision_score

    out = _class_probabilities(df, mapping, n_classes)
    if out is None:
        return None
    P, _, _ = out
    y = df["true_label"].to_numpy()
    ap_per_class = []
    for c in range(n_classes):
        y_true_c = (y == c).astype(int)
        if y_true_c.sum() == 0:
            continue
        ap_per_class.append(average_precision_score(y_true_c, P[:, c]))
    return float(np.mean(ap_per_class)) if ap_per_class else None


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


# ---- confound check: does the cluster track known non-degradation features? ----

def image_intensity_features(df: pd.DataFrame, imgs_dir: str) -> pd.DataFrame:
    """
    Per-crater brightness (mean pixel value), Sobel edge-magnitude (mean
    gradient magnitude - a texture/sharpness proxy), and saturation fraction
    (share of pixels clipped at the top of the 8-bit range - a sign the
    normalization window is too tight for this crater, or it's genuinely
    very bright) computed directly from the PNG. No metadata dependency,
    always available regardless of whether crater_id maps to a real Robbins
    ID.
    """
    import cv2
    rows = []
    for _, r in df.iterrows():
        path = os.path.join(imgs_dir, r["filename"])
        # a memmap-based set's display crop is a saved --save_raw_crops .npy
        # tensor (num_channels, H, W), already float-normalized - not a PNG
        # cv2.imread can read (it silently returns None for those).
        if path.endswith(".npy"):
            img = np.load(path)[0].astype(np.float32) * 255.0 if os.path.exists(path) else None
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append({"crater_id": r["crater_id"], "brightness": np.nan, "sobel_mean": np.nan,
                        "frac_saturated": np.nan})
            continue
        img = img.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(gx ** 2 + gy ** 2)
        rows.append({
            "crater_id": r["crater_id"],
            "brightness": float(img.mean()),
            "sobel_mean": float(sobel_mag.mean()),
            "frac_saturated": float((img >= 254.0).mean()),
        })
    return pd.DataFrame(rows)


_ROBBINS_ID_RE = re.compile(r"^\d{2}-\d-\d{6}$")


def robbins_lookup_features(
    df: pd.DataFrame,
    robbins_csv: str = "data/raw/lunar_crater_database_robbins_2018.csv",
) -> pd.DataFrame | None:
    """
    Diameter/lat/lon looked up from the Robbins catalog, ONLY for craters
    whose crater_id matches Robbins' own ID format (NN-N-NNNNNN). Returns
    None entirely if no crater_id in df matches that format - e.g. Julie's
    set uses sequential filename indices, not real Robbins IDs, so this
    gracefully no-ops there rather than guessing at a mapping that doesn't
    exist (see session notes on the unresolved Julie's-ID-mapping gap).
    """
    ids = df["crater_id"].astype(str)
    matches = ids.str.match(_ROBBINS_ID_RE)
    if not matches.any():
        return None
    robbins = pd.read_csv(
        robbins_csv, usecols=["CRATER_ID", "LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"])
    robbins["CRATER_ID"] = robbins["CRATER_ID"].astype(str)
    out = pd.DataFrame({"crater_id": ids[matches]}).merge(
        robbins, left_on="crater_id", right_on="CRATER_ID", how="left")
    return out.rename(columns={
        "LAT_CIRC_IMG": "lat", "LON_CIRC_IMG": "lon", "DIAM_CIRC_IMG": "diam_km",
    })[["crater_id", "lat", "lon", "diam_km"]]


def latent_geo_correlation(latents: np.ndarray, df: pd.DataFrame,
                           n_components: int = 2) -> dict | None:
    """
    Spearman (rank) correlation between the latent space's PCA components
    and lat/lon, for craters with a real Robbins ID (via
    robbins_lookup_features - None entirely if none match, e.g. Julie's
    set). Spearman, not Pearson, to match this module's existing
    non-parametric convention (confound_correlation's Kruskal-Wallis,
    geometry_comparison's spearmanr) - rank-based, so it doesn't assume a
    linear relationship or normally-distributed features, and is robust to
    the odd outlier crater.

    Complements confound_correlation()'s test (does a feature differ across
    the DISCRETE cluster groups KMeans happened to draw?) with a direct,
    CONTINUOUS one: does the embedding ITSELF vary smoothly with geography,
    independent of where cluster boundaries land? A crater's exact lat/lon
    should carry no degradation signal, so a strong |r| here (especially on
    PC1, the axis latent_separation.png actually plots) is a red flag the
    model may have partly learned "where on the Moon this is" instead of
    "how eroded is this crater."
    """
    from sklearn.decomposition import PCA
    from scipy.stats import spearmanr

    robbins_feats = robbins_lookup_features(df)
    if robbins_feats is None:
        return None

    proj = PCA(n_components=n_components, random_state=0).fit_transform(latents)
    pc_cols = [f"pc{i + 1}" for i in range(n_components)]
    pc_df = pd.DataFrame(proj, columns=pc_cols)
    pc_df["crater_id"] = df["crater_id"].to_numpy()

    merged = pc_df.merge(robbins_feats, on="crater_id", how="inner")
    out = {}
    for pc in pc_cols:
        for geo in ("lat", "lon"):
            sub = merged[[pc, geo]].dropna()
            if len(sub) < 3:
                out[f"{pc}_vs_{geo}"] = {"r": None, "p_value": None, "n": len(sub)}
                continue
            r, p = spearmanr(sub[pc], sub[geo])
            out[f"{pc}_vs_{geo}"] = {"r": float(r), "p_value": float(p), "n": int(len(sub))}
    return out


def confound_correlation(df: pd.DataFrame, feature_df: pd.DataFrame,
                         group_col: str = "cluster") -> dict:
    """
    For each feature column in feature_df (joined to df on crater_id), tests
    whether it differs significantly across cluster groups via Kruskal-Wallis
    (non-parametric one-way ANOVA) + epsilon-squared effect size (bounded
    ANOVA-eta-squared analogue for the H statistic). A significant, large-
    effect result for e.g. diameter or brightness is a red flag: the
    clustering may be tracking that confound instead of genuine degradation
    signal. group_col="cluster" tests the algorithm's own raw grouping;
    pass "pred_label" instead to test the aligned classes.
    """
    from scipy.stats import kruskal

    merged = df[["crater_id", group_col]].merge(feature_df, on="crater_id", how="inner")
    groups = sorted(merged[group_col].unique())
    out = {}
    for col in feature_df.columns:
        if col == "crater_id":
            continue
        vals_by_group = [merged.loc[merged[group_col] == g, col].dropna().to_numpy()
                         for g in groups]
        vals_by_group = [v for v in vals_by_group if len(v) > 0]
        n = sum(len(v) for v in vals_by_group)
        # kruskal() raises outright when every value pooled across groups is
        # identical (e.g. frac_saturated=0.0 for literally every Julie
        # crater - no PNG in that set ever saturates) - there's genuinely no
        # confound signal to report there, not an error condition.
        all_identical = n > 0 and len(np.unique(np.concatenate(vals_by_group))) < 2
        if len(vals_by_group) < 2 or all_identical:
            out[col] = {"h_stat": None, "p_value": None, "eta_sq": None, "n": n}
            continue
        h_stat, p_value = kruskal(*vals_by_group)
        n = sum(len(v) for v in vals_by_group)
        k = len(vals_by_group)
        eta_sq = (h_stat - k + 1) / (n - k) if n > k else float("nan")
        out[col] = {"h_stat": float(h_stat), "p_value": float(p_value),
                    "eta_sq": float(eta_sq), "n": int(n)}
    return out


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
