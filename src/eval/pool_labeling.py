"""Label the 100k-crater pool by a method OTHER than KMeans.

WHY THIS EXISTS
---------------
Every geographic map in this project so far colours the pool by KMeans
cluster id. That answers "how does this embedding space partition itself",
which is an unsupervised question, and it is why the maps have never been
directly readable as degradation: a cluster id is an arbitrary integer with
no relation to v-fr/r-fr/r-dr/v-dr, and the project's own measurements put
the clustering's silhouette near 0.09.

k-NN asks the supervised question instead. It takes the 642 HUMAN-LABELLED
craters, finds each pool crater's nearest labelled neighbours in the frozen
embedding, and votes. The output is a DEGRADATION CLASS, not a cluster id -
so the map is finally in the units the project is actually about, and the
colours can carry the ordinal ramp (fresh -> degraded) rather than arbitrary
categorical hues.

THE TWO THINGS THAT MAKE THIS HONEST
------------------------------------
1. NO CONTAMINATION. The labelled craters are excluded from the catalogs by
   build_catalog (--exclude_crater_ids plus --labels_csv), verified here at
   runtime: a non-empty intersection is a hard error, not a warning. Without
   that, a crater could be its own nearest neighbour and the propagated map
   would be partly a copy of the training labels.

2. THE PROPAGATION IS ONLY AS GOOD AS THE CLASSIFIER. 100k coloured points
   look authoritative whatever their accuracy, so this module refuses to
   write a map without also writing the cross-validated accuracy, macro F1,
   quadratic-weighted kappa and confusion matrix of the very classifier that
   produced it, measured on the labelled set under StratifiedKFold. Read
   knn_eval.json before believing the picture.

Methods are registered in _METHODS so more can be added (a linear head, a
prototype classifier, label spreading) without touching the plumbing.
"""
from __future__ import annotations
import os
import json
import zipfile
import argparse

import numpy as np
import pandas as pd


# ── method registry ─────────────────────────────────────────────────────────
# A method maps (pool_embeddings, labelled_embeddings, labelled_y) -> per-pool
# integer labels + a per-pool confidence in [0,1] + a dict of whatever it wants
# recorded about itself. Adding a method means adding a function here.

def _method_knn(Xp, Xl, yl, *, n_neighbors=5, n_classes=4, **kw):
    """Majority vote among the k nearest LABELLED craters. Brute force: see
    head.cv_knn_eval for why the tree backends are avoided (a threading race
    produced garbage indices twice), and note that spatial trees give no
    speedup above ~15 dimensions anyway - these embeddings are 384-dim."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1,
                               algorithm="brute").fit(Xl, yl)
    pred = knn.predict(Xp)
    # Vote fraction for the winning class. Not a calibrated probability, and
    # deliberately not called one: with k=5 it can only take 6 values.
    proba = knn.predict_proba(Xp)
    conf = proba.max(axis=1)
    return pred.astype(int), conf, {"n_neighbors": int(n_neighbors),
                                    "vote_levels": int(n_neighbors) + 1}


def _method_knn_balanced(Xp, Xl, yl, *, n_neighbors=25, n_classes=4, **kw):
    """k-NN with the TRAINING PRIOR divided out of the vote.

    WHY THIS EXISTS
    ---------------
    Plain majority voting does not merely inherit the label imbalance, it
    amplifies it. Measured on this pool: the labelled set is 6.9% v-fr and
    63.4% v-dr, but the propagated map came out 0.25% v-fr and 86.6% v-dr -
    v-fr shrank 28-fold. A class that is rare in the training set is rare
    among anyone's neighbours, so it almost never wins a vote, and the map
    collapses to one colour that is accurate (because guessing v-dr usually
    is) and useless (because it shows nothing).

    Dividing the vote fractions by the training prior converts "probability
    under the training prior" into "probability under a flat prior", which is
    the quantity you want when the training proportions are an artifact of
    which craters someone chose to label rather than an estimate of the pool.

    weights="distance" and a larger default k are not incidental: with k=5 and
    uniform votes predict_proba takes only 6 distinct values, so dividing by a
    prior mostly rounds back to the same answer. Smooth probabilities are what
    make the correction do anything.

    This trades overall accuracy for balanced accuracy, deliberately. Read
    eval.json's balanced figures, not `accuracy`, when judging it."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1,
                               algorithm="brute", weights="distance").fit(Xl, yl)
    proba = knn.predict_proba(Xp)
    prior = np.bincount(yl, minlength=n_classes).astype(float)
    prior /= prior.sum()
    adj = proba / np.maximum(prior, 1e-12)
    s = adj.sum(axis=1, keepdims=True)
    adj = np.divide(adj, s, out=np.full_like(adj, 1.0 / n_classes), where=s > 0)
    return adj.argmax(1).astype(int), adj.max(axis=1), {
        "n_neighbors": int(n_neighbors), "weights": "distance",
        "prior_corrected": True,
        "training_prior": prior.tolist()}


_METHODS = {"knn": _method_knn, "knn_balanced": _method_knn_balanced}


# ── the eval of the labelling method itself ─────────────────────────────────
def evaluate_method(Xl, yl, method: str, n_classes: int, class_names: list[str],
                    n_folds: int = 5, seed: int = 0, **kw) -> dict:
    """Cross-validated performance ON THE LABELLED SET of the exact method
    being used to paint the pool. This is the number that licenses (or does
    not license) reading the map as degradation."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                                 confusion_matrix, balanced_accuracy_score)
    fn = _METHODS[method]
    counts = np.unique(yl, return_counts=True)[1]
    folds = int(min(n_folds, counts.min()))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(yl), -1, dtype=int)
    for tr, te in skf.split(Xl, yl):
        p, _, _ = fn(Xl[te], Xl[tr], yl[tr], n_classes=n_classes, **kw)
        oof[te] = p
    cm = confusion_matrix(yl, oof, labels=list(range(n_classes)))
    return {
        "method": method,
        "n_labelled": int(len(yl)),
        "n_folds": folds,
        "accuracy": float(accuracy_score(yl, oof)),
        "macro_f1": float(f1_score(yl, oof, average="macro")),
        # With 44/88/103/407, plain accuracy is dominated by the majority class:
        # always answering v-dr scores 0.63 while being useless. Balanced
        # accuracy (mean per-class recall) is the number that can tell a real
        # classifier from that degenerate one.
        "balanced_accuracy": float(balanced_accuracy_score(yl, oof)),
        # quadratic weights: these classes are ORDINAL, so confusing v-fr with
        # v-dr must cost more than confusing v-fr with r-fr. Unweighted kappa
        # would treat those as the same mistake.
        "qwk": float(cohen_kappa_score(yl, oof, weights="quadratic")),
        "per_class_f1": {class_names[i]: float(v) for i, v in
                         enumerate(f1_score(yl, oof, average=None,
                                            labels=list(range(n_classes))))},
        "confusion": cm.tolist(),
        "class_names": class_names,
        "class_support": {class_names[i]: int(c) for i, c in
                          enumerate(np.bincount(yl, minlength=n_classes))},
    }


def covariate_shift(lab_meta: pd.DataFrame, pool_meta: pd.DataFrame) -> dict:
    """How far the LABELLED set is from the pool it is being used to paint.

    This is the caveat that governs every other number in eval.json. The
    cross-validated accuracy is measured on the labelled craters, because they
    are the only ones with ground truth - so it describes the pool only to the
    extent that the pool looks like them. Measured here, it does not: the
    labelled set is smaller-diameter, much higher-latitude, and 74% drawn from
    the middle acquisition band while a quarter of the pool sits in the west
    band, whose illumination azimuth is reversed and which contributes about
    30 labelled craters in total.

    None of that is fixable by a better classifier - it is a sampling fact
    about which craters someone chose to label. It is recorded rather than
    corrected so the headline accuracy is never read as a pool accuracy."""
    def _diam_km(df):
        d = df["diam"].astype(float)
        return d / 1000.0 if d.max() > 100 else d

    def _band(lon):
        lon = np.where(lon > 180, lon - 360, lon)
        return np.where(lon < -90, "west", np.where(lon < 90, "mid", "east"))

    out = {}
    for name, df in (("labelled", lab_meta), ("pool", pool_meta)):
        d = _diam_km(df)
        b = pd.Series(_band(df["lon"].to_numpy())).value_counts(normalize=True)
        out[name] = {
            "n": int(len(df)),
            "diam_km_p10": float(d.quantile(0.10)),
            "diam_km_median": float(d.median()),
            "diam_km_p90": float(d.quantile(0.90)),
            "abs_lat_median": float(df["lat"].abs().median()),
            "abs_lat_p90": float(df["lat"].abs().quantile(0.90)),
            "band_fractions": {k: float(v) for k, v in b.items()},
        }
    wl = out["labelled"]["band_fractions"].get("west", 0.0)
    wp = out["pool"]["band_fractions"].get("west", 0.0)
    out["warnings"] = []
    if out["pool"]["diam_km_p90"] > 1.3 * out["labelled"]["diam_km_p90"]:
        out["warnings"].append(
            f"pool p90 diameter {out['pool']['diam_km_p90']:.2f} km vs labelled "
            f"{out['labelled']['diam_km_p90']:.2f} km - the large craters are "
            f"under-represented in training")
    if abs(out["pool"]["abs_lat_median"] - out["labelled"]["abs_lat_median"]) > 10:
        out["warnings"].append(
            f"median |lat| {out['labelled']['abs_lat_median']:.0f} deg labelled vs "
            f"{out['pool']['abs_lat_median']:.0f} deg pool - different incidence "
            f"angles, the variable this project has found dominates the embedding")
    if wp > 2 * max(wl, 1e-9):
        out["warnings"].append(
            f"west band is {wp:.1%} of the pool but only {wl:.1%} of the labels "
            f"(~{int(wl * out['labelled']['n'])} craters) - and west is the band "
            f"whose illumination azimuth is reversed")
    return out


def class_colors(n_classes: int, palette: str = "viridis") -> list[str]:
    """Colours for ORDINAL classes - they must read as ordered AND be told
    apart at a glance on a grey basemap.

    `_ORDINAL_RAMP` (palette="ordinal") is a single-hue gold->dark-red ramp.
    It encodes the order honestly, but at 4 classes its steps are four browns
    that are genuinely hard to separate against lunar terrain, which is the
    whole job of a 100k-point map.

    viridis (the default) keeps the ordering - its LIGHTNESS is monotone, so
    it still reads fresh->degraded - while moving through yellow/green/blue/
    purple, so the four classes are far apart in hue as well. It is also
    colourblind-safe, unlike a red-brown ramp. The usual objection to
    multi-hue scales is rainbow maps like jet, whose lightness is
    non-monotonic so they invent structure; viridis was built specifically to
    fix that, so it is not the same thing.

    Order: index 0 is the FRESHEST class and gets the bright end - fresh
    craters are the bright-ejecta ones, so the map matches the terrain."""
    if palette == "ordinal":
        from src.eval.visualize import _class_color
        return [_class_color(i, n_classes) for i in range(n_classes)]
    import matplotlib
    cmap = matplotlib.colormaps[palette]
    # Reversed so index 0 (freshest) is the bright end. Endpoints trimmed a
    # little: pure viridis(0.0) is near-black and vanishes on dark terrain.
    pts = np.linspace(0.92, 0.08, n_classes) if n_classes > 1 else [0.5]
    return [matplotlib.colors.to_hex(cmap(float(p))) for p in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="knn", choices=sorted(_METHODS),
                    help="How to label the pool. More can be registered in _METHODS.")
    ap.add_argument("--pool-dat", required=True)
    ap.add_argument("--pool-meta", required=True)
    ap.add_argument("--labelled-dat", required=True)
    ap.add_argument("--labelled-meta", required=True)
    ap.add_argument("--labels-csv", default="configs/liorr_labels.csv")
    # Two separate lists, because they answer different questions and conflating
    # them is a real bug: --source-class-names must MATCH THE LABEL CSV's
    # vocabulary (it is what rows are looked up by), while --class-names only
    # names the groups that come out. With --merge-classes they necessarily
    # differ, and passing the merged names to the CSV lookup matches nothing.
    ap.add_argument("--source-class-names", nargs="+",
                    default=["v-fr", "r-fr", "r-dr", "v-dr"],
                    help="The class names AS SPELLED IN --labels-csv, in ordinal "
                         "order (index 0 = freshest). Do not change these to "
                         "rename outputs - use --class-names for that.")
    ap.add_argument("--class-names", nargs="+", default=None,
                    help="Output group names (default: --source-class-names). "
                         "With --merge-classes, one name per group.")
    ap.add_argument("--checkpoint",
                    default="data/raw/pretrained/dinov2_vits14_pretrain.pth")
    ap.add_argument("--model", default="dino_pretrained")
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--merge-classes", default=None,
                    help="Collapse the ordinal classes before fitting, as "
                         "groups of original indices: '0,1 2,3' merges v-fr+r-fr "
                         "into one class and r-dr+v-dr into another. --class-names "
                         "must then give one name per GROUP. Measured reason to "
                         "want this: at 4 classes the frozen embedding confuses "
                         "r-dr with v-dr on 52 of 103 craters (F1 0.41), while "
                         "fresh-vs-degraded scores 0.958 accuracy / 0.938 macro F1 "
                         "- it resolves the axis, not four levels on it.")
    ap.add_argument("--tag", default=None,
                    help="Output directory suffix (default: the method name). "
                         "Lets two configurations of one method sit side by side.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True, help="Title text for the figures.")
    ap.add_argument("--basemap", default=None,
                    help="Defaults to geographic_map._DEFAULT_BASEMAP.")
    ap.add_argument("--pool-embed-cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-umap", action="store_true",
                    help="skip the full-pool UMAP (a few minutes for ~100k craters)")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--palette", default="viridis",
                    help="Class colours. 'viridis' (default) is ordinal by "
                         "lightness but far apart in hue, so the classes are "
                         "separable on a grey basemap; 'ordinal' restores the "
                         "single-hue gold->dark-red ramp.")
    ap.add_argument("--pca-thumbnails", type=int, default=260,
                    help="How many RANDOM pool craters to draw as thumbnails on "
                         "the PCA. 100k thumbnails is not a figure - they would "
                         "overplot into a grey mat - so a sample is drawn over "
                         "all points shown as dots. 0 skips the figure.")
    args = ap.parse_args()

    from src.eval.holdout import embed_holdout
    from src.eval import geographic_map as GM
    from src.eval.pipeline import labels_from_memmap_csv

    if args.class_names is None:
        args.class_names = list(args.source_class_names)
    n_classes = len(args.class_names)
    tag = args.tag or args.method
    # One directory per labelling method, with the SAME filenames inside each.
    # KMeans is just another method on this axis, so adding a method means the
    # app gains a complete map + exports + figures for it with no new wiring -
    # the dashboard lists whatever `label_*` directories exist rather than
    # knowing any method's name.
    out_dir = os.path.join(args.out_dir, f"label_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # ── labelled set: embeddings + true labels ──────────────────────────────
    # ALWAYS the source vocabulary here - this is a lookup against the CSV.
    lab = labels_from_memmap_csv(args.labelled_meta, args.labels_csv,
                                 args.source_class_names)
    print(f"labelled set: {len(lab):,} craters with a recognized label")
    Xl = embed_holdout(args.labelled_dat, args.checkpoint, args.model,
                       row_idx=lab["row_idx"].to_numpy())
    yl = lab["true_label"].to_numpy().astype(int)

    # Collapse the ordinal scale BEFORE anything is fitted or evaluated, so the
    # reported accuracy is the accuracy of the labelling that actually gets
    # painted - merging afterwards would report a 4-class classifier's number
    # next to a 2-class map.
    merge_spec = None
    if args.merge_classes:
        groups = [[int(i) for i in g.split(",")] for g in args.merge_classes.split()]
        flat = [i for g in groups for i in g]
        if sorted(flat) != list(range(len(args.source_class_names))):
            raise ValueError(f"--merge-classes must use each original index "
                             f"exactly once; got {groups}")
        if len(groups) != len(args.class_names):
            raise ValueError(f"--merge-classes has {len(groups)} groups but "
                             f"--class-names has {len(args.class_names)} names")
        remap = {orig: gi for gi, g in enumerate(groups) for orig in g}
        yl = np.array([remap[v] for v in yl], dtype=int)
        n_classes = len(groups)
        merge_spec = {"groups": groups, "names": list(args.class_names)}
        print(f"merged classes {groups} -> {args.class_names}: "
              f"counts {np.bincount(yl, minlength=n_classes).tolist()}")

    # ── pool: embeddings ────────────────────────────────────────────────────
    pool = pd.read_csv(args.pool_meta, dtype={"id": str})
    cache = args.pool_embed_cache
    if cache and os.path.exists(cache):
        print(f"reusing pool embeddings {cache}")
        Xp = np.load(cache)
    else:
        Xp = embed_holdout(args.pool_dat, args.checkpoint, args.model)
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            np.save(cache, Xp)
    print(f"pool: {Xp.shape}, labelled: {Xl.shape}")
    if Xp.shape[1] != Xl.shape[1]:
        raise ValueError(
            f"embedding dim mismatch: pool {Xp.shape[1]} vs labelled "
            f"{Xl.shape[1]}. Both must come from the same checkpoint AND the "
            f"same normalization, or the neighbours are meaningless.")
    if len(Xp) != len(pool):
        raise ValueError(f"pool metadata has {len(pool)} rows, embedded {len(Xp)}")

    # ── the contamination check, as an ERROR ────────────────────────────────
    lab_ids = set(lab["crater_id"].astype(str))
    pool_ids = set(pool["id"].astype(str))
    overlap = lab_ids & pool_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} labelled craters are ALSO in the pool "
            f"(e.g. {sorted(overlap)[:5]}). A crater would then be its own "
            f"nearest neighbour and the map would partly be a copy of the "
            f"training labels. Rebuild the catalog with --exclude_crater_ids.")
    print(f"contamination check: 0 of {len(lab_ids):,} labelled craters in the pool")

    # ── eval the method BEFORE painting anything ────────────────────────────
    ev = evaluate_method(Xl, yl, args.method, n_classes, args.class_names,
                         seed=args.seed, n_neighbors=args.n_neighbors)
    ev["label"] = args.label
    ev["checkpoint"] = args.checkpoint
    ev["merge"] = merge_spec
    print(f"\n=== {args.method} cross-validated on the labelled set "
          f"({ev['n_folds']} folds) ===")
    print(f"  accuracy {ev['accuracy']:.3f}   macro-F1 {ev['macro_f1']:.3f}   "
          f"QWK {ev['qwk']:.3f}")
    for c, f1 in ev["per_class_f1"].items():
        print(f"    {c:6} F1 {f1:.3f}  (n={ev['class_support'][c]})")

    # ── label the pool ──────────────────────────────────────────────────────
    pred, conf, info = _METHODS[args.method](
        Xp, Xl, yl, n_neighbors=args.n_neighbors, n_classes=n_classes)
    ev["method_params"] = info
    # Recorded next to the accuracy it qualifies, never separately.
    lab_rows = pd.read_csv(args.labelled_meta, dtype={"id": str}).iloc[
        lab["row_idx"].to_numpy()]
    ev["covariate_shift"] = covariate_shift(lab_rows, pool)
    for w in ev["covariate_shift"]["warnings"]:
        print(f"  COVARIATE SHIFT: {w}")
    dist = {args.class_names[i]: int(n) for i, n in
            enumerate(np.bincount(pred, minlength=n_classes))}
    ev["pool_distribution"] = dist
    ev["pool_mean_confidence"] = float(conf.mean())
    print(f"\npool labelled: {dist}")
    print(f"mean vote confidence {conf.mean():.3f}")

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(ev, f, indent=2)

    # `cluster` is the column every downstream consumer (plot_map, write_gpkg,
    # cluster_profile) already expects, so the predicted class is written into
    # it rather than inventing a parallel name that each would need teaching.
    df = pool.assign(cluster=pred,
                     pred_class=[args.class_names[i] for i in pred],
                     confidence=conf)
    csv_path = os.path.join(out_dir, "clusters.csv")
    df.to_csv(csv_path, index=False)

    colors = class_colors(n_classes, palette=args.palette)
    basemap = args.basemap or GM._DEFAULT_BASEMAP
    acc = f"CV accuracy {ev['accuracy']:.2f}, QWK {ev['qwk']:.2f}"
    title = (f"{args.label} — {len(df):,} craters labelled by {args.method} "
             f"from {len(lab):,} human-labelled craters ({acc})")

    png = os.path.join(out_dir, "map.png")
    GM.plot_map(df, n_classes, png, basemap, title=title, colors=colors,
                legend_labels=args.class_names,
                legend_title=f"{args.method} — ordinal, fresh → degraded")
    print(f"  wrote {png}")

    gpkg = os.path.join(out_dir, "clusters.gpkg")
    GM.write_gpkg(df, gpkg, label=args.label, k=n_classes,
                  crs_wkt=GM.raster_crs_wkt(basemap)[0])
    from src.eval.cluster_profile import write_qml
    qml = os.path.join(out_dir, "clusters.qml")
    write_qml(qml, n_classes, colors, labels=args.class_names)
    print(f"  wrote {gpkg} (+ .qml)")

    # ── examples + average crater per predicted class ───────────────────────
    # Same figure the KMeans maps get, so the two methods are read the same
    # way. "closest" here means nearest the class's mean embedding among the
    # craters assigned to it - the pool has no ground truth to rank by.
    try:
        from src.eval.cluster_profile import profile, fig_mean_craters
        prof, means = profile(args.pool_dat, df)
        prof["color_hex"] = [colors[int(c) % len(colors)] for c in prof["cluster"]]
        fig_mean_craters(means, prof, colors,
                         os.path.join(out_dir, "examples.png"), n_classes,
                         dat_path=args.pool_dat,
                         clusters=df["cluster"].to_numpy(),
                         n_examples=args.n_examples)
        prof.to_csv(os.path.join(out_dir, "profile.csv"), index=False)
        print(f"  wrote {os.path.join(out_dir, 'examples.png')}")
    except Exception as e:
        print(f"  WARNING: example figure failed: {type(e).__name__}: {e}")

    # ── PCA of the FULL pool, coloured by the predicted class ───────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        P = PCA(n_components=2, random_state=0).fit(Xp)
        Z = P.transform(Xp)
        try:
            sil = float(silhouette_score(Xp, pred, sample_size=10000,
                                         random_state=0))
        except Exception:
            sil = float("nan")
        fig, ax = plt.subplots(figsize=(7.4, 6.4))
        for i, cn in enumerate(args.class_names):
            q = pred == i
            if not q.any():
                continue
            ax.scatter(Z[q, 0], Z[q, 1], s=2.0, color=colors[i], alpha=0.4,
                       linewidths=0, rasterized=True, label=f"{cn} (n={int(q.sum()):,})")
        ax.legend(fontsize=8, markerscale=4, frameon=False)
        ev_r = P.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev_r[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev_r[1]:.1%})", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.suptitle(f"{args.label} — all {len(Xp):,} pool craters, "
                     f"coloured by {args.method} class\n"
                     f"silhouette {sil:.3f} (full embedding, not these 2 PCs)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pp = os.path.join(out_dir, "pca.png")
        fig.savefig(pp, dpi=150, bbox_inches="tight"); plt.close(fig)
        ev["pool_silhouette"] = sil

        # ── the same PCA, coloured by ACQUISITION BAND ──────────────────────
        # Looking at a cluster-coloured PCA and concluding the bands do not
        # separate is not safe: this is 2 of 384 dimensions. So the band is
        # drawn directly, and - because the picture still cannot settle it -
        # the decisive number is computed instead: how well a linear model
        # recovers the band FROM THE FULL EMBEDDING. Chance is 1/3.
        lon = pool["lon"].to_numpy(dtype=float)
        lon = np.where(lon > 180, lon - 360, lon)
        bnd = np.where(lon < -90, 0, np.where(lon < 90, 1, 2))
        bacc = float("nan")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            sub = np.random.default_rng(0).choice(
                len(Xp), min(20000, len(Xp)), replace=False)
            bacc = float(cross_val_score(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                Xp[sub], bnd[sub], cv=4, scoring="balanced_accuracy").mean())
        except Exception as e:
            print(f"  WARNING: band-predictability test failed: {e}")
        ev["band_predictable_balanced_acc"] = bacc
        ev["band_chance_level"] = 1.0 / 3.0

        bcol = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868"}
        fig, ax = plt.subplots(figsize=(7.4, 6.4))
        for bi, bn in enumerate(("west", "mid", "east")):
            q = bnd == bi
            ax.scatter(Z[q, 0], Z[q, 1], s=2.0, color=bcol[bi], alpha=0.35,
                       linewidths=0, rasterized=True,
                       label=f"{bn} (n={int(q.sum()):,})")
        ax.legend(fontsize=8, markerscale=4, frameon=False)
        ax.set_xlabel(f"PC1 ({ev_r[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev_r[1]:.1%})", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.suptitle(
            f"{args.label} — same PCA, coloured by ACQUISITION BAND\n"
            f"band recoverable from the FULL embedding at balanced accuracy "
            f"{bacc:.3f} (chance 0.333) — these 2 PCs are "
            f"{ev_r[0] + ev_r[1]:.0%} of the variance, so the eye cannot "
            f"settle this either way", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        bp = os.path.join(out_dir, "pca_band.png")
        fig.savefig(bp, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {bp}  (band balanced-acc {bacc:.3f} vs chance 0.333)")

        # ── UMAP of the FULL pool, coloured by the predicted class ──────────
        # The non-linear companion to the PCA above: 2 PCs hold a few percent
        # of the variance, UMAP keeps local neighbourhoods, which is what the
        # k-NN labelling itself relies on.
        if not args.no_umap:
            try:
                import umap
                U = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                              metric="cosine", random_state=0).fit_transform(Xp)
                np.save(os.path.join(out_dir, "umap_coords.npy"), U.astype(np.float32))
                fig, ax = plt.subplots(figsize=(7.4, 6.4))
                for i, cn in enumerate(args.class_names):
                    q = pred == i
                    if q.any():
                        ax.scatter(U[q, 0], U[q, 1], s=1.5, color=colors[i], alpha=0.4,
                                   linewidths=0, rasterized=True,
                                   label=f"{cn} (n={int(q.sum()):,})")
                ax.legend(fontsize=8, markerscale=4, frameon=False)
                ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                fig.suptitle(f"{args.label} — UMAP of all {len(Xp):,} pool craters, "
                             f"coloured by {args.method} class", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, 0.94))
                up = os.path.join(out_dir, "umap.png")
                fig.savefig(up, dpi=150, bbox_inches="tight"); plt.close(fig)
                print(f"  wrote {up}")
            except Exception as e:
                print(f"  WARNING: UMAP failed: {type(e).__name__}: {e}")

        # ── PCA with actual crater thumbnails ──────────────────────────────
        # A coloured dot says which class the model assigned; it cannot say
        # whether that was reasonable. Thumbnails at their own coordinates
        # are the only view where the embedding can be checked by eye.
        try:
            if args.pca_thumbnails <= 0:
                raise RuntimeError("thumbnails disabled")
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            from src.eval.holdout import load_holdout_dat
            dd = load_holdout_dat(args.pool_dat)
            rng = np.random.default_rng(args.seed)
            pick = rng.choice(len(Xp), min(args.pca_thumbnails, len(Xp)),
                              replace=False)
            fig, ax = plt.subplots(figsize=(13, 11))
            ax.scatter(Z[:, 0], Z[:, 1], s=1.0, c="#d8d8d8", linewidths=0,
                       rasterized=True, zorder=1)
            for i in pick:
                im = np.asarray(dd[int(i)])[0]
                ab = AnnotationBbox(
                    OffsetImage(im, zoom=0.30, cmap="gray"),
                    (Z[i, 0], Z[i, 1]), frameon=True, pad=0.08, zorder=3)
                ab.patch.set_edgecolor(colors[int(pred[i])])
                ab.patch.set_linewidth(1.6)
                ax.add_artist(ab)
            ax.set_xlabel(f"PC1 ({ev_r[0]:.1%})", fontsize=9)
            ax.set_ylabel(f"PC2 ({ev_r[1]:.1%})", fontsize=9)
            fig.suptitle(
                f"{args.label} — {len(pick)} random pool craters at their PCA "
                f"coordinates\nborder colour = {args.method} class "
                f"({' / '.join(args.class_names)}), grey dots = all "
                f"{len(Xp):,}", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            ip = os.path.join(out_dir, "pca_images.png")
            fig.savefig(ip, dpi=140, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {ip}")
        except Exception as e:
            print(f"  WARNING: thumbnail PCA failed: {type(e).__name__}: {e}")
        with open(os.path.join(out_dir, "eval.json"), "w") as f:
            json.dump(ev, f, indent=2)
        print(f"  wrote {pp}  (silhouette {sil:.4f})")
    except Exception as e:
        print(f"  WARNING: PCA figure failed: {type(e).__name__}: {e}")

    # One zip so QGIS gets the layer, its style and the provenance together -
    # a .gpkg downloaded without its .qml silently renders in random colours
    # that contradict every figure here.
    zp = os.path.join(out_dir, "qgis.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (gpkg, qml, csv_path, os.path.join(out_dir, "eval.json")):
            if os.path.exists(p):
                z.write(p, os.path.basename(p))
    print(f"  wrote {zp}")
    print("EXIT=0")


if __name__ == "__main__":
    main()
