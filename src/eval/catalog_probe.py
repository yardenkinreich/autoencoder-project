"""
catalog_probe.py — score every (catalog x normalization) pair with ONE frozen
encoder, so the data-processing question is decided by measurement rather than
by argument.

THE QUESTION THIS ANSWERS
-------------------------
Six catalogs (2 source mosaics x {raw, sigma-100, sigma-200}) and five
normalizations give 30 ways to present the same craters to a model. Everything
else in this project - which architecture, how long to train, which loss - is
downstream of picking one. Training a model per option is unaffordable; but a
FROZEN off-the-shelf DINOv2 is a fixed measuring instrument, so running it over
each option and probing the embeddings isolates the effect of the data
presentation alone. Nothing about the encoder varies, so a difference in score
is a difference in the data.

This is a RANKING instrument, not an absolute one. A frozen ImageNet encoder is
not the model we will ultimately train, and the option that reads best here is a
hypothesis to train on, not a proven winner. What it reliably catches is the
opposite case: an option that destroys the signal (as the p1/p99 clip does) has
nowhere to hide, because no probe can recover what preprocessing deleted.

WHY A LINEAR PROBE AND CROSS-VALIDATION
---------------------------------------
642 labeled craters, 63% of them one class. A 2-layer MLP on a single split
(what supervised_probe.py does, appropriately, for its purpose) has enough
variance across seeds that a 30-way comparison would mostly measure the split.
So: multinomial logistic regression, stratified 5-fold, repeated over seeds,
and the headline number is MACRO RECALL - accuracy is nearly useless here
because always predicting `v-dr` scores 0.632 accuracy and 0.25 macro recall.

QWK is reported alongside because the classes are ORDINAL: confusing v-fr with
r-fr is a smaller error than confusing it with v-dr, and macro recall cannot
see that distinction while QWK can.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

CLASS_NAMES = ["v-fr", "r-fr", "r-dr", "v-dr"]
DEFAULT_CATALOGS = ["morph_raw", "morph_sigma100", "morph_sigma200",
                    "emp_raw", "emp_sigma100", "emp_sigma200"]
DEFAULT_NORMS = ["global", "global_wide", "zscore", "robust", "asinh"]
OUT_JSON = "logs/data_profiles/catalog_probe.json"


def load_labels(labels_csv: str) -> pd.DataFrame:
    lab = pd.read_csv(labels_csv, dtype={"CRATER_ID": str})
    lab = lab[lab["degree"].isin(CLASS_NAMES)].copy()
    lab["y"] = lab["degree"].map({c: i for i, c in enumerate(CLASS_NAMES)})
    return lab[["CRATER_ID", "degree", "y"]]


def encode(catalog_testset: str, norm: str, labels: pd.DataFrame, *,
           checkpoint: str, resize: int, batch: int, device: str):
    """Frozen-DINOv2 embeddings for a catalog testset under one normalization.

    The catalog's PARENT constants are used (stats_from), never the test set's
    own - fitting normalization on the test set would let the evaluation data
    define its own scaling, which is a leak, and would also make the 30 cells
    incomparable since each would sit on a different scale."""
    import torch
    from src.data.crater_catalog_dataset import CraterCatalog
    from src.models.dino_backbone import (
        load_dino_backbone, encode_stock_pretrained_dino, STOCK_DINO_VIT_KWARGS,
    )

    parent = os.path.dirname(catalog_testset.rstrip("/"))
    ds = CraterCatalog(root=catalog_testset, norm=norm, resize=resize,
                       stats_from=parent)
    ids = ds.index["crater_id"].astype(str).to_numpy()

    model = load_dino_backbone(checkpoint, **STOCK_DINO_VIT_KWARGS).to(device).eval()

    feats, keep_ids = [], []
    for i0 in range(0, len(ds), batch):
        xs = [ds[i][0] for i in range(i0, min(i0 + batch, len(ds)))]
        x = torch.stack(xs).to(device).float()
        if not torch.isfinite(x).all():
            raise ValueError(f"{catalog_testset} norm={norm}: non-finite input tensor")
        feats.append(encode_stock_pretrained_dino(model, x).cpu().numpy())
        keep_ids += list(ids[i0:i0 + len(xs)])
    X = np.concatenate(feats)
    del model
    torch.cuda.empty_cache()

    # Align embeddings to labels by crater id - never by row order. The
    # catalogs no longer all hold identical row counts (crops overlapping a
    # nodata hole are skipped), so positional alignment would silently pair
    # one crater's embedding with another's label.
    order = pd.DataFrame({"CRATER_ID": keep_ids, "row": np.arange(len(keep_ids))})
    m = order.merge(labels, on="CRATER_ID", how="inner")
    return X[m["row"].to_numpy()], m["y"].to_numpy(), m["CRATER_ID"].to_numpy()


def probe(X: np.ndarray, y: np.ndarray, *, folds: int, seeds: int) -> dict:
    """Stratified repeated CV with a linear probe. Returns macro recall (the
    headline), plus accuracy and QWK, each as mean and spread over folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import (accuracy_score, recall_score, cohen_kappa_score,
                                 confusion_matrix)

    accs, recs, qwks = [], [], []
    cm_total = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for seed in range(seeds):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                # class_weight="balanced" because the classes are 44/88/103/407.
                # Without it a linear probe maximizes accuracy by leaning on
                # v-dr, which is exactly the degenerate behaviour macro recall
                # exists to expose.
                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            accs.append(accuracy_score(y[te], p))
            recs.append(recall_score(y[te], p, average="macro", zero_division=0))
            qwks.append(cohen_kappa_score(y[te], p, weights="quadratic"))
            cm_total += confusion_matrix(y[te], p, labels=range(len(CLASS_NAMES)))

    def stat(v):
        v = np.asarray(v, dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std()),
                "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}

    per_class_recall = (np.diag(cm_total) / np.maximum(cm_total.sum(axis=1), 1)).tolist()
    return {"macro_recall": stat(recs), "accuracy": stat(accs), "qwk": stat(qwks),
            "per_class_recall": {c: float(r) for c, r in zip(CLASS_NAMES, per_class_recall)},
            "confusion_matrix": cm_total.tolist(),
            "n_samples": int(len(y)), "n_folds_total": len(recs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogs", nargs="*", default=DEFAULT_CATALOGS)
    ap.add_argument("--catalog-root", default="data/catalogs")
    ap.add_argument("--norms", nargs="*", default=DEFAULT_NORMS)
    ap.add_argument("--checkpoint", default="data/raw/pretrained/dinov2_vits14_pretrain.pth")
    ap.add_argument("--labels-csv", default="configs/liorr_labels.csv")
    ap.add_argument("--resize", type=int, default=224,
                    help="Frozen stock DINOv2 forwards at 224; resizing here rather "
                         "than inside the encoder keeps one interpolation, not two.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  resize={args.resize}  "
          f"{args.folds}-fold x {args.seeds} seeds\n")

    labels = load_labels(args.labels_csv)
    print(f"labels: {len(labels)} craters  "
          f"{dict(labels['degree'].value_counts())}\n")

    # The trivial baseline every cell must beat. Computed LAZILY from the first
    # cell's actual labels, not from the labels file: the file lists 770
    # craters but only the 642 inside the mosaics' +/-60 deg latitude coverage
    # are ever probed, and the majority-class fraction differs between the two
    # (0.668 vs 0.634). A baseline on the wrong denominator flatters or
    # penalizes every cell it is compared against.
    baseline = None

    results, failures = {}, []
    for cat in args.catalogs:
        ts = os.path.join(args.catalog_root, cat, "testset")
        if not os.path.exists(os.path.join(ts, "index.csv")):
            print(f"skip {cat}: no testset/index.csv")
            failures.append({"catalog": cat, "norm": None, "error": "missing testset"})
            continue
        results[cat] = {}
        for norm in args.norms:
            try:
                X, y, _ = encode(ts, norm, labels, checkpoint=args.checkpoint,
                                 resize=args.resize, batch=args.batch, device=device)
                if baseline is None:
                    _, cnt = np.unique(y, return_counts=True)
                    baseline = {"accuracy": float(cnt.max() / cnt.sum()),
                                "macro_recall": 1.0 / len(CLASS_NAMES),
                                "n": int(len(y)),
                                "note": "always predict the majority class, on the "
                                        "craters actually probed"}
                    print(f"majority-class baseline (n={baseline['n']}): "
                          f"acc={baseline['accuracy']:.3f} "
                          f"macro_recall={baseline['macro_recall']:.3f}\n", flush=True)
                r = probe(X, y, folds=args.folds, seeds=args.seeds)
                r["embed_dim"] = int(X.shape[1])
                results[cat][norm] = r
                print(f"{cat:16s} {norm:12s} n={r['n_samples']:3d}  "
                      f"macro_recall={r['macro_recall']['mean']:.4f} "
                      f"[{r['macro_recall']['lo']:.3f},{r['macro_recall']['hi']:.3f}]  "
                      f"acc={r['accuracy']['mean']:.4f}  "
                      f"qwk={r['qwk']['mean']:.4f}", flush=True)
            except Exception as e:
                print(f"{cat:16s} {norm:12s} FAILED: {type(e).__name__}: {e}", flush=True)
                failures.append({"catalog": cat, "norm": norm,
                                 "error": f"{type(e).__name__}: {e}"})

    # Ranking across every cell that ran, by the headline metric.
    flat = [{"catalog": c, "norm": n,
             "macro_recall": r["macro_recall"]["mean"],
             "macro_recall_lo": r["macro_recall"]["lo"],
             "macro_recall_hi": r["macro_recall"]["hi"],
             "accuracy": r["accuracy"]["mean"], "qwk": r["qwk"]["mean"]}
            for c, d in results.items() for n, r in d.items()]
    flat.sort(key=lambda d: -d["macro_recall"])
    if flat:
        print("\n=== ranked by macro recall ===")
        for i, d in enumerate(flat, 1):
            print(f"{i:3d}. {d['catalog']:16s} {d['norm']:12s} "
                  f"{d['macro_recall']:.4f} "
                  f"[{d['macro_recall_lo']:.3f},{d['macro_recall_hi']:.3f}]  "
                  f"qwk={d['qwk']:.4f}")
        print("\nCIs are across CV folds, so they describe split variability, NOT "
              "an independent-sample confidence interval - overlapping intervals "
              "mean 'not separated by this instrument', which for adjacent cells "
              "is the common case.")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump({"results": results, "ranked": flat, "failures": failures,
               "baseline": baseline, "class_names": CLASS_NAMES,
               "config": {"checkpoint": args.checkpoint, "resize": args.resize,
                          "folds": args.folds, "seeds": args.seeds,
                          "probe": "LogisticRegression(class_weight=balanced) on "
                                   "StandardScaler'd frozen DINOv2 CLS embeddings",
                          "encoder": "frozen stock DINOv2 ViT-S/14, L2-normalized CLS"}},
              open(args.out_json, "w"), indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
