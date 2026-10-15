"""
learning_curve.py — does more training help or hurt? Score every saved
teacher snapshot of a DINO run on a held-out labelled set.

DINOv2's do_test() writes {run}/eval/training_<it>/teacher_checkpoint.pth
every evaluation.eval_period_iterations (200 here) plus eval/final/. For each
snapshot this embeds the labelled set (default: the final_test half of the
baseline labels - never seen by a proto/SupCon run) and records the same
unsupervised numbers the eval suite headlines - k-means (k = #classes,
Hungarian-aligned) accuracy / macro-F1 / QWK / ARI, silhouette of the TRUE
classes - plus 5-fold k-NN accuracy as the supervised ceiling. Light by
design (no bootstrap, no figures per snapshot), so a 15-snapshot run costs
minutes, not 15 full eval suites, and nothing is appended to eval_history.

Usage:
  PYTHONPATH=src:$(pwd) python -m eval.learning_curve --run-dir logs/dino/.../ep30 \
      --set kaguya_tco_raw__asinh_final_test
writes {run}/learning_curve/<set>.csv (+ .png); several --run-dir share one plot
with --plot-out.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

from eval import holdout as H
from eval import metrics as M
from eval import pipeline as P
from eval.align import align, apply_mapping


def snapshots(run_dir: str):
    """[(iteration, checkpoint path)] sorted by iteration; final = last+1."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "eval", "training_*", "teacher_checkpoint.pth")):
        out.append((int(re.search(r"training_(\d+)", p).group(1)) + 1, p))
    out.sort()
    fin = os.path.join(run_dir, "eval", "final", "teacher_checkpoint.pth")
    if os.path.exists(fin):
        out.append((out[-1][0] + 1 if out else 0, fin))
    return out


def score(ckpt: str, spec: dict, cfg: dict, model: str = "dino") -> dict:
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    names = spec["class_names"]
    K = len(names)
    allow = None
    if spec.get("id_split_csv"):
        s = pd.read_csv(spec["id_split_csv"], dtype={"crater_id": str})
        allow = set(s.loc[s["split"] == spec["id_split_value"], "crater_id"])
    labels = P.labels_from_memmap_csv(spec["metadata_csv"], spec["labels_csv"], names,
                                      id_allowlist=allow)
    lat = H.embed_holdout(spec["dat_path"], ckpt, model, row_idx=labels["row_idx"].to_numpy())
    clusters, dists, _ = P.kmeans_cluster(lat, {**cfg, "n_clusters": K, "class_names": names})
    df = P.build_predictions(labels, lat, clusters, dists)
    mapping, _ = align(df, K)
    df = apply_mapping(df, mapping)
    y = df["true_label"].to_numpy()
    knn = cross_val_score(KNeighborsClassifier(5, algorithm="brute"), lat, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    return {"n": len(df), "kmeans_acc": M.acc_fn(df, K), "kmeans_macro_f1": M.macrof1_fn(df, K),
            "kmeans_qwk": M.qwk_fn(df, K), "kmeans_ari": adjusted_rand_score(y, df["cluster"]),
            "silhouette_true": float(silhouette_score(lat, y)), "knn_acc": float(knn)}


def plot(curves: dict, out_png: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["kmeans_acc", "kmeans_macro_f1", "kmeans_qwk", "kmeans_ari", "silhouette_true", "knn_acc"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=False)
    for ax, k in zip(axes.ravel(), keys):
        for name, df in curves.items():
            ax.plot(df["iteration"], df[k], marker="o", ms=3, label=name)
            i = df[k].idxmax()
            ax.plot(df.loc[i, "iteration"], df.loc[i, k], "k*", ms=9)
        ax.set_title(k); ax.set_xlabel("iteration"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + "  (★ = best snapshot)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"  plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", nargs="+", required=True)
    ap.add_argument("--set", required=True, help="labelled set name in --config")
    ap.add_argument("--config", default="configs/eval_suite.yaml")
    ap.add_argument("--every", type=int, default=1, help="use every N-th snapshot")
    ap.add_argument("--plot-out", default=None, help="one combined plot for all runs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    spec = cfg["labeled_sets"][a.set]
    curves = {}
    for rd in a.run_dir:
        snaps = snapshots(rd)
        snaps = snaps[::a.every] if a.every > 1 else snaps
        rows = []
        for it, ck in snaps:
            r = {"iteration": it, "checkpoint": ck, **score(ck, spec, cfg)}
            rows.append(r)
            print(f"  {rd.split('nunknown/')[-1]:28s} it {it:5d}  acc {r['kmeans_acc']:.3f}  "
                  f"mF1 {r['kmeans_macro_f1']:.3f}  qwk {r['kmeans_qwk']:.3f}  ari {r['kmeans_ari']:.3f}  "
                  f"sil {r['silhouette_true']:.3f}  knn {r['knn_acc']:.3f}", flush=True)
        df = pd.DataFrame(rows)
        od = os.path.join(rd, "learning_curve")
        os.makedirs(od, exist_ok=True)
        df.to_csv(os.path.join(od, f"{a.set}.csv"), index=False)
        name = rd.rstrip("/").split("nunknown/")[-1]
        curves[f"{rd.split('kaguya_tco_')[-1].split('/')[0]}/{name}" if "kaguya_tco_" in rd else name] = df
        plot({name: df}, os.path.join(od, f"{a.set}.png"), f"{name} on {a.set}")
    if a.plot_out:
        plot(curves, a.plot_out, f"learning curves on {a.set}")


if __name__ == "__main__":
    main()
