"""
diagnose_shortcut.py — Phase-0 diagnostics for the DINO brightness-shortcut
hypothesis (see docs/dino_mae_proto_finetune_status.md's "Brightness
collapse" section). No training - just a forward pass through each
checkpoint (latents aren't saved anywhere by evaluate.py, so this
re-embeds). For each (checkpoint, labeled_set), reusing the exact same
embedding/alignment/metric code evaluate.py itself uses:

  1. Full PCA on the latents; Pearson + Spearman correlation between the
     CONTINUOUS PC1 score and per-crater brightness - more precise than
     evaluate.py's own confound_correlation.csv, which only tests brightness
     against the discretized KMeans cluster assignment, not the raw
     projection.
  2. Project PC1 out of the latents entirely (residual = latents -
     outer(pc1_scores, pc1_loading)), re-run the SAME Hungarian-alignment +
     headline metrics (accuracy/macro_f1/qwk/mAP) on the residual, and
     compare against the original (PC1-included) numbers. If metrics hold
     up (or improve) on the residual, the semantic content was never
     destroyed by finetuning, just swamped by PC1 - a whitening problem
     (the existing LDA head in eval/head.py is functionally most of that
     fix already). If metrics collapse on the residual too, PC1 removal
     isn't enough and the damage is more fundamental than one axis.

Usage (one checkpoint/set at a time; see run_diagnostics.sbatch-style
callers for looping over a whole checkpoint trajectory):
    PYTHONPATH=src python -m eval.diagnose_shortcut \\
        --checkpoint <path> --autoencoder-model dino --label "no-proto @199" \\
        --set julie --config configs/eval_suite.yaml
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import yaml

from eval.align import align, apply_mapping
from eval import pipeline as P
from eval import metrics as M
from eval import holdout as H


def _embed_set(spec: dict, checkpoint: str, autoencoder_model: str, cfg: dict):
    """Same embedding dispatch as evaluate.py's evaluate_labeled_set() -
    kept in sync with that function by hand since factoring it out into a
    shared helper would touch evaluate.py's own tested code path."""
    class_names = spec.get("class_names", cfg["class_names"])
    imgs_dir = spec["imgs_dir"]
    run_cfg = {**cfg, "checkpoint": checkpoint, "imgs_dir": imgs_dir,
              "autoencoder_model": autoencoder_model, "class_names": class_names,
              "n_clusters": len(class_names)}

    if "dat_path" in spec:
        id_allowlist = None
        if spec.get("id_split_csv"):
            split_df = pd.read_csv(spec["id_split_csv"], dtype={"crater_id": str})
            id_allowlist = set(split_df.loc[split_df["split"] == spec["id_split_value"], "crater_id"])
        labels = P.labels_from_memmap_csv(spec["metadata_csv"], spec["labels_csv"], class_names,
                                          id_allowlist=id_allowlist)
        latents = H.embed_holdout(spec["dat_path"], checkpoint, autoencoder_model,
                                  row_idx=labels["row_idx"].to_numpy())
    else:
        labels = P.labels_from_png_dir(imgs_dir)
        latents = P.embed(labels, run_cfg)
    return labels, latents, class_names, imgs_dir, run_cfg


def _cluster_and_score(latents: np.ndarray, labels: pd.DataFrame, class_names: list[str],
                       run_cfg: dict, n_boot: int = 500) -> dict:
    """Same Hungarian-alignment + headline-metric recipe evaluate.py's
    evaluate_labeled_set() uses, factored out so it can run twice per
    checkpoint (once on the raw latents, once on the PC1-removed residual)
    without duplicating the KMeans/align/metrics wiring."""
    K = len(class_names)
    clusters, dists, _ = P.kmeans_cluster(latents, {**run_cfg, "n_clusters": K})
    df = P.build_predictions(labels, latents, clusters, dists)
    df.attrs["dist_cols"] = [c for c in df.columns if c.startswith("dist_")]
    mapping, info = align(df, K)
    df = apply_mapping(df, mapping)
    return {
        "purity": info["purity"],
        "accuracy": M.bootstrap_ci(df, K, M.acc_fn, n_boot)["point"],
        "macro_f1": M.bootstrap_ci(df, K, M.macrof1_fn, n_boot)["point"],
        "qwk": M.bootstrap_ci(df, K, M.qwk_fn, n_boot)["point"],
        "map": M.mean_average_precision(df, mapping, K),
    }


def diagnose(checkpoint: str, autoencoder_model: str, set_name: str, label: str,
            config_path: str = "configs/eval_suite.yaml") -> dict:
    cfg = yaml.safe_load(open(config_path))
    spec = cfg["labeled_sets"][set_name]
    labels, latents, class_names, imgs_dir, run_cfg = _embed_set(spec, checkpoint, autoencoder_model, cfg)

    # --- full PCA, continuous PC1 score vs. brightness -----------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, latents.shape[1]), random_state=0)
    scores = pca.fit_transform(latents)
    pc1_scores, pc1_loading = scores[:, 0], pca.components_[0]
    pc1_explained = pca.explained_variance_ratio_[0]

    feats = M.image_intensity_features(labels, imgs_dir)
    merged = labels[["crater_id"]].merge(feats, on="crater_id", how="inner")
    brightness = merged["brightness"].to_numpy()
    from scipy.stats import pearsonr, spearmanr
    pear_r, pear_p = pearsonr(pc1_scores, brightness)
    spear_r, spear_p = spearmanr(pc1_scores, brightness)

    # --- residual after removing PC1, same metric suite ----------------
    residual = latents - np.outer(pc1_scores, pc1_loading)
    original_metrics = _cluster_and_score(latents, labels, class_names, run_cfg)
    residual_metrics = _cluster_and_score(residual, labels, class_names, run_cfg)

    result = {
        "label": label, "checkpoint": checkpoint, "set": set_name,
        "n_samples": len(labels), "pc1_explained_variance": float(pc1_explained),
        "pc1_vs_brightness_pearson_r": float(pear_r), "pc1_vs_brightness_pearson_p": float(pear_p),
        "pc1_vs_brightness_spearman_r": float(spear_r), "pc1_vs_brightness_spearman_p": float(spear_p),
        "original": original_metrics, "residual_pc1_removed": residual_metrics,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--autoencoder-model", required=True,
                    choices=["mae", "cae", "dino", "dino_pretrained", "mae_pretrained"])
    ap.add_argument("--set", required=True, help="labeled_sets key in --config")
    ap.add_argument("--label", default=None, help="free-text label for this checkpoint, for the printed report")
    ap.add_argument("--config", default="configs/eval_suite.yaml")
    ap.add_argument("--out", default=None, help="optional path to append this result as one JSON line")
    args = ap.parse_args()

    label = args.label or args.checkpoint
    result = diagnose(args.checkpoint, args.autoencoder_model, args.set, label, args.config)

    print(f"\n=== {label} — {args.set} (n={result['n_samples']}) ===")
    print(f"PC1 explained variance: {result['pc1_explained_variance']:.3f}")
    print(f"PC1 vs brightness — pearson r={result['pc1_vs_brightness_pearson_r']:.3f} "
         f"(p={result['pc1_vs_brightness_pearson_p']:.2e}), "
         f"spearman r={result['pc1_vs_brightness_spearman_r']:.3f} "
         f"(p={result['pc1_vs_brightness_spearman_p']:.2e})")
    o, r = result["original"], result["residual_pc1_removed"]
    print(f"{'metric':10s} {'original':>10s} {'PC1-removed':>12s}")
    for k in ["purity", "accuracy", "macro_f1", "qwk", "map"]:
        ov = o.get(k); rv = r.get(k)
        ov_s = f"{ov:.3f}" if ov is not None else "—"
        rv_s = f"{rv:.3f}" if rv is not None else "—"
        print(f"{k:10s} {ov_s:>10s} {rv_s:>12s}")

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result, default=float) + "\n")
        print(f"Appended → {args.out}")


if __name__ == "__main__":
    main()
