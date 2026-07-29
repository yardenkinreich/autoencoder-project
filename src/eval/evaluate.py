r"""
evaluate.py — run the full test suite (all labeled sets + the held-out
unlabeled set) against one trained checkpoint and write a comparison-ready
report.

    PYTHONPATH=src python -m eval.evaluate \
        --checkpoint logs/mae/wac100m_sigma100_noband/.../autoencoder.pth \
        --autoencoder-model mae \
        --config configs/eval_suite.yaml \
        --out runs/eval/mae_baseline

Outputs (under --out):
    summary.json                 every metric from every set, one file to diff
                                  across runs/architectures for comparison
    summary.md                   human-readable version
    labeled/<set_name>/          per-labeled-set: metrics.json, report.md,
                                  confusion.csv, per_class.csv, reliability.csv,
                                  confusion_matrix.png, clusters_by_sample.png
    holdout/                     reconstruction_loss.json (mae/cae only),
                                  cluster_quality.json
    runs/eval_history.csv        one row appended per run (every architecture/
                                  checkpoint you've ever evaluated) so you can
                                  compare against the best-so-far. See
                                  `python -m eval.history --metric <col>`
                                  to list the top runs by any metric.

Nothing here knows how craters became clusters — swap the checkpoint/
architecture freely and rerun. New metric -> new function in metrics.py ->
one line here.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import yaml

from eval.contract import LabelScheme
from eval.align import align, apply_mapping
from eval import pipeline as P
from eval import metrics as M
from eval import visualize as V
from eval import holdout as H
from eval import history as HIST


def evaluate_labeled_set(name: str, imgs_dir: str, checkpoint: str,
                         autoencoder_model: str, cfg: dict, out_dir: str,
                         n_boot: int = 2000) -> dict:
    """Full metric suite for one labeled set. Writes artifacts under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    scheme = LabelScheme(names=cfg["class_names"])
    K = scheme.n_classes

    run_cfg = {**cfg, "checkpoint": checkpoint, "imgs_dir": imgs_dir,
              "autoencoder_model": autoencoder_model}

    labels = P.labels_from_png_dir(imgs_dir)
    latents = P.embed(labels, run_cfg)
    clusters, dists, _ = P.kmeans_cluster(latents, run_cfg)
    df = P.build_predictions(labels, latents, clusters, dists)
    df.attrs["dist_cols"] = [c for c in df.columns if c.startswith("dist_")]

    mapping, info = align(df, K)
    df = apply_mapping(df, mapping)

    cm = M.confusion(df, K)
    pc = M.per_class(df, K)
    pc.index = scheme.names
    sep = M.separation(latents, df, info)
    cal = M.ece(df, mapping, K)
    agree = M.clustering_agreement(df)
    geom = M.geometry_comparison(latents, df)
    boundary = M.boundary_uncertainty(latents, df)

    results = {
        "set_name": name,
        "n_samples": len(df),
        "class_names": scheme.names,
        "alignment": {**info, "cluster_to_class": mapping},
        "accuracy": M.bootstrap_ci(df, K, M.acc_fn, n_boot),
        "macro_f1": M.bootstrap_ci(df, K, M.macrof1_fn, n_boot),
        "quadratic_weighted_kappa": M.bootstrap_ci(df, K, M.qwk_fn, n_boot),
        "ordinal_mae": M.bootstrap_ci(df, K, M.ordmae_fn, n_boot),
        "separation": sep,
        "ece": (cal["ece"] if cal else None),
        "agreement": agree,
        "geometry": geom,
        "boundary_uncertainty": boundary,
    }

    pd.DataFrame(cm, index=scheme.names, columns=scheme.names).to_csv(
        os.path.join(out_dir, "confusion.csv"))
    pc.to_csv(os.path.join(out_dir, "per_class.csv"))
    if cal:
        cal["bins"].to_csv(os.path.join(out_dir, "reliability.csv"), index=False)
    json.dump(results, open(os.path.join(out_dir, "metrics.json"), "w"),
              indent=2, default=float)

    V.plot_confusion_matrix(df, scheme.names, os.path.join(out_dir, "confusion_matrix.png"))
    V.display_craters_by_cluster(df, imgs_dir, os.path.join(out_dir, "clusters_by_sample.png"))
    V.plot_latent_separation(latents, df, scheme.names, os.path.join(out_dir, "latent_separation.png"))
    _write_labeled_report(out_dir, results, cm, pc, scheme, cal)

    return results


def evaluate_holdout(cfg: dict, checkpoint: str, autoencoder_model: str,
                     out_dir: str) -> dict:
    """Reconstruction (mae/cae) + unsupervised clustering quality (any arch)."""
    os.makedirs(out_dir, exist_ok=True)
    dat_path = cfg["holdout"]["dat_path"]
    results = {}

    if autoencoder_model in ("mae", "cae"):
        recon = H.reconstruction_loss(dat_path, checkpoint, autoencoder_model)
        json.dump(recon, open(os.path.join(out_dir, "reconstruction_loss.json"), "w"),
                  indent=2, default=float)
        results["reconstruction_loss"] = recon
    else:
        results["reconstruction_loss"] = None   # DINO doesn't reconstruct images

    latents = H.embed_holdout(dat_path, checkpoint, autoencoder_model)
    quality = H.unsupervised_cluster_quality(latents, cfg["n_clusters"], cfg.get("seed", 0))
    json.dump(quality, open(os.path.join(out_dir, "cluster_quality.json"), "w"),
              indent=2, default=float)
    results["cluster_quality"] = quality

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--autoencoder-model", choices=["mae", "cae", "dino"], required=True)
    ap.add_argument("--config", default="configs/eval_suite.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-holdout", action="store_true",
                    help="skip the held-out set (e.g. if preprocess_holdout_set hasn't run yet)")
    ap.add_argument("--history", default="runs/eval_history.csv",
                    help="CSV that every run appends a summary row to, for "
                         "comparing against the best run so far")
    ap.add_argument("--no-history", action="store_true",
                    help="don't append this run to --history")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))

    all_results = {"checkpoint": args.checkpoint, "autoencoder_model": args.autoencoder_model,
                   "labeled_sets": {}, "holdout": None}

    for name, spec in cfg.get("labeled_sets", {}).items():
        out_dir = os.path.join(args.out, "labeled", name)
        print(f"\n=== labeled set: {name} ===")
        results = evaluate_labeled_set(
            name, spec["imgs_dir"], args.checkpoint, args.autoencoder_model,
            cfg, out_dir, args.n_boot,
        )
        all_results["labeled_sets"][name] = results
        _print_summary(name, results)

    if not args.skip_holdout and "holdout" in cfg:
        print("\n=== held-out unlabeled set ===")
        holdout_out = os.path.join(args.out, "holdout")
        all_results["holdout"] = evaluate_holdout(
            cfg, args.checkpoint, args.autoencoder_model, holdout_out,
        )
        if all_results["holdout"]["reconstruction_loss"]:
            r = all_results["holdout"]["reconstruction_loss"]
            print(f"reconstruction mean_loss: {r['mean_loss']:.4f} (n={r['n_samples']})")
        q = all_results["holdout"]["cluster_quality"]
        print(f"unsupervised silhouette: {q['silhouette']:.3f}")

    json.dump(all_results, open(os.path.join(args.out, "summary.json"), "w"),
              indent=2, default=float)
    _write_summary_md(args.out, all_results)
    print(f"\nWrote summary to {args.out}/summary.json, {args.out}/summary.md")

    if not args.no_history:
        HIST.append_to_history(all_results, args.history)


def _print_summary(name, results):
    a = results["alignment"]
    print(f"alignment: {a['mode']}  (purity={a['purity']:.3f}, n_clusters={a['n_clusters']})")
    for k in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"]:
        v = results[k]
        print(f"{k:28s} {v['point']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
    if results["ece"] is not None:
        print(f"{'ece':28s} {results['ece']:.3f}")
    print(f"{'ari / nmi / v_measure':28s} "
          f"{results['agreement']['ari']:.3f} / {results['agreement']['nmi']:.3f} / "
          f"{results['agreement']['v_measure']:.3f}")


def _write_labeled_report(out, r, cm, pc, scheme, cal):
    L = [f"# Eval report — {r['set_name']}\n"]
    a = r["alignment"]
    L.append(f"**Samples:** {r['n_samples']}  ")
    L.append(f"**Alignment:** `{a['mode']}`, purity {a['purity']:.3f}, "
             f"{a['n_clusters']} clusters mapped to {len(scheme.names)} classes\n")
    if a["mode"] == "majority_vote":
        L.append("> ⚠️ More clusters than classes: purity is optimistic. "
                 "These are *alignability* numbers on the labeled set, not "
                 "out-of-sample classification.\n")
    L.append("## Headline metrics (point [95% bootstrap CI])\n")
    for k in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"]:
        v = r[k]
        L.append(f"- **{k}**: {v['point']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
    if r["ece"] is not None:
        L.append(f"- **ECE**: {r['ece']:.3f}")
    L.append(f"- **ARI / NMI / V-measure**: {r['agreement']['ari']:.3f} / "
             f"{r['agreement']['nmi']:.3f} / {r['agreement']['v_measure']:.3f}")
    if r["separation"].get("silhouette_true") is not None:
        L.append(f"- **silhouette (true labels)**: {r['separation']['silhouette_true']:.3f}")
    L.append(f"- **Procrustes disparity**: {r['geometry']['procrustes_disparity']:.4f}  "
             f"(lower = predicted centroid layout matches true layout better)")
    L.append(f"- **distance correlation**: {r['geometry']['distance_correlation']:.3f}")
    L.append(f"- **boundary uncertainty**: {r['boundary_uncertainty']['boundary_uncertainty']:.3f}  "
             f"(higher = classes blur together more, evidence of a continuum)")
    L.append("\n## Per-class\n")
    L.append(pc.round(3).to_markdown())
    L.append("\n## Confusion matrix (rows=true, cols=pred)\n")
    L.append(pd.DataFrame(cm, index=scheme.names, columns=scheme.names).to_markdown())
    L.append("\nSee confusion_matrix.png, clusters_by_sample.png and "
             "latent_separation.png alongside this report.")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L))


def _write_summary_md(out, all_results):
    L = [f"# Eval summary\n",
         f"**Checkpoint:** `{all_results['checkpoint']}`  ",
         f"**Architecture:** `{all_results['autoencoder_model']}`\n"]
    for name, r in all_results["labeled_sets"].items():
        L.append(f"## {name}\n")
        for k in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"]:
            v = r[k]
            L.append(f"- **{k}**: {v['point']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
        L.append(f"- **ARI / NMI / V-measure**: {r['agreement']['ari']:.3f} / "
                 f"{r['agreement']['nmi']:.3f} / {r['agreement']['v_measure']:.3f}")
        L.append("")
    if all_results["holdout"]:
        L.append("## held-out unlabeled set\n")
        rl = all_results["holdout"]["reconstruction_loss"]
        if rl:
            L.append(f"- **reconstruction mean_loss**: {rl['mean_loss']:.4f} "
                     f"(std {rl['std_loss']:.4f}, n={rl['n_samples']})")
        q = all_results["holdout"]["cluster_quality"]
        L.append(f"- **unsupervised silhouette**: {q['silhouette']:.3f}")
        L.append(f"- **davies_bouldin**: {q['davies_bouldin']:.3f} (lower = better)")
        L.append(f"- **calinski_harabasz**: {q['calinski_harabasz']:.1f} (higher = better)")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
