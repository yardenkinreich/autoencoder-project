r"""
evaluate.py — run the full test suite (all labeled sets + the held-out
unlabeled set) against one trained checkpoint and write a comparison-ready
report.

    PYTHONPATH=src python -m eval.evaluate \
        --checkpoint logs/mae/scratch/none/na/.../ep50/models/autoencoder.pth \
        --autoencoder-model mae \
        --config configs/eval_suite.yaml \
        [--out logs/mae/scratch/none/na/.../ep50/eval_results/mae_baseline]

--out defaults to nesting inside the checkpoint's OWN training run directory
under logs/ - {training_run_dir}/eval_results/<run_id>/ - so a checkpoint
and every evaluation ever run against it live under one directory, not two
disconnected trees (this used to default to runs/eval/<run_id> - both
"logs/" and "runs/" have been consolidated under logs/, since logs/ is the
one the Snakefile itself builds RUN_DIR from). For a checkpoint that isn't
under logs/ at all (a stock/external pretrained checkpoint, never trained by
this pipeline), a stand-in directory is synthesized under logs/ using the
same family/source/... schema real training runs get - see run_layout.py.
run_id is generated fresh each run (architecture + start time + a hash of
the checkpoint path) and stored in summary.json, so a run directory stays
self-identifying even if renamed.

Outputs (under --out):
    summary.json                 every metric from every set, one file to diff
                                  across runs/architectures for comparison
    summary.md                   human-readable version
    labeled/<set_name>/          per-labeled-set: metrics.json, report.md,
                                  confusion.csv, per_class.csv, reliability.csv,
                                  confusion_matrix.png, clusters_by_sample.png
    holdout/                     reconstruction_loss.json (mae/cae only),
                                  cluster_quality.json
    logs/eval_history.csv        one row appended per run (every architecture/
                                  checkpoint you've ever evaluated) so you can
                                  compare against the best-so-far. See
                                  `python -m eval.history --metric <col>`
                                  to list the top runs by any metric.

Nothing here knows how craters became clusters — swap the checkpoint/
architecture freely and rerun. New metric -> new function in metrics.py ->
one line here.
"""
from __future__ import annotations
import argparse, datetime, json, os
import numpy as np
import pandas as pd
import yaml

from run_layout import default_eval_out
from eval.contract import LabelScheme
from eval.align import align, apply_mapping
from eval import head as HEAD
from eval import pipeline as P
from eval import metrics as M
from eval import visualize as V
from eval import holdout as H
from eval import history as HIST


def evaluate_labeled_set(name: str, spec: dict, checkpoint: str,
                         autoencoder_model: str, cfg: dict, out_dir: str,
                         n_boot: int = 2000) -> dict:
    """Full metric suite for one labeled set. Writes artifacts under out_dir.

    spec is this set's own labeled_sets.<name> entry. Two shapes:
      - PNG-based (julie): {imgs_dir}. Loaded via pipeline.labels_from_png_dir
        + pipeline.embed() (per-file PNG reads).
      - memmap-based (e.g. the reviewed new_set): {dat_path, metadata_csv,
        imgs_dir, labels_csv, class_names}. Loaded via
        pipeline.labels_from_memmap_csv + holdout.embed_holdout(row_idx=...)
        — the SAME craters.dat a training run would read, not a PNG re-crop,
        so FOV/scaling exactly matches what the checkpoint trained on. imgs_dir
        here is only used for visualize.py's display crops (.npy) and the
        confound-check's brightness/sobel features, never for embedding.
      Both shapes end up with the same df schema (crater_id/true_label/
      filename[/row_idx]) so everything from here down is shared. class_names
      may be a per-set override (spec) or the top-level default (cfg) — see
      eval_suite.yaml.
    """
    os.makedirs(out_dir, exist_ok=True)
    imgs_dir = spec["imgs_dir"]
    class_names = spec.get("class_names", cfg["class_names"])
    scheme = LabelScheme(names=class_names)
    K = scheme.n_classes

    # n_clusters must match THIS set's K, not the top-level default (e.g.
    # julie's 3 vs. a 4-class set) - align()'s honest one-to-one Hungarian
    # assignment only kicks in when n_clusters == n_classes; leaving a
    # mismatched global n_clusters here would silently fall back to
    # majority-vote alignment (or leave a class permanently unreachable).
    run_cfg = {**cfg, "checkpoint": checkpoint, "imgs_dir": imgs_dir,
              "autoencoder_model": autoencoder_model, "class_names": class_names,
              "n_clusters": K}

    if "dat_path" in spec:
        # id_split_csv/id_split_value: restrict to one split of a
        # crater_id/split CSV (e.g. configs/new_test_set_split.csv's
        # "final_test" rows) - see labels_from_memmap_csv's id_allowlist
        # docstring for why this matters (avoiding leakage against a
        # checkpoint that trained on the OTHER split, e.g. a prototypical-
        # loss finetune's train_pool half).
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
    # None (not bootstrapped) when there are no distance columns to derive
    # class probabilities from - same precondition as ece() above.
    map_ci = (M.bootstrap_ci(df, K, lambda d, k: M.mean_average_precision(d, mapping, k), n_boot)
             if M.mean_average_precision(df, mapping, K) is not None else None)

    # confound check: does the RAW cluster assignment track known
    # non-degradation features (brightness/sobel always; diameter/lat/lon
    # only for craters with a real Robbins ID - e.g. not Julie's set, see
    # robbins_lookup_features' docstring) rather than genuine degradation
    # signal? Kruskal-Wallis + effect size per feature, see metrics.py.
    img_feats = M.image_intensity_features(df, imgs_dir)
    robbins_feats = M.robbins_lookup_features(df)
    feature_df = (img_feats if robbins_feats is None
                 else img_feats.merge(robbins_feats, on="crater_id", how="left"))
    confound = M.confound_correlation(df, feature_df, group_col="cluster")
    # does the embedding ITSELF (not just the discrete cluster grouping
    # above) vary smoothly with geography? Robbins-ID sets only - see
    # latent_geo_correlation()'s docstring.
    geo_corr = M.latent_geo_correlation(latents, df)

    # Frozen-backbone-plus-head comparison: raw latents vs. an LDA head vs.
    # k-NN, all under a stricter genuinely-out-of-fold CV protocol than the
    # in-sample alignment above (see align.py's own docstring on why that's
    # "alignability", not out-of-sample classification, and head.py's
    # module docstring for why this exists - Meta's own DINOv2 guidance is
    # frozen backbone + a lightweight head, not full finetuning). None if
    # the rarest class has under 2 members (can't cross-validate).
    head_cmp = HEAD.evaluate_head_vs_raw(latents, df[["crater_id", "true_label"]], K, n_boot=n_boot)

    results = {
        "set_name": name,
        "n_samples": len(df),
        "class_names": scheme.names,
        "alignment": {**info, "cluster_to_class": mapping},
        "accuracy": M.bootstrap_ci(df, K, M.acc_fn, n_boot),
        "macro_f1": M.bootstrap_ci(df, K, M.macrof1_fn, n_boot),
        "quadratic_weighted_kappa": M.bootstrap_ci(df, K, M.qwk_fn, n_boot),
        "ordinal_mae": M.bootstrap_ci(df, K, M.ordmae_fn, n_boot),
        "map": map_ci,
        "separation": sep,
        "ece": (cal["ece"] if cal else None),
        "agreement": agree,
        "geometry": geom,
        "boundary_uncertainty": boundary,
        "confound_correlation": confound,
        "geo_correlation": geo_corr,
        "head_comparison": head_cmp,
    }

    pd.DataFrame(cm, index=scheme.names, columns=scheme.names).to_csv(
        os.path.join(out_dir, "confusion.csv"))
    pc.to_csv(os.path.join(out_dir, "per_class.csv"))
    pd.DataFrame(confound).T.to_csv(os.path.join(out_dir, "confound_correlation.csv"))
    if geo_corr:
        pd.DataFrame(geo_corr).T.to_csv(os.path.join(out_dir, "geo_correlation.csv"))
    if head_cmp is not None:
        HEAD.summarize_to_frame(head_cmp).to_csv(os.path.join(out_dir, "head_comparison.csv"))
    if cal:
        cal["bins"].to_csv(os.path.join(out_dir, "reliability.csv"), index=False)
    json.dump(results, open(os.path.join(out_dir, "metrics.json"), "w"),
              indent=2, default=float)

    V.plot_confusion_matrix(df, scheme.names, os.path.join(out_dir, "confusion_matrix.png"))
    V.display_craters_by_cluster(df, imgs_dir, os.path.join(out_dir, "clusters_by_sample.png"),
                                 scheme.names, mapping)
    V.plot_latent_separation(latents, df, scheme.names, os.path.join(out_dir, "latent_separation.png"),
                             technique="pca")
    V.plot_latent_separation(latents, df, scheme.names, os.path.join(out_dir, "latent_separation_umap.png"),
                             technique="umap")
    if head_cmp is not None:
        V.plot_lda_head_separation(latents, df, scheme.names,
                                   os.path.join(out_dir, "latent_separation_lda_head.png"))
    V.plot_agreement_examples(df, imgs_dir, scheme.names, os.path.join(out_dir, "agreement_examples.png"))

    # Unsupervised over-clustering exploration: KMeans at k FAR larger than
    # this set's own K classes, purely to look for finer latent structure -
    # no true_label anywhere in these plots (see visualize.py's
    # plot_unsupervised_cluster_dots/_images docstrings). Every labeled set
    # (julie, new_set, new_set_final_test, ...) gets these automatically, for
    # every checkpoint ever evaluated, not just DINO - k values skipped if
    # this set doesn't have enough craters for them.
    for target_k in (10, 50):
        if target_k >= len(df):
            continue
        cluster_labels, _dists, _km = P.kmeans_cluster(latents, {**run_cfg, "n_clusters": target_k})
        for technique in ("pca", "umap"):
            V.plot_unsupervised_cluster_dots(
                latents, cluster_labels, os.path.join(out_dir, f"unsupervised_k{target_k}_{technique}.png"),
                technique=technique, model_name=autoencoder_model)
            V.plot_unsupervised_cluster_images(
                latents, cluster_labels, df, imgs_dir,
                os.path.join(out_dir, f"unsupervised_k{target_k}_{technique}_images.png"),
                technique=technique, model_name=autoencoder_model)

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
    ap.add_argument("--autoencoder-model",
                    choices=["mae", "cae", "dino", "dino_pretrained", "mae_pretrained"],
                    required=True)
    ap.add_argument("--config", default="configs/eval_suite.yaml")
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to nesting inside the checkpoint's "
                         "own training run directory under logs/ - see "
                         "default_eval_out()")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-holdout", action="store_true",
                    help="skip the held-out set (e.g. if preprocess_holdout_set hasn't run yet)")
    ap.add_argument("--history", default="logs/eval_history.csv",
                    help="CSV that every run appends a summary row to, for "
                         "comparing against the best run so far")
    ap.add_argument("--no-history", action="store_true",
                    help="don't append this run to --history")
    ap.add_argument("--tag", default=None,
                    help="free-text label stored in summary.json/eval_history.csv and "
                         "shown in the dashboard's run picker (e.g. --tag lpsc to mark "
                         "runs pulled for a specific presentation/paper) - purely "
                         "cosmetic, doesn't affect anything computed")
    args = ap.parse_args()

    run_started = datetime.datetime.now()
    run_id = HIST.generate_run_id(args.checkpoint, args.autoencoder_model, run_started)
    out = args.out or default_eval_out(args.checkpoint, args.autoencoder_model, run_id)
    print(f"run_id: {run_id}")
    print(f"out: {out}")
    os.makedirs(out, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))

    all_results = {"run_id": run_id,
                   "run_timestamp": run_started.isoformat(timespec="seconds"),
                   "checkpoint": args.checkpoint, "autoencoder_model": args.autoencoder_model,
                   "tag": args.tag, "labeled_sets": {}, "holdout": None}

    labeled_sets = cfg.get("labeled_sets", {})
    from dashboard import model_meta as MM  # deferred - see run_layout.py's own
                                             # same-shaped import for why (avoids a
                                             # module-load-time eval/->dashboard/ dep)
    if MM.extract_model_params(args.checkpoint, args.autoencoder_model).get("proto_loss_enabled"):
        # This checkpoint's finetune used new_test_set_split.csv's "train_pool"
        # half as direct training supervision (src/train/prototypical_loss.py) -
        # scoring it against the FULL new_set would silently include those
        # craters in a number presented as "held-out". Only run sets that stay
        # leak-safe for this checkpoint (new_set_final_test, julie, ...).
        skipped = [n for n in labeled_sets if n == "new_set"]
        labeled_sets = {n: s for n, s in labeled_sets.items() if n != "new_set"}
        if skipped:
            print(f"proto_loss_enabled=true for this checkpoint - skipping {skipped} "
                 f"(leaks train_pool craters; use new_set_final_test instead)")

    for name, spec in labeled_sets.items():
        out_dir = os.path.join(out, "labeled", name)
        print(f"\n=== labeled set: {name} ===")
        results = evaluate_labeled_set(
            name, spec, args.checkpoint, args.autoencoder_model,
            cfg, out_dir, args.n_boot,
        )
        all_results["labeled_sets"][name] = results
        _print_summary(name, results)

    if not args.skip_holdout and "holdout" in cfg:
        print("\n=== held-out unlabeled set ===")
        holdout_out = os.path.join(out, "holdout")
        all_results["holdout"] = evaluate_holdout(
            cfg, args.checkpoint, args.autoencoder_model, holdout_out,
        )
        if all_results["holdout"]["reconstruction_loss"]:
            r = all_results["holdout"]["reconstruction_loss"]
            print(f"reconstruction mean_loss: {r['mean_loss']:.4f} (n={r['n_samples']})")
        q = all_results["holdout"]["cluster_quality"]
        print(f"unsupervised silhouette: {q['silhouette']:.3f}")

    json.dump(all_results, open(os.path.join(out, "summary.json"), "w"),
              indent=2, default=float)
    _write_summary_md(out, all_results)
    print(f"\nWrote summary to {out}/summary.json, {out}/summary.md")

    if not args.no_history:
        HIST.append_to_history(all_results, args.history)


def _print_summary(name, results):
    a = results["alignment"]
    print(f"alignment: {a['mode']}  (purity={a['purity']:.3f}, n_clusters={a['n_clusters']})")
    for k in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae", "map"]:
        v = results[k]
        if v is not None:
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
    for k in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae", "map"]:
        v = r[k]
        if v is not None:
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

    L.append("\n## Confound check — does the cluster track known non-degradation features?\n")
    L.append("Kruskal-Wallis test per feature vs. raw cluster assignment. "
            "p<0.05 AND eta_sq>=0.14 (large effect) flags a likely confound - "
            "the model may be clustering on this instead of degradation state.\n")
    conf_rows = []
    any_flag = False
    for feat, s in r["confound_correlation"].items():
        if s["p_value"] is None:
            continue
        flagged = s["p_value"] < 0.05 and (s["eta_sq"] or 0) >= 0.14
        any_flag = any_flag or flagged
        conf_rows.append({"feature": feat, "H": round(s["h_stat"], 2),
                          "p_value": round(s["p_value"], 4),
                          "eta_sq": round(s["eta_sq"], 3), "flagged": flagged})
    if conf_rows:
        L.append(pd.DataFrame(conf_rows).set_index("feature").to_markdown())
        if any_flag:
            L.append("\n> ⚠️ At least one feature is significantly correlated with "
                     "the clustering (large effect) - check whether the model "
                     "learned a confound instead of degradation signal.")
    else:
        L.append("(no features available - imgs_dir missing, or no crater_id "
                "matched the Robbins ID format for lat/lon/diameter)")

    L.append("\n## Per-class\n")
    L.append(pc.round(3).to_markdown())
    L.append("\n## Confusion matrix (rows=true, cols=pred)\n")
    L.append(pd.DataFrame(cm, index=scheme.names, columns=scheme.names).to_markdown())
    L.append("\nSee confusion_matrix.png, clusters_by_sample.png, "
             "latent_separation.png and agreement_examples.png alongside this report.")
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
