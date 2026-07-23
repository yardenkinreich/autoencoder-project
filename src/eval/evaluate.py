r"""
evaluate.py — run the full metric suite on a predictions.csv and write a report.

    pixi run python -m eval.evaluate \
        --predictions data/predictions.csv \
        --config configs/erosion4.yaml \
        --latents data/latents.npy \         # optional, enables silhouette
        --out runs/baseline

Outputs (under --out):
    metrics.json        all scalar metrics + CIs + alignment info
    confusion.csv       confusion matrix
    per_class.csv       precision/recall/f1/support
    reliability.csv     calibration bins (if probabilities present)
    report.md           human-readable summary with caveats

Nothing here knows how craters became clusters — swap your pipeline freely and
rerun. This is the "add to it later" surface: new metric -> new function in
metrics.py -> one line here.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import yaml

from eval.contract import LabelScheme, load_predictions
from eval.align import align, apply_mapping
from eval import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--latents", default=None,
                    help="optional .npy [N, D] row-aligned to predictions.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    scheme = LabelScheme(names=cfg["class_names"])
    df = load_predictions(args.predictions, scheme)
    K = scheme.n_classes

    # 1. align clusters -> classes (the unsupervised step)
    mapping, info = align(df, K)
    df = apply_mapping(df, mapping)

    latents = np.load(args.latents) if args.latents else None
    if latents is not None and len(latents) != len(df):
        raise ValueError(f"latents rows ({len(latents)}) != predictions rows ({len(df)})")

    # 2. metrics
    cm = M.confusion(df, K)
    pc = M.per_class(df, K)
    pc.index = scheme.names
    sep = M.separation(latents, df, info)
    cal = M.ece(df, mapping, K)

    results = {
        "alignment": {**info, "cluster_to_class": mapping},
        "accuracy": M.bootstrap_ci(df, K, M.acc_fn, args.n_boot),
        "macro_f1": M.bootstrap_ci(df, K, M.macrof1_fn, args.n_boot),
        "quadratic_weighted_kappa": M.bootstrap_ci(df, K, M.qwk_fn, args.n_boot),
        "ordinal_mae": M.bootstrap_ci(df, K, M.ordmae_fn, args.n_boot),
        "separation": sep,
        "ece": (cal["ece"] if cal else None),
        "n_samples": len(df),
        "class_names": scheme.names,
    }

    # 3. write artifacts
    pd.DataFrame(cm, index=scheme.names, columns=scheme.names).to_csv(
        os.path.join(args.out, "confusion.csv"))
    pc.to_csv(os.path.join(args.out, "per_class.csv"))
    if cal:
        cal["bins"].to_csv(os.path.join(args.out, "reliability.csv"), index=False)
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"),
              indent=2, default=float)
    _write_report(args.out, results, cm, pc, scheme, cal)
    print(f"\nWrote report to {args.out}/report.md")
    _print_summary(results, info)


def _print_summary(results, info):
    print("\n=== SUMMARY ===")
    print(f"alignment: {info['mode']}  (purity={info['purity']:.3f}, "
          f"n_clusters={info['n_clusters']})")
    for k in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"]:
        v = results[k]
        print(f"{k:28s} {v['point']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
    if results["ece"] is not None:
        print(f"{'ece':28s} {results['ece']:.3f}")


def _write_report(out, r, cm, pc, scheme, cal):
    L = []
    L.append("# MAE degradation-state pipeline — eval report\n")
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
    if r["separation"].get("silhouette_true") is not None:
        L.append(f"- **silhouette (true labels)**: {r['separation']['silhouette_true']:.3f}")
    L.append("\n## Per-class\n")
    L.append(pc.round(3).to_markdown())
    L.append("\n## Confusion matrix (rows=true, cols=pred)\n")
    L.append(pd.DataFrame(cm, index=scheme.names, columns=scheme.names).to_markdown())
    open(os.path.join(out, "report.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
