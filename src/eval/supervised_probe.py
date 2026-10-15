"""
supervised_probe.py — a small MLP trained directly on the actual expert
degradation labels, on top of a FROZEN checkpoint's CLS-token embeddings.
"Way forward" item 1 from the DIET-CP/crop-scale investigation: every
result so far (Hungarian-matched k-means, proto loss, DIET-CP) measures
whether an UNSUPERVISED representation happens to organize itself around
the four degradation classes, which is strictly harder than "does the
representation contain that information at all" - a representation can
score badly on Hungarian-matched clustering while still linearly/near-
linearly containing the class signal, if the dominant axis it clusters
around isn't the same axis a supervised readout would use. This probe
isolates that: no backbone training, no clustering, no alignment step -
just "how good is this checkpoint's frozen embedding for this task, when
something is explicitly told what the classes are." A strict upper bound
on what an unsupervised evaluation of the same checkpoint could ever show.

Uses new_test_set_split.csv's train_pool half (321 craters) as supervised
training data and the disjoint final_test half (321 craters, never seen
by this probe or by anything else that reports against "new_set_final_test")
for evaluation - the same leak-safe split proto loss and DIET-CP already
use, so results sit directly next to every other row already in the
dashboard/eval_history.csv for comparison.

train_pool is further split 80/20 (fixed seed) purely for early stopping
on the probe head - 321 labeled examples is small enough that reporting
the final-epoch model without any held-out signal risks reporting an
overfit number. final_test is never touched until the one, final
evaluation.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.eval.diagnose_shortcut import _embed_set
from src.eval import metrics as M
from src.eval import visualize as V
from src.eval import history as HIST
from src.eval.evaluate import _write_summary_md
from src.run_layout import default_eval_out

import yaml


def _spec(cfg: dict, split_value: str) -> dict:
    base = cfg["labeled_sets"]["new_set_final_test"]
    return {**base, "id_split_value": split_value}


def build_probe(embed_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3) -> nn.Module:
    return nn.Sequential(
        nn.Linear(embed_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


def train_probe(X_train, y_train, X_val, y_val, n_classes, epochs=300, lr=1e-3,
                weight_decay=1e-2, label_smoothing=0.1, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = build_probe(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train, dtype=torch.long, device=device)
    Xva = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_val, dtype=torch.long, device=device)

    best_val_acc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = nn.functional.cross_entropy(logits, ytr, label_smoothing=label_smoothing)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(Xva).argmax(dim=1) == yva).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)
    return model, best_val_acc, best_epoch


def _supervised_map_and_ece(P: np.ndarray, y_true: np.ndarray, n_classes: int, n_bins: int = 10):
    """mean_average_precision()/ece()'s own logic (metrics.py), but reading
    directly from the probe's real softmax output P instead of reverse-
    engineering a probability from cluster-distance columns via
    _class_probabilities() - there's no clustering/mapping step here to
    reuse that machinery from, and the probe's own P is a more direct
    signal than the distance-derived one those functions were built for."""
    from sklearn.metrics import average_precision_score
    ap_per_class = []
    for c in range(n_classes):
        y_true_c = (y_true == c).astype(int)
        if y_true_c.sum() == 0:
            continue
        ap_per_class.append(average_precision_score(y_true_c, P[:, c]))
    map_score = float(np.mean(ap_per_class)) if ap_per_class else None

    conf, pred = P.max(axis=1), P.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows, ece_val = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "acc": np.nan, "conf": np.nan})
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        ece_val += (m.sum() / len(conf)) * abs(acc_b - conf_b)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": int(m.sum()), "acc": acc_b, "conf": conf_b})
    return map_score, {"ece": float(ece_val), "bins": pd.DataFrame(rows)}


def evaluate_labeled_set_supervised(checkpoint: str, autoencoder_model: str, cfg: dict,
                                    out_dir: str, probe_args: argparse.Namespace,
                                    n_boot: int = 2000) -> dict:
    """Full metric suite for the supervised probe on new_set_final_test,
    matching evaluate.evaluate_labeled_set()'s output CONTRACT (same
    metrics.json/confusion.csv/per_class.csv/confusion_matrix.png/
    latent_separation.png, same results dict keys where they apply) so this
    slots into the SAME dashboard/eval_history.csv path every other run
    uses. Deliberately does NOT compute confound_correlation, geometry,
    geo_correlation, boundary_uncertainty, separation, or head_comparison -
    those are all diagnostics about whether an UNSUPERVISED cluster
    assignment tracks a confound or true structure; there is no cluster
    here, the probe predicts real classes directly, so those questions
    don't apply the way they do for every other row in eval_history.csv.
    julie is not evaluated here: its label scheme (3 classes) doesn't match
    this probe's fixed 4-class output head, and new_set (642, includes
    train_pool) would leak the 321 craters this probe trained on directly -
    new_set_final_test is the only leak-safe, scheme-compatible set.
    """
    os.makedirs(out_dir, exist_ok=True)
    class_names = cfg["labeled_sets"]["new_set_final_test"]["class_names"]
    n_classes = len(class_names)
    imgs_dir = cfg["labeled_sets"]["new_set_final_test"]["imgs_dir"]

    train_labels, train_latents, _, _, _ = _embed_set(
        _spec(cfg, "train_pool"), checkpoint, autoencoder_model, cfg)
    test_labels, test_latents, _, _, _ = _embed_set(
        _spec(cfg, "final_test"), checkpoint, autoencoder_model, cfg)

    scaler = StandardScaler().fit(train_latents)
    X_train_full = scaler.transform(train_latents)
    X_test = scaler.transform(test_latents)
    y_train_full = train_labels["true_label"].to_numpy()
    y_test = test_labels["true_label"].to_numpy()

    rng = np.random.RandomState(probe_args.seed)
    perm = rng.permutation(len(X_train_full))
    n_val = max(1, int(round(probe_args.val_frac * len(perm))))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model, best_val_acc, best_epoch = train_probe(
        X_train_full[train_idx], y_train_full[train_idx],
        X_train_full[val_idx], y_train_full[val_idx],
        n_classes, epochs=probe_args.epochs, lr=probe_args.lr,
        weight_decay=probe_args.weight_decay, label_smoothing=probe_args.label_smoothing,
        seed=probe_args.seed)
    print(f"internal early-stopping: best val acc {best_val_acc:.3f} at epoch {best_epoch}/{probe_args.epochs}")

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.as_tensor(X_test, dtype=torch.float32))
        P_test = torch.softmax(test_logits, dim=1).numpy()
    test_pred = P_test.argmax(axis=1)

    df = pd.DataFrame({"crater_id": test_labels["crater_id"].to_numpy(),
                       "true_label": y_test, "pred_label": test_pred})

    cm = M.confusion(df, n_classes)
    pc = M.per_class(df, n_classes)
    pc.index = class_names
    map_score, cal = _supervised_map_and_ece(P_test, y_test, n_classes)
    agree = M.clustering_agreement(df)

    results = {
        "set_name": "new_set_final_test",
        "n_samples": len(df),
        "class_names": class_names,
        "n_train": len(train_idx), "n_val": len(val_idx), "internal_val_acc": best_val_acc,
        "accuracy": M.bootstrap_ci(df, n_classes, M.acc_fn, n_boot),
        "macro_f1": M.bootstrap_ci(df, n_classes, M.macrof1_fn, n_boot),
        "quadratic_weighted_kappa": M.bootstrap_ci(df, n_classes, M.qwk_fn, n_boot),
        "ordinal_mae": M.bootstrap_ci(df, n_classes, M.ordmae_fn, n_boot),
        "map": {"point": map_score, "lo": None, "hi": None} if map_score is not None else None,
        "ece": cal["ece"],
        "agreement": agree,
    }

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, "confusion.csv"))
    pc.to_csv(os.path.join(out_dir, "per_class.csv"))
    cal["bins"].to_csv(os.path.join(out_dir, "reliability.csv"), index=False)
    json.dump(results, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2, default=float)

    V.plot_confusion_matrix(df, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    V.plot_latent_separation(test_latents, df, class_names,
                             os.path.join(out_dir, "latent_separation.png"), technique="pca")
    V.plot_latent_separation(test_latents, df, class_names,
                             os.path.join(out_dir, "latent_separation_umap.png"), technique="umap")

    L = [f"# Supervised probe report — new_set_final_test\n",
        f"**Samples:** {len(df)} (probe trained on {len(train_idx)}/{len(val_idx)} "
        f"train_pool split, internal val acc {best_val_acc:.3f} at epoch {best_epoch})\n",
        "## Headline metrics (point [95% bootstrap CI])\n"]
    for k in ["accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"]:
        v = results[k]
        L.append(f"- **{k}**: {v['point']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
    if map_score is not None:
        L.append(f"- **map**: {map_score:.3f}")
    L.append(f"- **ECE**: {results['ece']:.3f}")
    L.append(f"- **ARI / NMI / V-measure**: {agree['ari']:.3f} / {agree['nmi']:.3f} / {agree['v_measure']:.3f}")
    L.append("\n## Per-class\n")
    L.append(pc.round(3).to_markdown())
    L.append("\n## Confusion matrix (rows=true, cols=pred)\n")
    L.append(pd.DataFrame(cm, index=class_names, columns=class_names).to_markdown())
    L.append("\nNo confound/geometry/geo-correlation/head-comparison sections here - "
            "those diagnose unsupervised cluster behavior, which doesn't apply to a "
            "direct supervised classifier's predictions. See confusion_matrix.png and "
            "latent_separation.png alongside this report.")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L))

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="data/raw/pretrained/dinov2_vits14_pretrain.pth")
    ap.add_argument("--autoencoder-model", default="dino_pretrained")
    ap.add_argument("--config", default="configs/eval_suite.yaml")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-eval", action="store_true",
                    help="Write the full dashboard-integrated eval (metrics.json, "
                         "confusion_matrix.png, latent_separation.png, report.md, "
                         "summary.json/summary.md nested under the checkpoint's own "
                         "eval_results/ dir, appended to --history) instead of the "
                         "quick JSONL-append mode.")
    ap.add_argument("--tag", default="supervised_probe",
                    help="Stored in summary.json/eval_history.csv and shown in the "
                         "dashboard's run picker - --full-eval mode only.")
    ap.add_argument("--out-dir", default=None,
                    help="--full-eval mode only: overrides the default nested "
                         "eval_results/<run_id> path.")
    ap.add_argument("--history", default="logs/eval_history.csv")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--label", default="stock", help="Quick mode only: JSONL 'label' field")
    ap.add_argument("--out", default=None, help="Quick mode only: JSONL append path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if args.full_eval:
        run_started = datetime.datetime.now()
        run_id = HIST.generate_run_id(args.checkpoint, args.autoencoder_model, run_started)
        out_dir = args.out_dir or default_eval_out(args.checkpoint, args.autoencoder_model, run_id)
        print(f"run_id: {run_id}")
        print(f"out: {out_dir}")

        set_out = os.path.join(out_dir, "labeled", "new_set_final_test")
        results = evaluate_labeled_set_supervised(
            args.checkpoint, args.autoencoder_model, cfg, set_out, args, n_boot=2000)

        all_results = {
            "run_id": run_id, "run_timestamp": run_started.isoformat(timespec="seconds"),
            "checkpoint": args.checkpoint, "autoencoder_model": args.autoencoder_model,
            "tag": args.tag, "eval_kind": "supervised_probe",
            "labeled_sets": {"new_set_final_test": results}, "holdout": None,
        }
        os.makedirs(out_dir, exist_ok=True)
        json.dump(all_results, open(os.path.join(out_dir, "summary.json"), "w"),
                  indent=2, default=float)
        _write_summary_md(out_dir, all_results)
        print(f"\nWrote summary to {out_dir}/summary.json, {out_dir}/summary.md")
        if not args.no_history:
            HIST.append_to_history(all_results, args.history)
        return

    class_names = cfg["labeled_sets"]["new_set_final_test"]["class_names"]
    n_classes = len(class_names)

    train_labels, train_latents, _, _, _ = _embed_set(
        _spec(cfg, "train_pool"), args.checkpoint, args.autoencoder_model, cfg)
    test_labels, test_latents, _, _, _ = _embed_set(
        _spec(cfg, "final_test"), args.checkpoint, args.autoencoder_model, cfg)

    scaler = StandardScaler().fit(train_latents)
    X_train_full = scaler.transform(train_latents)
    X_test = scaler.transform(test_latents)
    y_train_full = train_labels["true_label"].to_numpy()
    y_test = test_labels["true_label"].to_numpy()

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(X_train_full))
    n_val = max(1, int(round(args.val_frac * len(perm))))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model, best_val_acc, best_epoch = train_probe(
        X_train_full[train_idx], y_train_full[train_idx],
        X_train_full[val_idx], y_train_full[val_idx],
        n_classes, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing, seed=args.seed)
    print(f"internal early-stopping: best val acc {best_val_acc:.3f} at epoch {best_epoch}/{args.epochs}")

    model.eval()
    with torch.no_grad():
        test_pred = model(torch.as_tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()

    df = pd.DataFrame({"true_label": y_test, "pred_label": test_pred})
    result = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(y_test),
        "internal_val_acc": best_val_acc,
        "accuracy": M.acc_fn(df, n_classes),
        "macro_f1": M.macrof1_fn(df, n_classes),
        "qwk": M.qwk_fn(df, n_classes),
        "ordinal_mae": M.ordmae_fn(df, n_classes),
    }
    print(json.dumps(result, indent=2))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(M.confusion(df, n_classes), index=class_names, columns=class_names))

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\nAppended -> {args.out}")


if __name__ == "__main__":
    main()
