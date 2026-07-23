r"""
run_pipeline.py — THE ONLY FILE YOU EDIT to wire in your real repo.

Its job: take Julie's labeled craters, push them through YOUR frozen MAE +
clustering pipeline, and emit predictions.csv in the contract schema. Once this
emits a valid table, the whole eval suite works and never needs touching.

Two functions to fill in (marked TODO). Everything below them is fixed plumbing.

    pixi run python -m eval.run_pipeline \
        --labels data/julie_labels.csv \      # crater_id, true_label
        --config configs/erosion4.yaml \
        --out data/predictions.csv \
        --latents-out data/latents.npy        # optional

julie_labels.csv schema:
    crater_id, true_label        (true_label = 0..K-1, 0 = freshest)
and whatever columns your preprocessing needs to find the image (path, lon/lat...).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import yaml


# ============================================================================
# TODO #1 — preprocess + embed. Replace the body with calls into your repo.
# This is where "check preprocessing" lives: swap steps here, rerun, compare.
# ============================================================================
def embed_craters(labels: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Return latent vectors [N, D], row-aligned to `labels`.

    Wire in your actual code, e.g.:
        from src.data.preprocess import load_and_preprocess   # your fn
        from src.train.mae import load_encoder                 # your fn
        imgs = load_and_preprocess(labels, **cfg["preprocess"])  # [N,1,224,224]
        encoder = load_encoder(cfg["checkpoint"])
        return encoder.embed(imgs)                              # [N, D]
    """
    raise NotImplementedError(
        "Fill embed_craters() with your preprocessing + MAE encoder. "
        "Until then, use --synthetic to smoke-test the harness."
    )


# ============================================================================
# TODO #2 — cluster the latents. Replace with your KMeans / whatever.
# Return (cluster_ids [N], distances [N, n_clusters] or None).
# ============================================================================
def cluster_latents(latents: np.ndarray, cfg: dict):
    """
    e.g.:
        from sklearn.cluster import KMeans
        km = KMeans(cfg["n_clusters"], random_state=cfg["seed"]).fit(latents)
        return km.labels_, km.transform(latents)   # transform = per-centroid dist
    """
    raise NotImplementedError(
        "Fill cluster_latents() with your clustering. Use --synthetic to test."
    )


# ---- fixed plumbing below --------------------------------------------------

def _synthetic(labels, cfg, seed=0):
    """Fabricate separable latents + clusters so the harness can be tested
    before the real pipeline is wired in. NOT for science — only plumbing."""
    rng = np.random.default_rng(seed)
    K = len(cfg["class_names"])
    y = labels["true_label"].to_numpy()
    centers = rng.normal(scale=4.0, size=(K, 8))
    latents = centers[y] + rng.normal(scale=1.5, size=(len(y), 8))  # overlap on purpose
    from sklearn.cluster import KMeans
    n_clusters = cfg.get("n_clusters", K)
    km = KMeans(n_clusters, n_init=10, random_state=seed).fit(latents)
    return latents, km.labels_, km.transform(latents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latents-out", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="bypass TODOs with fake separable data to test harness")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    labels = pd.read_csv(args.labels)
    assert {"crater_id", "true_label"} <= set(labels.columns)

    if args.synthetic:
        latents, clusters, dists = _synthetic(labels, cfg)
    else:
        latents = embed_craters(labels, cfg)
        clusters, dists = cluster_latents(latents, cfg)

    out = pd.DataFrame({
        "crater_id": labels["crater_id"].values,
        "true_label": labels["true_label"].astype(int).values,
        "cluster": np.asarray(clusters).astype(int),
    })
    if dists is not None:
        for j in range(dists.shape[1]):
            out[f"dist_{j}"] = dists[:, j]
    out.to_csv(args.out, index=False)
    if args.latents_out is not None:
        np.save(args.latents_out, latents)
    print(f"Wrote {len(out)} predictions to {args.out} "
          f"({out['cluster'].nunique()} clusters)")


if __name__ == "__main__":
    main()
