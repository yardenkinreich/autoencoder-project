"""
holdout.py — reconstruction-loss and unsupervised clustering-quality scoring
on the held-out UNLABELED test set (configs/holdout_crater_ids.csv,
materialized into actual image data by `snakemake preprocess_holdout_set`).

Unlike pipeline.py's labeled-set eval, there's no ground truth here — these
metrics answer "how good is the model on data it never trained on", not
"does it recover known classes". Works alongside pipeline.py's labeled-set
metrics rather than replacing them.
"""
from __future__ import annotations
import os
import numpy as np


def load_holdout_dat(path: str, num_channels: int = 1, size: int = 128) -> np.memmap:
    """Mirrors src/train/train.py's load_memmap."""
    file_size = os.path.getsize(path)
    total_floats = file_size // 4
    n_pixels = num_channels * size * size
    n = total_floats // n_pixels
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, num_channels, size, size))


def reconstruction_loss(dat_path: str, checkpoint: str, autoencoder_model: str,
                        device: str = "cuda") -> dict:
    """
    Per-crater reconstruction loss on the held-out set. MAE/CAE only — DINO
    doesn't reconstruct images the way these two do (self-distillation, not
    an autoencoder objective), so there's no equivalent number for it.

    MAE is scored one crater at a time (not batched): forward_loss() reduces
    to a scalar over the whole batch internally, and hand-computing a
    per-sample version by calling forward_encoder()/forward_decoder()
    directly would silently skip the bottleneck injection old checkpoints'
    forward() applies (MaskedAutoencoderViTBottleneck overrides forward(),
    not those inner methods) — giving a loss that doesn't match what the
    checkpoint actually does at inference. Going through the model's own
    forward() per-sample is slower but correct regardless of architecture
    variant. CAE has no such bottleneck complexity and stays batched.
    """
    if autoencoder_model not in ("mae", "cae"):
        raise ValueError(
            f"reconstruction_loss only applies to mae/cae (DINO doesn't "
            f"reconstruct images), got {autoencoder_model!r}")

    import torch
    import torch.nn.functional as F
    from src.cluster.cluster import build_model

    data = load_holdout_dat(dat_path)
    model = build_model(autoencoder_model, bottleneck=64, model_path=checkpoint, device=device)
    model.eval()

    losses = []
    with torch.no_grad():
        if autoencoder_model == "cae":
            batch_size = 64
            for i in range(0, len(data), batch_size):
                batch = torch.from_numpy(np.array(data[i:i + batch_size])).to(device)
                recon = model(batch)
                per_sample = F.mse_loss(recon, batch, reduction="none").mean(dim=(1, 2, 3))
                losses.extend(per_sample.cpu().numpy().tolist())
        else:  # mae — one sample at a time, see docstring
            for i in range(len(data)):
                x = torch.from_numpy(np.array(data[i:i + 1])).to(device)
                loss = model(x, mask_ratio=0.75)[0]   # (loss, pred, mask, [cls_bottleneck])
                losses.append(loss.item())

    losses = np.asarray(losses)
    return {
        "mean_loss": float(losses.mean()),
        "std_loss": float(losses.std()),
        "median_loss": float(np.median(losses)),
        "n_samples": len(losses),
    }


def unsupervised_cluster_quality(latents: np.ndarray, n_clusters: int, seed: int = 0) -> dict:
    """
    Clustering quality on the held-out set with NO ground truth available:
    cluster the embeddings, then measure how well-separated the discovered
    clusters are (silhouette on the model's own cluster assignment, not
    against any true label — a signal independent of whether the labeled-set
    classes are ordinal/degradation-state at all). Works for any architecture
    (mae/cae/dino) since it only needs embeddings, not reconstructions.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(latents)
    labels = km.labels_
    return {
        "silhouette": float(silhouette_score(latents, labels)),
        "davies_bouldin": float(davies_bouldin_score(latents, labels)),   # lower = better
        "calinski_harabasz": float(calinski_harabasz_score(latents, labels)),  # higher = better
        "n_clusters": n_clusters,
        "n_samples": len(latents),
    }


def embed_holdout(dat_path: str, checkpoint: str, autoencoder_model: str,
                  device: str = "cuda", batch_size: int = 64) -> np.ndarray:
    """Embeddings for the held-out set — works for mae/cae/dino alike."""
    import torch
    from src.cluster.cluster import build_model, ENCODE_FNS

    data = load_holdout_dat(dat_path)
    model = build_model(autoencoder_model, bottleneck=64, model_path=checkpoint, device=device)
    encode = ENCODE_FNS[autoencoder_model]

    latents = []
    for i in range(0, len(data), batch_size):
        batch = torch.from_numpy(np.array(data[i:i + batch_size])).to(device)
        latents.append(encode(model, batch).cpu().numpy())
    return np.concatenate(latents, axis=0)
