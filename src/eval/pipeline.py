r"""
pipeline.py — turn Julie's labeled crater PNGs into latents + KMeans clusters.

Wired to the repo's real code (src/models/mae_bottleneck.py): MAE encoder, no
masking, CLS token through the trained clustering bottleneck (same path
src/cluster/cluster.py and src/display/display.py use). This is the half
that touches YOUR repo.

We embed IN-PROCESS rather than shelling out to a Snakemake rule: the metrics
need the latent vectors (silhouette/separation) and per-centroid distances
(calibration), which a geojson-writing batch rule discards. KMeans.transform()
gives the distances for free.

Faithfulness note: eval MUST preprocess exactly as the encoder was trained.
Your load_images() does grayscale -> resize(128) -> /255.0 (NOT per-sample
min-max, NOT global-percentile). We mirror that here. Changing to global
scaling is a *training* decision; for valid eval we match the checkpoint.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# ── must mirror src/train/latent_space.py ────────────────────────────────────
INPUT_SIZE = 128
MAE_KWARGS = dict(
    img_size=128, patch_size=8, in_chans=1, norm_pix_loss=True,
    embed_dim=768, depth=12, num_heads=12,
    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0,
)
# filename state -> ordinal class. Julie's labeled set spans states 2,3,4 only
# (no fresh/New craters), so the ordinal axis is 3-wide: Semi-New=0 .. Old=2.
# (Training still clusters the full population into 4; this map is EVAL-ONLY.)
STATE_TO_ORDINAL = {2: 0, 3: 1, 4: 2}
ORDINAL_NAMES = ["Semi-New", "Semi-Old", "Old"]


# ── labels from PNG filenames ────────────────────────────────────────────────
def labels_from_png_dir(imgs_dir: str) -> pd.DataFrame:
    """
    Build the labels table directly from Julie's PNG folder.
    Filename convention (from load_images): <crater_id>_<state>.png, state in 1..4.
    Returns crater_id, true_label (0..3 ordinal), filename — row order = sorted
    files, which is exactly the order encode() will produce latents in.
    """
    files = sorted(f for f in os.listdir(imgs_dir) if f.endswith(".png"))
    if not files:
        raise FileNotFoundError(f"no .png files in {imgs_dir}")
    rows, skipped = [], []
    for f in files:
        state = int(f.split("_")[1].split(".")[0])
        if state not in STATE_TO_ORDINAL:
            skipped.append((f, state))          # e.g. a stray New (state 1)
            continue
        rows.append({
            "crater_id": f.rsplit(".", 1)[0],
            "true_label": STATE_TO_ORDINAL[state],
            "filename": f,
        })
    if skipped:
        import warnings
        states = sorted({s for _, s in skipped})
        warnings.warn(
            f"skipped {len(skipped)} PNG(s) with states {states} outside the "
            f"eval class set {sorted(STATE_TO_ORDINAL)} — they are not in Julie's "
            f"3-state labeled set and have no ground-truth class.", stacklevel=2)
    return pd.DataFrame(rows)


# ── global scaling for eval, matching training ───────────────────────────────
def _resolve_scaling(cfg: dict):
    """
    Load frozen (lo, hi) from configs/global_scaling.json. png_max is the PNG's
    integer ceiling (255 for 8-bit) used to invert PNG quantization.
    """
    import json
    sc = json.load(open(cfg["scaling_json"]))
    return float(sc["lo"]), float(sc["hi"]), float(cfg.get("png_max", 255.0))


def _png_to_global(arr_0_255: np.ndarray, lo: float, hi: float,
                   png_max: float) -> np.ndarray:
    """
    Map an 8-bit PNG into the SAME [0,1] global scale the mosaic-trained encoder
    expects.

    ⚠️ INTENSITY-SPACE ASSUMPTION — this is the one thing that can silently
    invalidate the comparison. The frozen (lo, hi) live in raw mosaic-DN space.
    A PNG is [0, png_max]. These two only line up if the PNGs were exported from
    the same mosaic by LINEARLY mapping that exact [lo, hi] window onto
    [0, png_max]. If so, the inverse below restores the global scale exactly:

        png  =  (raw - lo) / (hi - lo) * png_max      (export, assumed)
        ->  back to [0,1] global:  png / png_max

    i.e. when the export used this same window, the correct eval scaling is just
    png/png_max — and that equals the training global_normalize output. So we
    divide by png_max here. If the PNGs were exported with ANY per-image stretch
    or a different window, no constant can fix it and the latents are not
    comparable — re-export Julie's PNGs through the same global window instead.
    """
    return arr_0_255 / png_max


# ── embed: call the repo's real MAE encoder ──────────────────────────────────
def embed(labels: pd.DataFrame, cfg: dict, synthetic: bool = False) -> np.ndarray:
    if synthetic:
        return _synth_latents(labels, cfg)

    import torch
    from src.models.mae_bottleneck import load_mae, encode_for_clustering
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_mae auto-detects whether this checkpoint has the clustering
    # bottleneck (old runs) or not (current default) and builds accordingly —
    # same helper src/cluster/cluster.py and src/display/display.py use, so
    # eval embeds craters exactly the way production clustering does.
    model = load_mae(checkpoint_path=cfg["checkpoint"], device=device, **MAE_KWARGS)

    imgs_dir = cfg["imgs_dir"]
    lo, hi, png_max = _resolve_scaling(cfg)
    tensors = []
    for f in labels["filename"]:
        img = Image.open(os.path.join(imgs_dir, f)).convert("L").resize(
            (INPUT_SIZE, INPUT_SIZE))
        arr = np.array(img, dtype=np.float32)
        arr = _png_to_global(arr, lo, hi, png_max)      # match training scaling
        tensors.append(torch.from_numpy(arr).unsqueeze(0))
    x = torch.stack(tensors).float().to(device)        # (N,1,128,128)

    latents = encode_for_clustering(model, x).cpu().numpy()
    return latents


def kmeans_cluster(latents: np.ndarray, cfg: dict):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=cfg["n_clusters"], n_init=10,
                random_state=cfg.get("seed", 0)).fit(latents)
    return km.labels_, km.transform(latents), km


def build_predictions(labels, latents, clusters, dists) -> pd.DataFrame:
    out = pd.DataFrame({
        "crater_id": labels["crater_id"].values,
        "true_label": labels["true_label"].astype(int).values,
        "cluster": np.asarray(clusters).astype(int),
    })
    if dists is not None:
        for j in range(dists.shape[1]):
            out[f"dist_{j}"] = dists[:, j]
    return out


def _synth_latents(labels, cfg, seed=0):
    rng = np.random.default_rng(seed)
    K = len(cfg["class_names"]); D = cfg.get("latent_dim", 768)
    y = labels["true_label"].to_numpy()
    centers = rng.normal(scale=3.0, size=(K, D))
    return centers[y] + rng.normal(scale=2.0, size=(len(y), D))