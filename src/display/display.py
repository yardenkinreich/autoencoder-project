"""
display.py
──────────
Encode the full crater memmap into latent space, cluster, and export
CSV + GeoJSON for QGIS.

Changes vs old version
───────────────────────
  • EXTRACT_SIZE (320) removed — new memmap is already OUTPUT_SIZE=128,
    no center-crop needed.
  • INPUT_SIZE 224 → 128 to match preprocess.py OUTPUT_SIZE.
  • load_memmap now infers N from file size (mirrors train.py).
  • MemmapDataset no longer center-crops — tensors are ready to use.
  • model.encode() → model.encoder() for CAE (correct attribute name).
  • encode_images import from src.cluster.cluster (latent_space.py),
    which already uses the updated MAE_KWARGS (patch_size=8, img_size=128).
"""

import os
import sys
import shutil
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import geopandas as gpd
from shapely.geometry import Point

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src/models/mae"))
from src.models.autoencoder import ConvAutoencoder
from src.cluster.cluster import encode_images
from src.helper_functions import cluster_and_plot

# ── constants (mirror preprocess.py) ─────────────────────────────────────────
INPUT_SIZE = 128   # ← was 224; matches OUTPUT_SIZE in preprocess.py


# ── dataset ───────────────────────────────────────────────────────────────────

class MemmapDataset(Dataset):
    """
    Wraps the pre-processed memmap.  Tensors are already (1, 128, 128) —
    no center-crop or resize needed (old pipeline had 320→224; new pipeline
    writes the final size directly).
    """
    def __init__(self, memmap_array):
        self.data = memmap_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # copy() avoids a read-only memmap tensor warning
        return (torch.from_numpy(self.data[idx].copy()),)


# ── memmap loader (mirrors train.py) ─────────────────────────────────────────

def load_memmap(path: str, num_channels: int = 1) -> np.memmap:
    """Infer N from file size — works for any OUTPUT_SIZE."""
    file_size    = os.path.getsize(path)
    total_floats = file_size // 4   # float32 = 4 bytes
    size         = INPUT_SIZE
    n_pixels     = num_channels * size * size
    assert total_floats % n_pixels == 0, (
        f"File size {file_size} not divisible by {4 * n_pixels}. "
        f"Check INPUT_SIZE ({size}) and num_channels ({num_channels})."
    )
    N = total_floats // n_pixels
    mm = np.memmap(path, dtype=np.float32, mode="r",
                   shape=(N, num_channels, size, size))
    print(f"  Memmap shape : {mm.shape}")
    print(f"  Value range  : [{mm[:100].min():.3f}, {mm[:100].max():.3f}]")
    return mm


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    craters = load_memmap(args.dataset_path)
    dataset = MemmapDataset(craters)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)

    # ── encode ────────────────────────────────────────────────────────────
    if args.autoencoder_model == "cae":
        model = ConvAutoencoder(latent_dim=args.latent_dim)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device).eval()

        latents_list = []
        with torch.no_grad():
            for i, (x,) in enumerate(loader):
                if i % 100 == 0:
                    print(f"  Batch {i}/{len(loader)}")
                latents_list.append(model.encoder(x.to(device)).cpu().numpy())
        latents = np.concatenate(latents_list, axis=0)
        np.save(args.latent_output, latents)

    else:
        # mae / dino / dino_pretrained / mae_pretrained all go through the
        # same generic build_model()/ENCODE_FNS dispatch in cluster.py - one
        # place that knows how to turn (autoencoder_model, model, imgs) into
        # an embedding, so nothing architecture-specific is needed here.
        encode_images(
            loader, args.model_path, args.bottleneck, device,
            args.latent_output, args.latent_output,
            autoencoder_model=args.autoencoder_model,
            is_dataloader=True,
        )
        latents = np.load(args.latent_output)

    print(f"  Latent database → {args.latent_output}  shape: {latents.shape}")

    # ── metadata ──────────────────────────────────────────────────────────
    metadata = pd.read_csv(args.metadata_path)
    coords   = metadata[["x", "y"]].values

    if len(coords) != len(latents):
        print(f"⚠️  Length mismatch: metadata={len(coords)}, latents={len(latents)}")
        min_len = min(len(coords), len(latents))
        coords  = coords[:min_len]
        latents = latents[:min_len]
        print(f"   Truncating both to {min_len}")

    # ── cluster ───────────────────────────────────────────────────────────
    _, _, _, labels = cluster_and_plot(
        latent           = latents,
        technique        = args.technique,
        n_clusters       = args.num_clusters,
        save_path        = os.path.dirname(args.out_df),
        use_gpu          = args.use_gpu,
        reduce_latent_95 = False,
    )

    # copy generated plot to the exact path Snakemake expects
    generated_png = os.path.join(
        os.path.dirname(args.out_df),
        f"{args.technique}_clusters_kmeans_k{args.num_clusters}.png"
    )
    if os.path.exists(generated_png):
        shutil.copy(generated_png, args.clustering_png)
        print(f"  Clustering plot → {args.clustering_png}")
    else:
        print(f"⚠️  Plot not found at {generated_png} — check cluster_and_plot() naming")

    labels_int = labels.astype(int)

    # ── CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "x":       coords[:, 0],
        "y":       coords[:, 1],
        "cluster": labels_int,
    })
    df.to_csv(args.out_df, index=False)
    print(f"  CSV → {args.out_df}  ({len(df)} rows)")

    # ── GeoJSON ───────────────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(
        {"cluster": labels_int},
        geometry=[Point(xy) for xy in coords],
        crs="ESRI:103881",   # Moon 2000 equirectangular
    )
    geojson_path = args.out_df.replace(".csv", ".geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"  GeoJSON → {geojson_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",        required=True)
    parser.add_argument("--autoencoder_model",  default="mae",
                        choices=["cae", "mae", "dino", "dino_pretrained", "mae_pretrained"])
    parser.add_argument("--dataset_path",      required=True)
    parser.add_argument("--metadata_path",     required=True)
    parser.add_argument("--num_clusters",      type=int,   default=5)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--latent_dim",        type=int,   default=20)
    parser.add_argument("--out_df",            required=True)
    parser.add_argument("--clustering_png",    required=True)
    parser.add_argument("--technique",         choices=["pca", "tsne"], default="pca")
    parser.add_argument("--latent_output",     default="latents_all.npy")
    parser.add_argument("--use_gpu",           action="store_true")
    parser.add_argument("--bottleneck",        type=int,   default=64)
    parser.add_argument("--cluster_method",    default="kmeans",
                        choices=["kmeans", "gmm", "hdbscan", "spectral", "agglomerative"])
    parser.add_argument("--device",            default="cuda")
    args = parser.parse_args()
    main(args)