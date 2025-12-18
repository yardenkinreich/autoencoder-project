import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import timm
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import argparse
from src.train.train import ConvAutoencoder
from src.helper_functions import *
import os
import sys
sys.path.append(os.path.abspath("src/models/mae"))
from src.train.train import *
from src.models.mae.models_mae import *
from src.cluster.cluster import encode_images


def main(args):# --- Arguments ---

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.autoencoder_model == "cnn":
        craters = np.load(args.dataset_path).astype(np.float32)
        model = ConvAutoencoder(latent_dim=args.latent_dim)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()

        # --- Load dataset ---
        # Reshape to 1x100x100 for Conv2d
        craters = craters.reshape(-1, 1, 100, 100)
        dataset = TensorDataset(torch.tensor(craters))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # --- Encode latents ---
        latents = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                latent = model.encoder(x)
                latents.append(latent.cpu().numpy())
        latents = np.concatenate(latents, axis=0)

    elif args.autoencoder_model == "mae":
        # Load data
        file_size = os.path.getsize(args.dataset_path)
        N = file_size // (224 * 224 * 3 * 4)
        craters = np.memmap(
            args.dataset_path,
            dtype=np.float32,
            mode="r",
            shape=(N, 3, 224, 224)
        )

        print(f"Data range: min={craters.min()}, max={craters.max()}, mean={craters.mean()}")

        # Create DataLoader to process in batches
        # Copy batches on-the-fly to avoid read-only warning
        class MemmapDataset(torch.utils.data.Dataset):
            def __init__(self, memmap_array):
                self.data = memmap_array
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                # Copy to make it writable
                return torch.from_numpy(np.array(self.data[idx]))
        
        dataset = MemmapDataset(craters)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Pass DataLoader instead of tensor
        encode_images(loader, args.model_path, args.bottleneck, device,
                    args.latent_output, args.latent_output, 
                    autoencoder_model=args.autoencoder_model,
                    is_dataloader=True)
        latents = np.load(args.latent_output)

    # --- Load metadata ---
    metadata = pd.read_csv(args.metadata_path)
    coords = metadata[['x', 'y']].values

    fig_reg, emb, labels = cluster_and_plot(
        latent=latents,
        technique=args.technique,
        n_clusters=args.num_clusters,
        save_path=os.path.dirname(args.out_df),
        use_gpu=args.use_gpu
    )

    # --- Save Clustering Info ---
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels
    })
    df.to_csv(args.out_df, index=False)
    print(f"Saved crater clusters to {args.out_df}. You can now overlay in GIS.")

    # --- GeoDataFrame & GeoJSON ---
    geometry = [Point(xy) for xy in coords]
    gdf = gpd.GeoDataFrame({'cluster': labels}, geometry=geometry)
    geojson_path = args.out_df.replace(".csv", ".geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"GeoJSON with clusters saved to: {geojson_path}")


    
#arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to trained autoencoder")
parser.add_argument("--autoencoder_model", type=str, choices=["cnn", "mae"], default="cnn", help="Type of autoencoder model")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to craters.npy dataset")
parser.add_argument("--metadata_path", type=str, required=True, help="CSV with crater coordinates")
parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--latent_dim", type=int, default=20, help="Dimensionality of latent space of the autoencoder")
parser.add_argument("--out_df", type=str, required=True, help="Path to save output CSV with crater clusters")
parser.add_argument("--cluster_method", type=str, choices=["kmeans", "gmm"], default="kmeans", help="Clustering method to use")
parser.add_argument("--technique", type=str, choices=["pca", "tsne"], default="pca", help="Dimensionality reduction technique for visualization")
parser.add_argument("--find_optimal_clusters", action="store_true", help="For GMM, find optimal number of clusters using BIC")
parser.add_argument("--latent_output", type=str, default="latents_all.npy", help="Path to save full dataset latent vectors as .npy")
parser.add_argument("--freeze_until", type=int, default=-2, help="For MAE: number of encoder transformer blocks to freeze from the end (negative number)")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for computations if available")
parser.add_argument("--pretrained_model", type=str, default='facebook/vit-mae-large', help="Pretrained model name for MAE")
parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio for MAE")
parser.add_argument("--bottleneck", type=int, default=6, help="Bottleneck size for CNN autoencoder")
args = parser.parse_args()
main(args)
    