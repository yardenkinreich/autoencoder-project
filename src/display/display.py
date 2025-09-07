import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
from src.train.train import ConvAutoencoder 
import geopandas as gpd
from shapely.geometry import Point
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main(args):# --- Arguments ---

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(latent_dim=args.latent_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Load dataset ---
    craters = np.load(args.dataset_path).astype(np.float32)
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

    # --- Load metadata ---
    metadata = pd.read_csv(args.metadata_path)
    coords = metadata[['x', 'y']].values  # adjust column names

    # --- Cluster Kmeans ---
    if args.cluster_method == "kmeans":
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
        clusters = kmeans.fit_predict(latents)

        # --- Save crater points + cluster IDs ---
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": clusters
        })
        df.to_csv(args.out_df, index=False)
        print(f"Saved crater clusters to {args.out_df}. You can now overlay in GIS.")

        # --- Create GeoDataFrame ---
        geometry = [Point(xy) for xy in coords]  # coords = Nx2 array of (x, y)
        gdf = gpd.GeoDataFrame({'cluster': clusters}, geometry=geometry)

        # --- Save GeoJSON ---
        geojson_path = args.out_df.replace(".csv", ".geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")

        print(f"GeoJSON with clusters saved to: {geojson_path}")

        if args.technique == "pca":
            technique = PCA(n_components=2)
        elif args.technique == "tsne":           
            technique = TSNE(n_components=2)
        latents_2d = technique.fit_transform(latents)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                        c=clusters, cmap="viridis", alpha=0.7)

        plt.colorbar(scatter, label="Cluster ID")
        lt.title(f"Kmeans clustering (n={args.num_clusters})")
        plt.xlabel(f"{technique}1")
        plt.ylabel(f"{technique}2")
        plt.tight_layout()
        plt.savefig(args.out_df.replace(".csv", "_clusters.png"))
        plt.show()


    # --- Cluster GMM ---
    elif args.cluster_method == "gmm":
        if args.find_optimal_clusters:
            # Determine optimal number of clusters using BIC
            lowest_bic = np.inf
            best_gmm = None
            n_components_range = range(4, args.num_clusters)  # try clusters 2..10 (adjust range as needed)
            bics = []

            for n in n_components_range:
                gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=0)
                gmm.fit(latents)
                bic = gmm.bic(latents)
                bics.append(bic)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm

            print(f"Best number of clusters (BIC): {best_gmm.n_components}")
        else:
            best_gmm = GaussianMixture(n_components=args.num_clusters, covariance_type="full", random_state=0)
            best_gmm.fit(latents)

        # Hard cluster assignments
        clusters = best_gmm.predict(latents)

        # Soft cluster probabilities
        probs = best_gmm.predict_proba(latents)  # shape: [n_samples, n_clusters]

        # --- Save crater points + cluster IDs ---
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": clusters
        })

        # also save probabilities for each cluster
        for k in range(best_gmm.n_components):
            df[f"p_cluster{k}"] = probs[:, k]

        df.to_csv(args.out_df, index=False)
        print(f"Saved crater clusters with probabilities to {args.out_df}.")
        
        geometry = [Point(xy) for xy in coords]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)

        geojson_path = args.out_df.replace(".csv", ".geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")

        if args.technique == "pca":
            technique = PCA(n_components=2)
        elif args.technique == "tsne":           
            technique = TSNE(n_components=2)
        latents_2d = technique.fit_transform(latents)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                        c=clusters, cmap="viridis", alpha=0.7)

        plt.colorbar(scatter, label="Cluster ID")
        plt.title(f"GMM clustering (n={best_gmm.n_components})")
        plt.xlabel(f"{args.technique}1")
        plt.ylabel(f"{args.technique}2")
        plt.tight_layout()
        plt.savefig(args.out_df.replace(".csv", f"_{args.technique}.png"))
        plt.show()


#arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to trained autoencoder")
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
args = parser.parse_args()
main(args)
    