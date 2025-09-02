import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
from src.train.train import ConvAutoencoder 
import geopandas as gpd
from shapely.geometry import Point


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

    # --- Cluster ---
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
    clusters = kmeans.fit_predict(latents)

    # --- Load metadata ---
    metadata = pd.read_csv(args.metadata_path)
    coords = metadata[['x', 'y']].values  # adjust column names

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
    geojson_path = args.out_df.replace(".csv", "_clusters.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"GeoJSON with clusters saved to: {geojson_path}")



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
args = parser.parse_args()
main(args)
