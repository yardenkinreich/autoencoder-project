import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import numpy as np







def compare_clusters(latents, technique, cluster_method , num_clusters, use_gpu, out_df):

    STATE_LABELS = {
    1: "New Crater",
    2: "Semi New Crater",
    3: "Semi Old Crater",
    4: "Old Crater"
    }

    STATE_COLORS = {
    1: "tab:blue",
    2: "tab:green",
    3: "tab:orange",
    4: "tab:red"
    }

    fig_reg, fig_diag, emb, labels = cluster_and_plot(
        latent=latents,
        technique=technique,
        n_clusters=num_clusters,
        save_path=os.path.dirname(out_df),
        use_gpu=use_gpu
    )


    # Load data
    latents = np.load("latents_julie.npy")
    states = np.load("states_julie.npy")  # ground truth 1–4

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(latents)

    # Compare
    ari = adjusted_rand_score(states, clusters)
    nmi = normalized_mutual_info_score(states, clusters)
    fmi = fowlkes_mallows_score(states, clusters)

    print(f"ARI: {ari:.3f}  NMI: {nmi:.3f}  FMI: {fmi:.3f}")


parser = argparse.ArgumentParser()
parser.add_argument("--latents", type=str, required=True, help="Path to latens.npy file")
parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--out_df", type=str, required=True, help="Path to save output CSV with crater clusters")
parser.add_argument("--cluster_method", type=str, choices=["kmeans", "gmm"], default="kmeans", help="Clustering method to use")
parser.add_argument("--technique", type=str, choices=["pca", "tsne"], default="pca", help="Dimensionality reduction technique for visualization")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for computations if available")
args = parser.parse_args()
main(args)

