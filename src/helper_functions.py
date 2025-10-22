from rasterio.windows import from_bounds
import numpy as np
import cv2
from numpy import cos, radians
import pyproj
import random
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
try:
    import cupy as cp
    import cudf
    from cuml.manifold import TSNE as cuTSNE
    from cuml.decomposition import PCA as cuPCA
    from cuml.cluster import KMeans as cuKMeans

    GPU_AVAILABLE = True
except ImportError:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    GPU_AVAILABLE = False
from matplotlib import colors
import argparse
import os


# Data Preprocessing Functions

def crop_crater(map_ref, lat, lon, diameter, offset, transformer):

    if lon > 180:
        lon -= 360
    # Convert latitude and longitude to map's coordinate system
    x, y = transformer.transform(lon, lat)
    # Define bounding box in projected coordinates
    radius = (diameter / 2) * 1000  # Convert km to meters
    radius_with_offset_x = (radius + radius * offset) / cos(radians(lat))
    radius_with_offset_y = radius + radius * offset
    min_x, min_y = x - radius_with_offset_x, y - radius_with_offset_y
    max_x, max_y = x + radius_with_offset_x, y + radius_with_offset_y

    # Get the window for cropping
    window = from_bounds(min_x, min_y, max_x, max_y, transform=map_ref.transform)

    # Read and crop the data
    cropped_image = map_ref.read(window=window)

    cropped_image = cropped_image.reshape((cropped_image.shape[1], cropped_image.shape[2]))

    projected_height = int(cropped_image.shape[0] / cos(radians(abs(lat))))
    if projected_height > cropped_image.shape[0]:
        cropped_image_projected = cv2.resize(cropped_image, (cropped_image.shape[1], projected_height))
    else:
        cropped_image_projected = cropped_image

    flipped_image = flip_crater(cropped_image_projected)

    return flipped_image


def flip_crater(img):
    '''
    Flips crater s.t. the shadow will always be on the r.h.s
    '''
    qtr_img_width = np.int16(img.shape[1] / 4)
    half_img_width = np.int16(img.shape[1] / 2)

    left_crater_side = img[:, qtr_img_width:half_img_width]
    right_crater_side = img[:, half_img_width:-qtr_img_width]

    if left_crater_side.mean() > right_crater_side.mean():
        pass
    else:
        img = np.fliplr(img)

    return img


def cluster_and_plot(latent, technique='tsne', n_clusters=5, save_path=None, random_state=42, use_gpu=False):
    """
    Cluster a latent space using KMeans and visualize it in 2D using PCA or t-SNE.
    Can use GPU via RAPIDS cuML if available and use_gpu=True.
    """

    if isinstance(latent, str):
        latents = np.load(latent)
    else:       
        latents = latent

    # Ensure latent is on CPU or GPU
    if use_gpu and GPU_AVAILABLE:
        latents = cp.array(latents)
        mean = cp.mean(latents, axis=0)
        std = cp.std(latents, axis=0)
        latents = (latents - mean) / (std + 1e-8)
    else:
        latents = StandardScaler().fit_transform(latents)

     # --- KMeans clustering ---
    cluster_labels = cuKMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(
        latents) if use_gpu and GPU_AVAILABLE else KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(latents)

    cluster_labels_cpu = cp.asnumpy(cluster_labels) if use_gpu and GPU_AVAILABLE else cluster_labels
    
    # --- Dimensionality reduction ---
    if technique.lower() == 'pca':
        if use_gpu and GPU_AVAILABLE:
            reducer = cuPCA(n_components=2, random_state=random_state)
        else:
            reducer = PCA(n_components=2, random_state=random_state)
    elif technique.lower() == 'tsne':
        if use_gpu and GPU_AVAILABLE:
            reducer = cuTSNE(n_components=2, random_state=random_state, perplexity=100)
        else:
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=100)
    else:
        raise ValueError("technique must be 'pca' or 'tsne'")
    
    embedding = reducer.fit_transform(latents)
    embedding_cpu = cp.asnumpy(embedding) if use_gpu and GPU_AVAILABLE else embedding
    
    # --- Plotting  ---

    fig_regular, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.colormaps['cubehelix'].resampled(n_clusters)
    ax.scatter(embedding_cpu[:,0], embedding_cpu[:,1], c=cluster_labels_cpu, cmap=cmap, s=10)
    ax.set_title(f"{technique.upper()} clustering")
    
    # Diagonal plot
    diagonal_scores = -(embedding_cpu[:,0]) + (embedding_cpu[:,1])
    cluster_scores = [(c, diagonal_scores[cluster_labels_cpu==c].mean()) for c in np.unique(cluster_labels_cpu)]
    sorted_clusters = [c for c,_ in sorted(cluster_scores, key=lambda x:x[1], reverse=True)]
    remap_dict = {old:new for new,old in enumerate(sorted_clusters)}
    remapped_labels = np.vectorize(remap_dict.get)(cluster_labels_cpu)
    
    fig_diagonal, ax2 = plt.subplots(figsize=(6,6))
    ax2.scatter(embedding_cpu[:,0],
                embedding_cpu[:,1],
                c=remapped_labels, cmap=cmap, s=10)
    ax2.set_title(f"{technique.upper()} clusters (diagonal-sorted)")
    
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        fig_regular.savefig(f"{save_path}/{technique}_clusters_regular.png", dpi=300, bbox_inches='tight')
        fig_diagonal.savefig(f"{save_path}/{technique}_clusters_diagonal.png", dpi=300, bbox_inches='tight')
    
    plt.close(fig_regular)
    plt.close(fig_diagonal)
    
    return fig_regular, fig_diagonal, embedding_cpu, cluster_labels_cpu

def main():
    parser = argparse.ArgumentParser(description="Cluster and plot latent vectors.")
    parser.add_argument("--latent", type=str, required=True, help="Path to latent .npy file")
    parser.add_argument("--technique", type=str, default="tsne", choices=["tsne", "pca"], help="Dimensionality reduction method")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--save_path", type=str, default="results/plots", help="Directory to save plots")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for computations if available")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    latents = np.load(args.latent)
    print(f"Loaded latents of shape {latents.shape}")

    fig_reg, fig_diag, emb, labels = cluster_and_plot(
        latent=latents,
        technique=args.technique,
        n_clusters=args.clusters,
        save_path=args.save_path,
        use_gpu=args.use_gpu
    )

    print(f"Saved plots to {args.save_path}")
    print(f"Cluster labels shape: {labels.shape}")


# ------------------------
# Allows both import + terminal run
# ------------------------
if __name__ == "__main__":
    main()




