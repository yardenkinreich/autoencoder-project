from rasterio.windows import from_bounds
import numpy as np
import cv2
from numpy import cos, radians
import pyproj
import random
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import umap


try:
    import cupy as cp
#    import cudf
    from cuml.manifold import TSNE as cuTSNE
    from cuml.decomposition import PCA as cuPCA
    from cuml.cluster import KMeans as cuKMeans
    from cuml import UMAP as cuUMAP

    GPU_AVAILABLE = True
    print("RAPIDS cuML imported successfully. GPU computations enabled.")
except ImportError:
    pass
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import colors
import argparse
import os
from PIL import Image
from torchvision import transforms



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


def cluster_and_plot(latent, technique='tsne', n_clusters=5, save_path=None, 
                     random_state=42, use_gpu=False, cluster_method='kmeans',
                     imgs_dir=None, ground_truth_labels=None, state_labels=None, 
                     state_colors=None, reduce_latent_95=False):
    """
    Cluster a latent space using KMeans or other methods and visualize in 2D using PCA or t-SNE.
    Can use GPU via RAPIDS cuML if available and use_gpu=True.
    
    Args:
        ground_truth_labels: Optional array of manual labels for ground truth visualization
        state_labels: Optional dict mapping label values to names (e.g., {0: "Intact", 1: "Degraded"})
        state_colors: Optional dict mapping label values to colors
    
    Returns:
        fig_regular: Regular clustering visualization
        fig_gt: Ground truth visualization (if ground_truth_labels provided)
        embedding_cpu: 2D embedding coordinates
        cluster_labels_cpu: Predicted cluster labels
    """
    # Load latents
    if isinstance(latent, str):
        latents = np.load(latent)
    else:       
        latents = latent

    # Validation
    if latents.shape[0] < n_clusters:
        raise ValueError(f"Number of samples ({latents.shape[0]}) must be >= n_clusters ({n_clusters})")
    print("Original latent dim:", latents.shape[1])

    # --- Latent Dimensionality Reduction ---
    if reduce_latent_95:
        reducer_95 = PCA(n_components=0.95, random_state=random_state)
        latents = reducer_95.fit_transform(latents)
        print("After PCA (95% variance):", latents.shape[1])

    # --- Clustering ---

    cluster_method = cluster_method.lower()
    if cluster_method == 'kmeans':
        if use_gpu and GPU_AVAILABLE:
            latents = cp.asarray(latents)
            cluster_labels = cuKMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(latents)
            cluster_labels_cpu = cp.asnumpy(cluster_labels)
        else:
            cluster_labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(latents)
            cluster_labels_cpu = cluster_labels

    elif cluster_method == 'hdbscan':
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=10, min_samples=10)
        cluster_labels = clusterer.fit_predict(latents)
        cluster_labels_cpu = cluster_labels

    elif cluster_method == 'gmm':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=random_state)
        cluster_labels = gmm.fit_predict(latents)
        cluster_labels_cpu = cluster_labels

    elif cluster_method == 'spectral':
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state)
        cluster_labels = spectral.fit_predict(latents)
        cluster_labels_cpu = cluster_labels
    
    elif cluster_method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert to CPU if using GPU
        data_to_cluster = cp.asnumpy(latents) if use_gpu and GPU_AVAILABLE else latents
        
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',  # Ward linkage minimizes variance (good for your data)
            # linkage='average',  # Alternative: average linkage
            # linkage='complete',  # Alternative: complete linkage (max distance)
        )
        cluster_labels = agg.fit_predict(data_to_cluster)
        cluster_labels_cpu = cluster_labels

    print(f"Finished Clustering! Clustering method: {cluster_method}, Number of clusters: {n_clusters}")

    # --- Plotting Tactic ---

    n_samples = latents.shape[0] if not use_gpu else int(latents.shape[0])
    if technique.lower() == 'pca':
        if use_gpu and GPU_AVAILABLE:
            plot_reduce = cuPCA(n_components=2, random_state=random_state)
        else:
            plot_reduce = PCA(n_components=2, random_state=random_state)

    elif technique.lower() == 'tsne':
        if use_gpu and GPU_AVAILABLE:
            plot_reduce = cuTSNE(n_components=2, random_state=random_state)
        else:
            plot_reduce = TSNE(n_components=2, random_state=random_state)
            
    elif technique.lower() == 'umap':
        if use_gpu and GPU_AVAILABLE:
            plot_reduce = cuUMAP(n_components=2, random_state=random_state)
        else:
            plot_reduce = umap.UMAP(n_components=2, random_state=random_state)
        
    else:
        raise ValueError("technique must be 'pca' or 'tsne' or 'umap'")
    
    embedding = plot_reduce.fit_transform(latents)
    embedding_cpu = cp.asnumpy(embedding) if use_gpu and GPU_AVAILABLE else embedding
    
    # Get axis labels for PCA
    if technique.lower() == 'pca':
        explained_var = plot_reduce.explained_variance_ratio_
        x_label = f"PC1 ({explained_var[0]*100:.1f}% variance)"
        y_label = f"PC2 ({explained_var[1]*100:.1f}% variance)"
        total_var = explained_var.sum() * 100
        print(f"[PCA] Total variance explained (PC1+PC2): {total_var:.2f}%")
    else:
        x_label = f"{technique.upper()} Component 1"
        y_label = f"{technique.upper()} Component 2"
    
    # --- Plotting ---
    cmap = plt.cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')

    # Figure 1: K-means clustering results
    fig_regular, ax = plt.subplots(figsize=(8, 6))

    if imgs_dir is None:
        scatter = ax.scatter(
            embedding_cpu[:,0],
            embedding_cpu[:,1],
            c=cluster_labels_cpu,
            cmap=cmap,
            s=20,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{technique.upper()} + {cluster_method} Clustering (k={n_clusters})")
    else:
        scatter_images(
            ax=ax,
            coords=embedding_cpu,
            labels=cluster_labels_cpu,
            imgs_dir=imgs_dir,
            cmap=cmap,
            xlabel=x_label,
            ylabel=y_label,
            title=f"{technique.upper()} + {cluster_method} Clustering with Images (k={n_clusters})"
        )
    
    # Figure 2: Ground truth labels (if provided)
    fig_gt = None
    if ground_truth_labels is not None:
        fig_gt, ax_gt = plt.subplots(figsize=(10, 7))
        
        if imgs_dir is None:
            # Regular scatter plot colored by ground truth
            ax_gt.set_title(
                f"{technique.upper()} Projection of Latent Space "
                f"Colored by Ground-Truth Deformation State"
            )
            ax_gt.set_xlabel(x_label)
            ax_gt.set_ylabel(y_label)
            
            for state in np.unique(ground_truth_labels):
                mask = ground_truth_labels == state
                label = state_labels.get(state, f"State {state}") if state_labels else f"State {state}"
                color = state_colors.get(state, "gray") if state_colors else cmap(state % cmap.N)
                ax_gt.scatter(
                    embedding_cpu[mask, 0],
                    embedding_cpu[mask, 1],
                    label=label,
                    c=[color],
                    alpha=0.7,
                    s=20
                )
            ax_gt.legend()
        else:
            # Image scatter with ground truth colors
            scatter_images(
                ax=ax_gt,
                coords=embedding_cpu,
                labels=ground_truth_labels,
                imgs_dir=imgs_dir,
                state_colors=state_colors,  # Pass the specific color mapping
                xlabel=x_label,
                ylabel=y_label,
                title=f"{technique.upper()} with Ground Truth Deformation Labels"
            )
            
            # Add a legend manually for ground truth
            if state_labels:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='white', edgecolor=state_colors.get(s, 'gray'), 
                        label=state_labels.get(s, f"State {s}"), linewidth=2)
                    for s in sorted(np.unique(ground_truth_labels))
                ]
                ax_gt.legend(handles=legend_elements, loc='best')
    
    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_regular.savefig(
            f"{save_path}/{technique}_clusters_kmeans_k{n_clusters}.png", 
            dpi=300, bbox_inches='tight'
        )
        if fig_gt is not None:
            fig_gt.savefig(
                f"{save_path}/{technique}_ground_truth_labels.png",
                dpi=300, bbox_inches='tight'
            )
        print(f"Saved plots to {save_path}")
    
    # Return figures and data
    return fig_regular, fig_gt, embedding_cpu, cluster_labels_cpu

def scatter_images(ax, coords, labels, imgs_dir, cmap=None, state_colors=None,
                   xlabel="Component 1", ylabel="Component 2", title="Latent space"):
    """
    Scatter images at 2D coordinates with cluster-colored borders.
    
    Args:
        ax: matplotlib Axes to plot on
        coords: (N,2) array of 2D coordinates
        labels: array of cluster labels (for border color)
        imgs_dir: folder with .png images
        cmap: matplotlib colormap for clusters
        state_colors: optional dict mapping labels to specific colors (for ground truth)
        xlabel, ylabel, title: strings for plot labeling
    """
    files = sorted([
        os.path.join(imgs_dir, f)
        for f in os.listdir(imgs_dir)
        if f.endswith(".png") and os.path.isfile(os.path.join(imgs_dir, f))
    ])
    assert len(files) == len(coords), "Number of images must match number of samples"

    for (x, y), lbl, fname in zip(coords, labels, files):
        img = Image.open(fname).convert("L")  # grayscale PIL image
        img_arr = np.array(img)                     # numeric array
        imgbox = OffsetImage(img_arr, zoom=0.15, cmap="gray")  # now grayscale works

        # Determine edge color
        if state_colors is not None:
            edge_color = state_colors.get(lbl, "black")
        elif cmap is not None:
            edge_color = cmap(lbl % cmap.N)
        else:
            edge_color = "black"

        # Border only, not zoomed
        ab = AnnotationBbox(
            imgbox,
            (x, y),
            frameon=True,
            pad=0,  # space between image and border
            bboxprops=dict(
                edgecolor=edge_color,
                linewidth=1.5
            )
        )
        ax.add_artist(ab)

    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axis("on")  # keep axes visible
    ax.update_datalim(coords)
    ax.autoscale()

def vicreg_regularizer(z, gamma_var=1.0, gamma_cov=1.0, eps=1e-4):
    """
    z: [batch, latent_dim] bottleneck embeddings
    gamma_var, gamma_cov: weights for variance and covariance penalties
    """
    # --- Variance loss ---
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - std))

    # --- Covariance loss ---
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / z.shape[1]

    return gamma_var * var_loss + gamma_cov * cov_loss

def smooth_entropy_regularizer(z, temperature=0.1, gamma_ent=0.1):
    """
    Encourages local structure while avoiding hard cluster collapse.
    """
    sim = torch.cdist(z, z, p=2)  # pairwise distances
    sim = torch.exp(-sim / temperature)
    q = sim / sim.sum(dim=1, keepdim=True)
    ent = - (q * torch.log(q + 1e-8)).sum(dim=1).mean()
    return -gamma_ent * ent  # maximize entropy (soft structure)

def unpatchify_mask(mask, patch_size=16, img_size=224):
    """
    Convert a [B, num_patches] boolean mask into an image-space binary mask [B, 1, H, W].
    Each patch is upsampled to patch_size×patch_size.
    """
    B, num_patches = mask.shape
    h = w = int(num_patches ** 0.5)
    mask = mask.reshape(B, h, w)
    mask = np.kron(mask, np.ones((patch_size, patch_size)))  # upsample
    mask = torch.tensor(mask).unsqueeze(1).float()  # (B, 1, H, W)
    return mask



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




