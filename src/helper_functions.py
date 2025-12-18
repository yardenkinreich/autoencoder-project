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

try:
    import cupy as cp
#    import cudf
    from cuml.manifold import TSNE as cuTSNE
    from cuml.decomposition import PCA as cuPCA
    from cuml.cluster import KMeans as cuKMeans

    GPU_AVAILABLE = True
except ImportError:
    pass
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
GPU_AVAILABLE = False
from matplotlib import colors
import argparse
import os
from PIL import Image
from torchvision import transforms


# Data Preprocessing Functions

def crop_crater(map_ref, lat, lon, diameter, offset, transformer, autoencoder_model):

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

def normalize_image_to_imagenet_stats(image_path):
    """
    Normalize a PNG image to have ImageNet mean and std.
    Input:
        image_path: path to PNG file
    Output:
        normalized_image: 3D numpy array (3, 224, 224) with ImageNet normalization applied
    """
    # Load the PNG image and convert to greyscale
    pil_img = Image.open(image_path).convert('L')  # 'L' mode = greyscale
    
    # Convert to numpy array
    img_array = np.array(pil_img)
    
    # Flip crater so shadow is always on the right
    img_array = flip_crater(img_array)
    
    # Normalize to [0, 255] range with contrast stretching
    img_min = img_array.min()
    img_max = img_array.max()
    
    if img_max > img_min:
        img_array = ((img_array.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_array = np.full_like(img_array, 128, dtype=np.uint8)
    
    # Convert greyscale to RGB by stacking 3 times (for ImageNet normalization)
    img_array = np.stack([img_array] * 3, axis=-1)
    
    # Convert to PIL for transforms
    pil_img = Image.fromarray(img_array, mode='RGB')
    
    # Apply MAE-style transforms
    mae_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Converts to [0,1] and changes to (C, H, W)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor_img = mae_transform(pil_img)  # (3, 224, 224)
    
    return tensor_img



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
                     random_state=42, use_gpu=False, cluster_method='kmeans',imgs_dir=None):
    """
    Cluster a latent space using KMeans or other methods and visualize in 2D using PCA or t-SNE.
    Can use GPU via RAPIDS cuML if available and use_gpu=True.
    
    Returns:
        fig_regular: Regular clustering visualization
        fig_diagonal: Diagonal-sorted clustering visualization
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
    
    # Standardize
    if use_gpu and GPU_AVAILABLE:
        latents = cp.array(latents)
        mean = cp.mean(latents, axis=0)
        std = cp.std(latents, axis=0)
        latents = (latents - mean) / (std + 1e-8)
    else:
        latents = StandardScaler().fit_transform(latents)
    
    # --- Dimensionality reduction ---
    
    if technique.lower() == 'pca':
        if use_gpu and GPU_AVAILABLE:
            reducer = cuPCA(n_components=0.95, random_state=random_state)
        else:
            reducer = PCA(n_components=0.95, random_state=random_state)

    embedding_lower_dim = reducer.fit_transform(latents)

    print("Original latent dim:", latents.shape[1])
    print("After PCA (95% variance):", embedding_lower_dim.shape[1])

    # --- Clustering ---

    cluster_method = cluster_method.lower()
    if cluster_method == 'kmeans':
        if use_gpu and GPU_AVAILABLE:
            cluster_labels = cuKMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(embedding_lower_dim)
            cluster_labels_cpu = cp.asnumpy(cluster_labels)
        else:
            cluster_labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(embedding_lower_dim)
            cluster_labels_cpu = cluster_labels
    else:
        raise ValueError(f"Unknown cluster_method: {cluster_method}. Supported: 'kmeans'")
    
    # --- Plotting Tactic ---

    n_samples = latents.shape[0] if not use_gpu else int(latents.shape[0])
    if technique.lower() == 'pca':
        if use_gpu and GPU_AVAILABLE:
            plot_reduce = cuPCA(n_components=2, random_state=random_state)
        else:
            plot_reduce = PCA(n_components=2, random_state=random_state)

    elif technique.lower() == 'tsne':
        #perplexity = min(30, max(5, n_samples // 3))  # Adaptive
        if use_gpu and GPU_AVAILABLE:
            plot_reduce = cuTSNE(n_components=2, random_state=random_state)
        else:
            plot_reduce = TSNE(n_components=2, random_state=random_state)
    else:
        raise ValueError("technique must be 'pca' or 'tsne'")
    
    embedding = plot_reduce.fit_transform(embedding_lower_dim)
    embedding_cpu = cp.asnumpy(embedding) if use_gpu and GPU_AVAILABLE else embedding
    
    # --- Plotting ---
    cmap = plt.cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')

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
    else:
        scatter_images(
            ax=ax,
            coords=embedding_cpu,
            labels=cluster_labels_cpu,
            imgs_dir=imgs_dir,
            cmap=cmap,
            xlabel="PCA 1",
            ylabel="PCA 2",
            title="PCA + KMeans Clustering with Images"
            )

    '''
    ax.set_xlabel(f'{technique.upper()} Component 1')
    ax.set_ylabel(f'{technique.upper()} Component 2')
    ax.set_title(f"{technique.upper()} Clustering (k={n_clusters}, method={cluster_method})")
    ax.axis("off")
    '''
    
    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_regular.savefig(f"{save_path}/{technique}_clusters_regular_k{n_clusters}_{cluster_method}.png", 
                           dpi=300, bbox_inches='tight')
        print(f"Saved plots to {save_path}")
    
    # Return in the format expected by compare_clusters
    return fig_regular, embedding_cpu, cluster_labels_cpu


def scatter_images(ax, coords, labels, imgs_dir, cmap=None,
                   xlabel="Component 1", ylabel="Component 2", title="Latent space"):
    """
    Scatter images at 2D coordinates with cluster-colored borders.
    
    Args:
        ax: matplotlib Axes to plot on
        coords: (N,2) array of 2D coordinates
        labels: array of cluster labels (for border color)
        imgs_dir: folder with .png images
        img_size: int, size of the images in points
        cmap: matplotlib colormap for clusters
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

        # Border only, not zoomed
        ab = AnnotationBbox(
            imgbox,
            (x, y),
            frameon=True,
            pad=0,  # space between image and border
            bboxprops=dict(
                edgecolor=cmap(lbl % cmap.N) if cmap else "black",
                linewidth=1
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




