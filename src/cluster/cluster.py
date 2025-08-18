import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

def main(args):
    # Load latent vectors
    latent_vectors = np.load(args.latent_input)  # shape: (num_craters, latent_dim)

    # Dimensionality reduction for visualization (to 2D)
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    # Clustering into 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_vectors)

    # Dot plot of clusters
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(latent_2d[cluster_labels==i,0],
                    latent_2d[cluster_labels==i,1],
                    label=f'Cluster {i+1}')
    plt.title("Crater Clusters (Dots)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.savefig(args.dot_plot)
    plt.close()

    # Image plot of clusters
    plt.figure(figsize=(12, 10))
    for i, (x, y) in enumerate(latent_2d):
        img_path = os.path.join(args.images_dir, f"{i:05d}.jpeg")  # adjust naming scheme if needed
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert("L"))
        im = OffsetImage(img, zoom=0.3, cmap='gray')
        ab = AnnotationBbox(im, (x, y), frameon=False)
        plt.gca().add_artist(ab)
    plt.title("Crater Clusters (Images)")
    plt.savefig(args.image_plot)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_input', required=True, help="Path to saved latent vectors (npy)")
    parser.add_argument('--images_dir', required=True, help="Directory with crater images")
    parser.add_argument('--dot_plot', required=True, help="Path to save dot cluster plot")
    parser.add_argument('--image_plot', required=True, help="Path to save image cluster plot")
    args = parser.parse_args()
    main(args)