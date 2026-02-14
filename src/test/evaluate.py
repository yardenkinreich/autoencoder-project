import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import json
from torchvision import transforms
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                             fowlkes_mallows_score, confusion_matrix,
                             homogeneity_score, completeness_score, v_measure_score,
                             silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import spearmanr, entropy
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

# --- Project Root Configuration ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from helper_functions import cluster_and_plot
except ImportError:
    print("Warning: helper_functions not found. Ensure cluster_and_plot is accessible.")

# --- Global Configurations ---
STATE_LABELS_4 = {1: "New", 2: "Semi-New", 3: "Semi-Old", 4: "Old"}
STATE_COLORS_4 = {1: "tab:blue", 2: "tab:green", 3: "tab:orange", 4: "tab:red"}

STATE_LABELS_2 = {1: "Fresh", 2: "Degraded"}
STATE_COLORS_2 = {1: "tab:blue", 2: "tab:red"}

# --- GEOMETRY & STRUCTURAL FUNCTIONS ---

def compare_cluster_geometry(embedding, predicted_labels, ground_truth, label_mapping):
    """
    Measures spatial alignment between predicted centroids and GT centroids
    for matched classes only.
    """
    unique_gt = sorted(np.unique(ground_truth))

    # Invert mapping: GT -> predicted label
    aligned_to_pred = {v: k for k, v in label_mapping.items()}

    gt_centroids = []
    pr_centroids = []

    for gt in unique_gt:
        if gt not in aligned_to_pred:
            continue

        pr = aligned_to_pred[gt]

        gt_mask = ground_truth == gt
        pr_mask = predicted_labels == pr

        if gt_mask.sum() == 0 or pr_mask.sum() == 0:
            continue

        gt_centroids.append(embedding[gt_mask].mean(axis=0))
        pr_centroids.append(embedding[pr_mask].mean(axis=0))

    gt_centroids = np.array(gt_centroids)
    pr_centroids = np.array(pr_centroids)

    # Not enough structure to compare
    if len(gt_centroids) < 2:
        return {
            'procrustes_disparity': np.nan,
            'distance_correlation': np.nan
        }

    # Procrustes
    _, _, disparity = procrustes(gt_centroids, pr_centroids)

    # Distance correlation
    gt_dists = cdist(gt_centroids, gt_centroids)[np.triu_indices(len(gt_centroids), k=1)]
    pr_dists = cdist(pr_centroids, pr_centroids)[np.triu_indices(len(pr_centroids), k=1)]

    if np.std(gt_dists) == 0 or np.std(pr_dists) == 0:
        corr = np.nan
    else:
        corr, _ = spearmanr(gt_dists, pr_dists)

    return {
        'procrustes_disparity': disparity,
        'distance_correlation': corr
    }


def analyze_boundary_uncertainty(embedding, ground_truth):
    """
    Uses SVM probability entropy to see if the classes are distinct or a continuum.
    """
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    # Use cross-validation to get probabilities for the whole set
    cv_folds = min(5, np.min(np.unique(ground_truth, return_counts=True)[1]))
    if cv_folds < 2: return {'boundary_uncertainty': 0.0} # Not enough data
    
    pred_proba = cross_val_predict(clf, embedding, ground_truth, cv=cv_folds, method='predict_proba')
    return {'boundary_uncertainty': np.mean(entropy(pred_proba, axis=1))}

# --- VISUALIZATION & REPORTING ---

def plot_confusion_matrix(ground_truth, predicted_labels, save_path, current_labels):
    unique_gt = sorted(np.unique(ground_truth))
    unique_pred = sorted(np.unique(predicted_labels))
    gt_names = [current_labels.get(i, f"GT_{i}") for i in unique_gt]
    
    # Hungarian Alignment
    cm_for_solver = confusion_matrix(ground_truth, predicted_labels, labels=unique_gt + unique_pred)
    cm_subset = cm_for_solver[:len(unique_gt), len(unique_gt):]
    row_ind, col_ind = linear_sum_assignment(-cm_subset)
    pred_to_aligned = {unique_pred[c]: unique_gt[r] for r, c in zip(row_ind, col_ind)}
    
    aligned_labels = np.array([pred_to_aligned.get(l, -1) for l in predicted_labels])
    cm_post = confusion_matrix(ground_truth, aligned_labels, labels=unique_gt)
    cm_post_norm = cm_post.astype('float') / cm_post.sum(axis=1)[:, np.newaxis]

    # Print Report
    print("\n" + "="*40 + "\nCONFUSION MATRICES\n" + "="*40)
    print("\nPOST-ALIGNMENT (RAW):")
    print(pd.DataFrame(cm_post, index=gt_names, columns=gt_names))
    print("\nPOST-ALIGNMENT (NORMALIZED %):")
    print((pd.DataFrame(cm_post_norm, index=gt_names, columns=gt_names) * 100).round(2))

    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm_post, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=gt_names, yticklabels=gt_names)
    axes[0].set_title("Aligned Raw Counts")
    sns.heatmap(cm_post_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1], xticklabels=gt_names, yticklabels=gt_names)
    axes[1].set_title("Aligned Normalized (Recall)")
    plt.savefig(os.path.join(save_path, 'confusion_analysis.png'))
    plt.close()

    return cm_post, cm_post_norm, pred_to_aligned

# --- MAIN EVALUATION CALL ---

def compare_clusters(technique, cluster_method, n_clusters, use_gpu, out_dir, combine_labels=False):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    latents = np.load(os.path.join(out_dir, "latents.npy"))
    states = np.load(os.path.join(out_dir, "states.npy")).squeeze()

    # Apply Label Combination Option
    if combine_labels:
        print("\n[INFO] Combining labels into Binary: Fresh (1,2) vs Degraded (3,4)")
        states = np.array([1 if x <= 2 else 2 for x in states])
        current_labels = STATE_LABELS_2
        current_colors = STATE_COLORS_2
        n_clusters = 2 
    else:
        current_labels = STATE_LABELS_4
        current_colors = STATE_COLORS_4

    # Run actual clustering and plotting
    from helper_functions import cluster_and_plot
    _, _, emb, labels = cluster_and_plot(
        latent=latents, technique=technique, n_clusters=n_clusters, 
        save_path=out_dir, use_gpu=use_gpu, cluster_method=cluster_method,
        ground_truth_labels=states, state_labels=current_labels, 
        state_colors=current_colors, imgs_dir="data/raw/craters_for_danny"
    )

    # 1. Alignment and Confusion Matrix
    cm, cm_norm, label_mapping = plot_confusion_matrix(states, labels, out_dir, current_labels)

    aligned_labels = np.array([label_mapping.get(l, -1) for l in labels])

    # 2. Geometry and Structural Check
    geom_results = compare_cluster_geometry(emb, aligned_labels, states, label_mapping)
    
    # 3. Boundary Uncertainty
    boundary_results = analyze_boundary_uncertainty(emb, states)

    # 4. Compile Results for the Summary Printout
    results = {
        'standard_metrics': {
            'ari': adjusted_rand_score(states, aligned_labels),
            'nmi': normalized_mutual_info_score(states, aligned_labels)
        },
        'structural_similarity': {
            'procrustes_disparity': geom_results['procrustes_disparity'],
            'distance_correlation': geom_results['distance_correlation'],
            'v_measure': v_measure_score(states, labels),
            'pred_silhouette': silhouette_score(emb, labels),
            'gt_silhouette': silhouette_score(emb, states)
        },
        'continuum_evidence': {
            'boundary_uncertainty': boundary_results['boundary_uncertainty']
        }
    }

    # Final Summary Printout
    print("\n=== SUMMARY ===")
    print(f"ARI: {results['standard_metrics']['ari']:.3f}")
    print(f"Procrustes disparity: {results['structural_similarity']['procrustes_disparity']:.4f}")
    print(f"Distance correlation: {results['structural_similarity']['distance_correlation']:.3f}")
    print(f"V-Measure: {results['structural_similarity']['v_measure']:.3f}")
    print(f"Silhouette (predicted): {results['structural_similarity']['pred_silhouette']:.3f}")
    print(f"Silhouette (ground truth): {results['structural_similarity']['gt_silhouette']:.3f}")
    print(f"Boundary uncertainty: {results['continuum_evidence']['boundary_uncertainty']:.3f}")
    
    return results

def display_craters_by_cluster(cluster_labels, imgs_dir, 
                                samples_per_cluster=5, figsize=(15, 10),
                                save_path=None):
    """
    Display sample crater images from each cluster in a grid layout.
    
    Args:
        cluster_labels: Array of predicted cluster labels (from clustering)
        imgs_dir: Directory containing crater images (.png files)
        samples_per_cluster: Number of samples to show from each cluster
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    
    # Get list of image files
    img_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith('.png')])
    
    if len(img_files) != len(cluster_labels):
        print(f"WARNING: Found {len(img_files)} images but cluster_labels has {len(cluster_labels)} entries")
        print("Using minimum length...")
        min_len = min(len(img_files), len(cluster_labels))
        img_files = img_files[:min_len]
        cluster_labels = cluster_labels[:min_len]
    
    # Get unique clusters
    unique_clusters = sorted(np.unique(cluster_labels))
    n_clusters = len(unique_clusters)
    
    print(f"Found {n_clusters} clusters")
    
    # Create figure
    fig, axes = plt.subplots(n_clusters, samples_per_cluster, 
                            figsize=figsize)
    
    # Handle case of single cluster or single sample
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    if samples_per_cluster == 1:
        axes = axes.reshape(-1, 1)
    
    # For each cluster
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        # Get indices of samples in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        print(f"\nCluster {cluster_id}: {len(cluster_indices)} samples")
        
        # Randomly sample from this cluster
        n_samples = min(samples_per_cluster, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        
        # Display samples
        for col_idx, sample_idx in enumerate(sampled_indices):
            ax = axes[cluster_idx, col_idx]
            
            # Get image filename
            img_file = img_files[sample_idx]
            img_path = os.path.join(imgs_dir, img_file)
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('L')  # Grayscale
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Cluster: {cluster_id}", fontsize=10)
            else:
                ax.text(0.5, 0.5, f"Image not found", 
                       ha='center', va='center')
                ax.set_title(f"Cluster: {cluster_id}", fontsize=10)
            
            ax.axis('off')
        
        # Fill remaining slots if we have fewer samples than requested
        for col_idx in range(n_samples, samples_per_cluster):
            ax = axes[cluster_idx, col_idx]
            ax.axis('off')
        
        # Add cluster label on the left
        axes[cluster_idx, 0].set_ylabel(f'Cluster {cluster_id}', 
                                       fontsize=12, rotation=0, 
                                       ha='right', va='center')
    
    plt.suptitle('Sample Craters by Cluster', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cluster_parser = subparsers.add_parser("cluster_compare")
    cluster_parser.add_argument("--n_clusters", type=int, default=4)
    cluster_parser.add_argument("--out_dir", type=str, required=True)
    cluster_parser.add_argument("--cluster_method", choices=["kmeans", "gmm"], default="kmeans")
    cluster_parser.add_argument("--technique", choices=["pca", "tsne", "umap"], default="pca")
    cluster_parser.add_argument("--use_gpu", action="store_true")
    
    args = parser.parse_args()
    if args.command == "cluster_compare":
        compare_clusters(args.technique, args.cluster_method, args.n_clusters, args.use_gpu, args.out_dir)