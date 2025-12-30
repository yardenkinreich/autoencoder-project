import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import argparse
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import sys
from sklearn.metrics import confusion_matrix
import pandas as pd

# --- Project Root Configuration ---
# Add the project root to Python's module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_functions import cluster_and_plot 

# --- Normalization Constants (Must match preprocess.py) ---
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def reverse_normalization(tensor: np.ndarray) -> np.ndarray:
    """
    Reverses ImageNet normalization on a NumPy array (C, H, W) 
    and converts it into a displayable NumPy array (H, W, C) in the range [0, 1].
    """
    denorm_tensor = torch.from_numpy(tensor)
    denorm_tensor = denorm_tensor * IMAGENET_STD + IMAGENET_MEAN
    img_np = denorm_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    return img_np

def verify_preprocess_integrity(npy_path, csv_path, crops_dir, output_png , num_samples=4):
    """
    Verifies the preprocess by loading random samples from the .npy file,
    denormalizing them, and plotting them next to their corresponding JPEG crop
    found via the metadata CSV.
    
    Applies the standard model transforms (Resize/Crop) to the JPEG for valid comparison.
    """
    print(f"\n--- Pipeline Verification ---")
    
    # --- View Transformation (For Reference JPEG) ---
    # Applies the exact same geometry steps as the model pipeline, but without normalization.
    view_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # 1. Load Memmap (Explicitly calculating shape as per user workflow)
    if not os.path.exists(npy_path):
        print(f"Error: NPY file not found at {npy_path}")
        return
    print(f"1. Loading NPY: {npy_path}")
    
    # Calculate N based on file size (3 channels, 224x224, 4 bytes for float32)
    file_size = os.path.getsize(npy_path)
    bytes_per_image = 224 * 224 * 3 * 4
    N = file_size // bytes_per_image
    
    if N == 0:
        print("Error: File size is too small to contain any images of shape (3, 224, 224).")
        return

    # Load using np.memmap (Raw binary read)
    data = np.memmap(
        npy_path,
        dtype=np.float32,
        mode="r",
        shape=(N, 3, 224, 224)
    )
    print(f"   Data Shape: {data.shape}")

    # 2. Load CSV to map Index -> Crater ID
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    print(f"2. Loading Metadata: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Metadata Rows: {len(df)}")

    # Safety check
    if len(data) != len(df):
        print(f"WARNING: Sample count mismatch! NPY: {len(data)} vs CSV: {len(df)}")
        print("Indices might not align if files are from different runs.")

    # 3. Pick Random Indices
    indices = np.random.choice(min(len(data), len(df)), num_samples, replace=False)

    # 4. Plot
    fig, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 8))
    if len(indices) == 1: axes = axes.reshape(2, 1)

    print(f"3. Comparing {num_samples} samples...")

    for i, idx in enumerate(indices):
        # Get ID from CSV (Column 'id' based on your save_crater_metadata function)
        try:
            crater_id = df.iloc[idx]['id']
        except KeyError:
            # Fallback if column name is different (e.g. no header)
            crater_id = df.iloc[idx, 0] 

        # --- ROW 1: Original NPY Crop (Reference) ---
        raw_path = os.path.join(crops_dir, f"{crater_id}.npy")
        crater_img = np.load(raw_path)         # raw crop
        img_pil = Image.fromarray(crater_img.astype(np.uint8)).convert("RGB")
        
        if os.path.exists(raw_path):
            # Apply the transform (Resize -> CenterCrop) to match the .npy geometry
            img_tensor = view_transform(img_pil)
            img_view_ref = img_tensor.permute(1, 2, 0).numpy()
            
            ax_top = axes[0, i]
            ax_top.imshow(img_view_ref)
            ax_top.set_title(f"ID: {crater_id}\n(NPY -> Processed)")
        else:
            ax_top.text(0.5, 0.5, "NPY Missing", ha='center')
            ax_top.set_title(f"ID: {crater_id}\n(Not Found)")
        ax_top.axis('off')

        # --- ROW 2: Denormalized NPY Data ---
        ax_bot = axes[1, i]
        
        # Load from NPY and denormalize
        npy_sample = np.array(data[idx]) # Read from mmap to memory
        img_view = reverse_normalization(npy_sample)
        
        ax_bot.imshow(img_view)
        ax_bot.set_title(f"Index: {idx}\n(From NPY Processed -> Denorm)")
        ax_bot.axis('off')

    plt.tight_layout()
    print("   Saving comparison plot to preprocess_verification.png...")
    plt.savefig(output_png, dpi=200)
    

def evaluate_model_edge_cases():
    """
    Placeholder function for evaluating model performance on specific, challenging
    edge cases (e.g., highly obscured craters, complex terrain, etc.).
    """
    #print("--- Running Edge Case Evaluation (Placeholder) ---")
    # Add logic here to load specific datasets and generate model predictions
    # E.g.: 
    # results = model.predict(edge_case_data)
    # print(f"Average confidence on edge cases: {np.mean(results.confidence)}")
    pass


def compare_clusters(technique, cluster_method, n_clusters, use_gpu, out_dir):
    """
    Comprehensive clustering evaluation with structural similarity metrics.
    """
    
    print("=" * 60)
    print("CLUSTERING EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)
    
    # Load data
    latents = np.load(os.path.join(out_dir, "latents.npy"))
    states = np.load(os.path.join(out_dir, "states.npy")).squeeze()
    
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
    
    # Clustering
    fig_reg, fig_gt, emb, labels = cluster_and_plot(
        latent=latents,
        technique=technique,
        n_clusters=n_clusters,
        save_path=out_dir,
        use_gpu=use_gpu,
        cluster_method=cluster_method,
        imgs_dir="data/raw/craters_for_danny",
        ground_truth_labels=states,
        state_labels=STATE_LABELS,
        state_colors=STATE_COLORS
    )
    
    # Standard metrics
    ari = adjusted_rand_score(states, labels)
    nmi = normalized_mutual_info_score(states, labels)
    
    print("\n" + "=" * 60)
    print("STANDARD CLUSTERING METRICS")
    print("=" * 60)
    print(f"ARI: {ari:.3f} | NMI: {nmi:.3f}")
    
    # NEW: Structural similarity metrics
    geometry_results = compare_cluster_geometry(emb, labels, states, n_clusters)
    quality_results = evaluate_cluster_structure(labels, states)
    internal_results = internal_cluster_quality(emb, labels, states)
    
    # NEW: Continuum evidence
    overlap_results = analyze_distribution_overlap(emb, labels, states, n_clusters)
    boundary_results = analyze_boundary_uncertainty(emb, states, STATE_LABELS)
    
    # Compile all results
    results = {
        'standard_metrics': {
            'ari': ari,
            'nmi': nmi,
        },
        'structural_similarity': {
            **geometry_results,
            **quality_results,
            **internal_results
        },
        'continuum_evidence': {
            'distribution_overlaps': overlap_results,
            'boundary_uncertainty': boundary_results['mean_entropy'],
            'uncertain_sample_pct': np.sum(boundary_results['uncertain_samples'])/len(states)*100
        }
    }
    
    # Save
    import json
    with open(os.path.join(out_dir, 'comprehensive_results.json'), 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'continuum_evidence'}, 
                  f, indent=2, default=str)
    
    return results

def compare_cluster_geometry(embedding, predicted_labels, ground_truth, n_clusters=4):
    """
    Compare geometric arrangement of predicted vs. ground truth clusters.
    Uses Procrustes analysis to measure structural similarity.
    """
    from scipy.spatial import procrustes
    from scipy.spatial.distance import cdist
    
    # Compute centroids for predicted clusters
    pred_centroids = np.array([
        embedding[predicted_labels == i].mean(axis=0) 
        for i in range(n_clusters)
    ])
    
    # Compute centroids for ground truth
    gt_centroids = np.array([
        embedding[ground_truth == state].mean(axis=0)
        for state in sorted(np.unique(ground_truth))
    ])
    
    # Procrustes analysis: measure shape similarity
    mtx1, mtx2, disparity = procrustes(pred_centroids, gt_centroids)
    
    print(f"\n{'='*60}")
    print("CLUSTER GEOMETRY COMPARISON")
    print(f"{'='*60}")
    print(f"Procrustes disparity: {disparity:.4f}")
    print("(Lower = more similar geometric arrangement, 0 = identical)")
    
    # Pairwise distances between centroids
    pred_dist_matrix = cdist(pred_centroids, pred_centroids, metric='euclidean')
    gt_dist_matrix = cdist(gt_centroids, gt_centroids, metric='euclidean')
    
    # Correlation of distance matrices (Mantel-like test)
    from scipy.stats import spearmanr
    
    # Flatten upper triangular (avoid diagonal and duplicates)
    pred_dists = pred_dist_matrix[np.triu_indices(n_clusters, k=1)]
    gt_dists = gt_dist_matrix[np.triu_indices(n_clusters, k=1)]
    
    corr, pval = spearmanr(pred_dists, gt_dists)
    
    print(f"\nInter-cluster distance correlation: {corr:.3f} (p={pval:.4f})")
    print("(Measures if relative cluster spacing is preserved)")
    
    return {
        'procrustes_disparity': disparity,
        'distance_correlation': corr,
        'distance_correlation_pvalue': pval,
        'pred_centroids': pred_centroids,
        'gt_centroids': gt_centroids
    }

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

def evaluate_cluster_structure(predicted_labels, ground_truth):
    """
    Evaluate clustering quality with metrics less sensitive to label noise.
    """
    homogeneity = homogeneity_score(ground_truth, predicted_labels)
    completeness = completeness_score(ground_truth, predicted_labels)
    v_measure = v_measure_score(ground_truth, predicted_labels)
    
    print(f"\n{'='*60}")
    print("CLUSTER QUALITY METRICS")
    print(f"{'='*60}")
    print(f"Homogeneity: {homogeneity:.3f}")
    print("  → Each cluster contains only members of a single class")
    print(f"Completeness: {completeness:.3f}")
    print("  → All members of a class are assigned to the same cluster")
    print(f"V-Measure: {v_measure:.3f}")
    print("  → Harmonic mean (balanced measure)")
    
    return {
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure
    }

from sklearn.metrics import silhouette_score, silhouette_samples

def internal_cluster_quality(embedding, predicted_labels, ground_truth):
    """
    Compare internal cluster quality metrics.
    High scores suggest well-separated clusters regardless of labels.
    """

    STATE_LABELS = {
            1: "New Crater",
            2: "Semi New Crater",
            3: "Semi Old Crater",
            4: "Old Crater"
        }

    # Silhouette for predicted clusters
    pred_silhouette = silhouette_score(embedding, predicted_labels)
    
    # Silhouette for ground truth (as if it were clustering)
    gt_silhouette = silhouette_score(embedding, ground_truth)
    
    print(f"\n{'='*60}")
    print("INTERNAL CLUSTER QUALITY (Silhouette Score)")
    print(f"{'='*60}")
    print(f"Predicted clusters: {pred_silhouette:.3f}")
    print(f"Ground truth labels: {gt_silhouette:.3f}")
    print(f"Difference: {abs(pred_silhouette - gt_silhouette):.3f}")
    print("\nInterpretation:")
    if abs(pred_silhouette - gt_silhouette) < 0.1:
        print("  ✓ Similar internal structure (clusters are comparably separated)")
    else:
        print(f"  → Predicted clustering is {'better' if pred_silhouette > gt_silhouette else 'worse'} separated")
    
    # Per-cluster silhouette
    pred_silhouettes = silhouette_samples(embedding, predicted_labels)
    gt_silhouettes = silhouette_samples(embedding, ground_truth)
    
    print("\nPer-cluster average silhouette:")
    print("Predicted clusters:")
    for i in range(len(np.unique(predicted_labels))):
        mask = predicted_labels == i
        print(f"  Cluster {i}: {pred_silhouettes[mask].mean():.3f}")
    
    print("Ground truth labels:")
    for state in sorted(np.unique(ground_truth)):
        mask = ground_truth == state
        label = STATE_LABELS.get(state, f"State {state}")
        print(f"  {label}: {gt_silhouettes[mask].mean():.3f}")
    
    return {
        'pred_silhouette': pred_silhouette,
        'gt_silhouette': gt_silhouette,
        'silhouette_difference': abs(pred_silhouette - gt_silhouette)
    }

def analyze_distribution_overlap(embedding, predicted_labels, ground_truth, n_clusters=4):
    """
    Analyze overlap between predicted and ground truth distributions.
    High overlap supports the "continuum" argument.
    """

    STATE_LABELS = {
        1: "New Crater",
        2: "Semi New Crater",
        3: "Semi Old Crater",
        4: "Old Crater"
    }

    
    from scipy.spatial.distance import jensenshannon
    from sklearn.neighbors import KernelDensity
    
    print(f"\n{'='*60}")
    print("DISTRIBUTION OVERLAP ANALYSIS")
    print(f"{'='*60}")
    
    # For each ground truth class, compute overlap with predicted clusters
    overlaps = []
    
    for state in sorted(np.unique(ground_truth)):
        gt_mask = ground_truth == state
        gt_points = embedding[gt_mask]
        
        # Which clusters contain this ground truth class?
        cluster_distribution = []
        for cluster in range(n_clusters):
            cluster_mask = predicted_labels == cluster
            overlap_count = np.sum(gt_mask & cluster_mask)
            cluster_distribution.append(overlap_count)
        
        cluster_distribution = np.array(cluster_distribution) / np.sum(cluster_distribution)
        
        # Entropy of distribution (high = spread across clusters = continuum)
        from scipy.stats import entropy
        overlap_entropy = entropy(cluster_distribution)
        
        label = STATE_LABELS.get(state, f"State {state}")
        print(f"\n{label}:")
        print(f"  Distribution across clusters: {cluster_distribution.round(3)}")
        print(f"  Entropy: {overlap_entropy:.3f} (max={np.log(n_clusters):.3f})")
        
        if overlap_entropy > 0.5:
            print(f"  → High entropy: spans multiple clusters (supports continuum)")
        
        overlaps.append({
            'state': state,
            'label': label,
            'distribution': cluster_distribution,
            'entropy': overlap_entropy
        })
    
    return overlaps

def analyze_boundary_uncertainty(embedding, ground_truth, state_labels):
    """
    Analyze uncertainty at boundaries between degradation states.
    Points near decision boundaries suggest continuum rather than discrete states.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_predict
    
    print(f"\n{'='*60}")
    print("DECISION BOUNDARY ANALYSIS")
    print(f"{'='*60}")
    
    # Train classifier to learn ground truth boundaries
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    
    # Cross-validated predictions with probabilities
    pred_proba = cross_val_predict(clf, embedding, ground_truth, 
                                    cv=5, method='predict_proba')
    
    # Entropy of predictions (high = uncertain classification)
    from scipy.stats import entropy
    prediction_entropy = entropy(pred_proba, axis=1)
    
    # For each ground truth class, find samples with high uncertainty
    print("\nClassification confidence by ground truth state:")
    for state in sorted(np.unique(ground_truth)):
        mask = ground_truth == state
        state_entropy = prediction_entropy[mask]
        
        max_prob = pred_proba[mask].max(axis=1)
        uncertain_pct = np.sum(max_prob < 0.6) / len(max_prob) * 100
        
        label = state_labels.get(state, f"State {state}")
        print(f"\n{label}:")
        print(f"  Mean prediction entropy: {state_entropy.mean():.3f}")
        print(f"  Samples with <60% confidence: {uncertain_pct:.1f}%")
        
        if uncertain_pct > 30:
            print(f"  → High uncertainty: boundaries are ambiguous")
    
    # Overall boundary ambiguity
    high_uncertainty_mask = prediction_entropy > np.percentile(prediction_entropy, 75)
    
    print(f"\n{'='*60}")
    print(f"Overall: {np.sum(high_uncertainty_mask)} samples ({np.sum(high_uncertainty_mask)/len(ground_truth)*100:.1f}%) ")
    print(f"have high classification uncertainty (top 25%)")
    print("This suggests substantial ambiguity in manual category boundaries.")
    
    return {
        'prediction_entropy': prediction_entropy,
        'uncertain_samples': high_uncertainty_mask,
        'mean_entropy': prediction_entropy.mean()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Data Preprocessing Check
    check_parser = subparsers.add_parser("data_check", help="Run data preprocessing verification")
    check_parser.add_argument("--npy-path", type=str, required=True, help="Path to the .npy file containing preprocessed images.")
    check_parser.add_argument("--csv-path", type=str, required=True, help="Path to the CSV file with image metadata.")
    check_parser.add_argument("--crops-dir", type=str, required=True, help="Directory containing original JPEG crops.")
    check_parser.add_argument("--num-samples", type=int, default=4, help="Number of random samples to verify.")
    check_parser.add_argument("--output-png", type=str, default="preprocess_verification.png", help="Output PNG file for the verification plot.")

    # Model Cluster Comparison
    cluster_parser = subparsers.add_parser("cluster_compare",help="Tool for image preprocessing verification and model latent space evaluation.")
    cluster_parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters (should match ground truth states)")
    cluster_parser.add_argument("--out_df", type=str, default="", help="Path to save output CSV with clustering metrics. Required to run cluster comparison.")
    cluster_parser.add_argument("--cluster_method", type=str, choices=["kmeans", "gmm"], default="kmeans", help="Clustering algorithm to use.")
    cluster_parser.add_argument("--technique", type=str, choices=["pca", "tsne"], default="pca", help="Dimensionality reduction technique for visualization.")
    cluster_parser.add_argument("--use_gpu", action="store_true", help="Use GPU for computations if available.")
    
    args = parser.parse_args()

    # 1. Run Data Preprocessing Check if requested
    if args.command == "data_check":
        print("\n==============================================")
        print("Running Data Preprocessing Check")
        print("==============================================")
        verify_preprocess_integrity(
            npy_path=args.npy_path,
            csv_path=args.csv_path,
            crops_dir=args.crops_dir,
            num_samples=args.num_samples,
            output_png=args.output_png
        )
        print("==============================================")

    # 2. Run Model Evaluation/Clustering if required arguments are provided
    if args.command == "cluster_compare":
        print("\n==============================================")
        print("Running Model Cluster Comparison")
        print("==============================================")
        compare_clusters(
            technique=args.technique,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            use_gpu=args.use_gpu,
            out_df=args.out_df
        )
        print("==============================================")
    # 3. Run Edge Case Evaluation
    if args.command == "edge_case_eval":
        print("\n==============================================")
        print("Running Model Edge Case Evaluation")
        print("==============================================")
        evaluate_model_edge_cases() 