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


def compare_clusters(technique, cluster_method, num_clusters, use_gpu, out_dir):
    """Compare your clustering results to Julie's ground truth clusters."""

    print("--- Running Cluster Comparison ---")
    
    # Load latent representations and ground-truth labels
    latents = np.load(os.path.join(out_dir, "latents_julie_with_flip.npy"))  # your latent vectors
    states = np.load(os.path.join(out_dir, "states_julie_with_flip.npy"))     # ground truth labels (1–4)

    # Run clustering (returns predicted cluster labels)
    fig_reg, emb, labels = cluster_and_plot(
        latent=latents,
        technique=technique,
        n_clusters=num_clusters,
        save_path=out_dir,
        use_gpu=use_gpu,
        cluster_method=cluster_method,
        imgs_dir="data/raw/craters_for_danny"
    )
    
    # Save the figures
    fig_reg.savefig(os.path.join(out_dir, "cluster_regular.png"))

    # Compute similarity metrics (labels vs ground truth)
    ari = adjusted_rand_score(states, labels)
    nmi = normalized_mutual_info_score(states, labels)
    fmi = fowlkes_mallows_score(states, labels)



    ct = pd.crosstab(states, labels)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    print(ct)
    print(ct_norm)

    from scipy.stats import spearmanr
    spearmanr(labels, states)

    print(f"\nCluster similarity with Julie's ground truth:")
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
    print(f"Fowlkes–Mallows Index (FMI): {fmi:.3f}")

    # Save results
    os.makedirs(out_dir, exist_ok=True)  # ✓ Create the directory
    metrics_file = os.path.join(out_dir, "cluster_metrics.csv")  # ✓ File path
    with open(metrics_file, "w") as f:
        f.write("metric,value\n")
        f.write(f"ARI,{ari:.4f}\n")
        f.write(f"NMI,{nmi:.4f}\n")
        f.write(f"FMI,{fmi:.4f}\n")

    print(f"\nSaved metrics to {metrics_file}")

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
    cluster_parser.add_argument("--num_clusters", type=int, default=4, help="Number of clusters (should match ground truth states)")
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
            num_clusters=args.num_clusters,
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