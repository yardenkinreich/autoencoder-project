"""
GPU-Accelerated PCA + KMeans Clustering for Lunar Craters
Run with: python pca_kmeans_gpu.py
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal use
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# ====== CONFIGURATION - UPDATE THESE ======
DATASET_PATH    = 'data/processed_new/cae/craters.npy'
METADATA_PATH   = 'data/processed_new/cae/metadata.csv'
OUTPUT_DIR          = 'logs/pca_only_znormalized/first_component_removed'  # Change to 'logs/pca_kmeans_gpu' for clustering results
OUTPUT_CSV          = f'{OUTPUT_DIR}/clusters.csv'
OUTPUT_GEOJSON      = f'{OUTPUT_DIR}/clusters.geojson'
OUTPUT_PLOT_PCA     = f'{OUTPUT_DIR}/plot_pca.png'
OUTPUT_PLOT_MAP     = f'{OUTPUT_DIR}/plot_spatial.png'
OUTPUT_PCA_LATENTS  = f'{OUTPUT_DIR}/pca_latents.npy'       # Full PCA transformed data
OUTPUT_PCA_MODEL    = f'{OUTPUT_DIR}/pca_model.pkl'          # Fitted PCA model
OUTPUT_SCALER_MODEL = f'{OUTPUT_DIR}/scaler_model.pkl'       # Fitted scaler
OUTPUT_LABELS       = f'{OUTPUT_DIR}/cluster_labels.npy'     # Cluster labels

N_COMPONENTS    = 50       # PCA components
N_CLUSTERS      = 4        # KMeans clusters
REMOVE_PC1      = True    # Set True to remove first PCA component before KMeans
CHUNK_SIZE      = 10000    # Images loaded from disk at a time
SAMPLE_SIZE     = 100000   # Samples used to FIT PCA (sklearn CPU, no GPU OOM risk)
RANDOM_SEED     = 42
IMAGE_CHANNELS  = 1        # 1 for CAE grayscale, 3 for MAE
IMAGE_SIZE      = 224
# ==========================================


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_gpu():
    """Check GPU availability and memory"""
    try:
        import cupy as cp
        mem_free = cp.cuda.Device().mem_info[0] / 1e9
        mem_total = cp.cuda.Device().mem_info[1] / 1e9
        print(f"✅ GPU available!")
        print(f"   Memory free:  {mem_free:.1f} GB")
        print(f"   Memory total: {mem_total:.1f} GB")
        return True, mem_free, mem_total
    except Exception as e:
        print(f"❌ GPU not available: {e}")
        return False, 0, 0


def load_data(dataset_path):
    """Load crater dataset via memmap"""
    print(f"Loading dataset: {dataset_path}")
    file_size = os.path.getsize(dataset_path)
    N = file_size // (IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * 4)
    data = np.memmap(
        dataset_path,
        dtype=np.float32,
        mode="r",
        shape=(N, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    )
    print(f"Total craters: {N:,}")
    print(f"Shape: {data.shape}")
    return data, N


def fit_pca_gpu(data, N, sample_size, n_components):
    from sklearn.decomposition import PCA as skPCA

    print(f"Fitting PCA on {sample_size:,} sample (of {N:,} total)...")
    np.random.seed(RANDOM_SEED)
    sample_idx = np.random.choice(N, size=sample_size, replace=False)
    sample_idx.sort()

    n_features = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS
    
    # Initialize X_scaled - This was missing in your version!
    X_scaled = np.zeros((sample_size, n_features), dtype=np.float32)

    # Soft Z-Normalization Constant
    SOFT_CONSTANT = 0.1 

    print("Applying Local Soft Z-Normalization to sample...")
    for i in range(0, sample_size, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, sample_size)
        batch_idx = sample_idx[i:end]
        batch_cpu = data[batch_idx].reshape(end - i, -1).astype(np.float32)
        
        # Calculate local stats per crater patch
        mu = batch_cpu.mean(axis=1, keepdims=True)
        sigma = batch_cpu.std(axis=1, keepdims=True)
        
        # Apply normalization
        X_scaled[i:end] = (batch_cpu - mu) / (sigma + SOFT_CONSTANT)
        
        if i % 20000 == 0:
            print(f"  Processed: {end:,}/{sample_size:,}")

    print(f"Fitting PCA with {n_components} components...")
    pca = skPCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_SEED)
    pca.fit(X_scaled)
    del X_scaled

    var_explained = pca.explained_variance_ratio_
    # Return dummy values for scaler since we are now using local stats
    return pca, (None, None), var_explained


def transform_all_gpu(data, N, pca, scaler, n_components, remove_pc1=False):
    """Transform all data using fitted PCA"""
    import cupy as cp

    actual_components = n_components - 1 if remove_pc1 else n_components
    print(f"Transforming all {N:,} craters...")
    if remove_pc1:
        print(f"  (PC1 will be removed - using components 2-{n_components})")

    # Check if PCA results fit on GPU
    pca_bytes = N * actual_components * 4
    gpu_free = cp.cuda.Device().mem_info[0]
    store_gpu = pca_bytes < gpu_free * 0.7

    if store_gpu:
        print(f"  Storing PCA results on GPU ({pca_bytes/1e9:.2f} GB)")
        X_pca_all = cp.zeros((N, actual_components), dtype=cp.float32)
    else:
        print(f"  Storing PCA results on CPU ({pca_bytes/1e9:.2f} GB)")
        X_pca_all = np.zeros((N, actual_components), dtype=np.float32)

    mean_, std_ = scaler  # Unpack CPU scaler tuple

    SOFT_CONSTANT = 0.1

    for i in range(0, N, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, N)
        batch_cpu = data[i:end].reshape(end - i, -1).astype(np.float32)
        
        # Apply Local Normalization (SAME as fitting step)
        mu = batch_cpu.mean(axis=1, keepdims=True)
        sigma = batch_cpu.std(axis=1, keepdims=True)
        batch_scaled = (batch_cpu - mu) / (sigma + SOFT_CONSTANT)
        
        batch_pca = pca.transform(batch_scaled) 

        if remove_pc1:
            batch_pca = batch_pca[:, 1:]

        if store_gpu:
            X_pca_all[i:end] = cp.array(batch_pca)  # move result to GPU
        else:
            X_pca_all[i:end] = batch_pca

        del batch_cpu, batch_scaled, batch_pca

        if i % 100000 == 0:
            print(f"  Transforming: {end:,}/{N:,} ({100*end/N:.1f}%)")

    print(f"✅ Transform complete! Shape: {X_pca_all.shape}")
    return X_pca_all, store_gpu


def run_kmeans_gpu(X_pca_all, store_gpu, n_clusters):
    """Run KMeans on GPU"""
    import cupy as cp
    from cuml.cluster import KMeans as cumlKMeans

    print(f"Running KMeans with {n_clusters} clusters on GPU...")

    if store_gpu:
        X_for_kmeans = X_pca_all
    else:
        print("  Moving PCA results to GPU...")
        X_for_kmeans = cp.array(X_pca_all)

    kmeans = cumlKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        max_iter=300,
        n_init=10
    )
    kmeans.fit(X_for_kmeans)
    labels = kmeans.labels_.get().astype(np.int32)

    print(f"✅ KMeans complete!")
    print(f"\nCluster distribution:")
    for c in range(n_clusters):
        count = (labels == c).sum()
        print(f"  Cluster {c}: {count:,} ({100*count/len(labels):.1f}%)")

    return labels, kmeans


def fit_pca_cpu(data, N, sample_size, n_components):
    from sklearn.decomposition import IncrementalPCA
    
    print(f"Fitting IncrementalPCA on CPU with {n_components} components...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=CHUNK_SIZE)
    
    SOFT_CONSTANT = 0.1

    # We skip the "Pass 1/2: Fitting scaler" entirely now!
    print("Fitting PCA with Local Normalization...")
    for i in range(0, N, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, N)
        batch = data[i:end].reshape(end - i, -1).astype(np.float32)
        
        # Local Normalization
        mu = batch.mean(axis=1, keepdims=True)
        sigma = batch.std(axis=1, keepdims=True)
        batch_scaled = (batch - mu) / (sigma + SOFT_CONSTANT)
        
        ipca.partial_fit(batch_scaled)
        
        if i % 100000 == 0:
            print(f"  PCA Progress: {end:,}/{N:,}")

    var_explained = ipca.explained_variance_ratio_
    print(f"✅ PCA fitted! Variance: {var_explained.sum()*100:.2f}%")

    return ipca, (None, None), var_explained


def transform_all_cpu(data, N, pca, scaler_params, n_components, remove_pc1=False):
    actual_components = n_components - 1 if remove_pc1 else n_components
    X_pca_all = np.zeros((N, actual_components), dtype=np.float32)
    
    SOFT_CONSTANT = 0.1

    for i in range(0, N, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, N)
        batch = data[i:end].reshape(end - i, -1).astype(np.float32)
        
        # Local Normalization
        mu = batch.mean(axis=1, keepdims=True)
        sigma = batch.std(axis=1, keepdims=True)
        batch_scaled = (batch - mu) / (sigma + SOFT_CONSTANT)
        
        batch_pca = pca.transform(batch_scaled)
        
        if remove_pc1:
            batch_pca = batch_pca[:, 1:]
            
        X_pca_all[i:end] = batch_pca
        
        if i % 100000 == 0:
            print(f"  Transforming: {end:,}/{N:,}")

    return X_pca_all, False


def run_kmeans_cpu(X_pca_all, n_clusters):
    """Fallback: Run MiniBatchKMeans on CPU"""
    from sklearn.cluster import MiniBatchKMeans

    print(f"Running MiniBatchKMeans on CPU with {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        batch_size=CHUNK_SIZE,
        n_init=10,
        max_iter=100
    )
    kmeans.fit(X_pca_all)
    labels = kmeans.labels_.astype(np.int32)

    print(f"✅ KMeans complete!")
    for c in range(n_clusters):
        count = (labels == c).sum()
        print(f"  Cluster {c}: {count:,} ({100*count/len(labels):.1f}%)")

    return labels, kmeans


def save_results(metadata, labels, N):
    """Save CSV and GeoJSON results"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine ID column
    id_col = 'id' if 'id' in metadata.columns else \
             'CRATER_ID' if 'CRATER_ID' in metadata.columns else None

    # Save CSV
    df_dict = {
        'lat': metadata['lat'].values if 'lat' in metadata.columns else metadata['LAT_CIRC_IMG'].values,
        'lon': metadata['lon'].values if 'lon' in metadata.columns else metadata['LON_CIRC_IMG'].values,
        'cluster': labels
    }
    if id_col:
        df_dict['crater_id'] = metadata[id_col].values
    if 'x' in metadata.columns:
        df_dict['x'] = metadata['x'].values
        df_dict['y'] = metadata['y'].values

    df = pd.DataFrame(df_dict)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved CSV: {OUTPUT_CSV}")

    # Save GeoJSON
    if 'x' in metadata.columns and 'y' in metadata.columns:
        geometry = [Point(xy) for xy in zip(metadata['x'].values, metadata['y'].values)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="ESRI:103881")
        gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
        print(f"✅ Saved GeoJSON: {OUTPUT_GEOJSON}")
    else:
        print("⚠️  No x/y columns found - skipping GeoJSON")

    return df


def save_plots(X_pca_all, store_gpu, labels, metadata, var_explained, N):
    """Save visualization plots"""
    CLUSTER_COLORS = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange',
                      4: 'purple', 5: 'brown', 6: 'pink', 7: 'gray'}

    # Get 2D coords for visualization
    n_viz = min(50000, N)
    np.random.seed(RANDOM_SEED)
    viz_idx = np.random.choice(N, size=n_viz, replace=False)

    if store_gpu:
        import cupy as cp
        viz_pca = X_pca_all[viz_idx, :2].get()
    else:
        viz_pca = X_pca_all[viz_idx, :2]

    viz_labels = labels[viz_idx]

    # Plot 1: PCA scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(N_CLUSTERS):
        mask = viz_labels == c
        ax.scatter(
            viz_pca[mask, 0], viz_pca[mask, 1],
            c=CLUSTER_COLORS.get(c, 'gray'),
            label=f'Cluster {c} ({(labels==c).sum():,})',
            alpha=0.5, s=5
        )
    pc1_var = var_explained[0 if not REMOVE_PC1 else 1] * 100
    pc2_var = var_explained[1 if not REMOVE_PC1 else 2] * 100
    ax.set_xlabel(f"PC{'1' if not REMOVE_PC1 else '2'} ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC{'2' if not REMOVE_PC1 else '3'} ({pc2_var:.1f}%)")
    ax.set_title(
        f"PCA + KMeans (k={N_CLUSTERS})\n"
        f"{n_viz:,} sample from {N:,} total craters"
        f"{' | PC1 removed' if REMOVE_PC1 else ''}"
    )
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PCA, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved PCA plot: {OUTPUT_PLOT_PCA}")

    # Plot 2: Spatial distribution
    lat_col = 'lat' if 'lat' in metadata.columns else 'LAT_CIRC_IMG'
    lon_col = 'lon' if 'lon' in metadata.columns else 'LON_CIRC_IMG'

    fig, ax = plt.subplots(figsize=(16, 8))
    for c in range(N_CLUSTERS):
        mask = labels == c
        ax.scatter(
            metadata[lon_col].values[mask],
            metadata[lat_col].values[mask],
            c=CLUSTER_COLORS.get(c, 'gray'),
            label=f'Cluster {c} ({mask.sum():,})',
            alpha=0.3, s=1
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Distribution of KMeans Clusters\n({N:,} total craters)")
    ax.legend(markerscale=5, loc='upper right')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_MAP, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spatial plot: {OUTPUT_PLOT_MAP}")


def main():
    print_section("GPU PCA + KMeans Crater Clustering")
    print(f"Config:")
    print(f"  Dataset:     {DATASET_PATH}")
    print(f"  Metadata:    {METADATA_PATH}")
    print(f"  PCA comps:   {N_COMPONENTS}")
    print(f"  Clusters:    {N_CLUSTERS}")
    print(f"  Remove PC1:  {REMOVE_PC1}")
    print(f"  Sample size: {SAMPLE_SIZE:,}")

    # Create all output directories upfront
    for d in set([OUTPUT_DIR, os.path.dirname(OUTPUT_PCA_LATENTS),
                  os.path.dirname(OUTPUT_PCA_MODEL), os.path.dirname(OUTPUT_CSV)]):
        if d:
            os.makedirs(d, exist_ok=True)
    print(f"Output directory ready: {OUTPUT_DIR}")

    # Check GPU
    print_section("Checking GPU")
    gpu_available, mem_free, mem_total = check_gpu()

    # Load data
    print_section("Loading Data")
    data, N = load_data(DATASET_PATH)

    # Fit PCA
    print_section("Fitting PCA")
    if gpu_available:
        pca, scaler, var_explained = fit_pca_gpu(data, N, SAMPLE_SIZE, N_COMPONENTS)
    else:
        pca, scaler, var_explained = fit_pca_cpu(data, N, SAMPLE_SIZE, N_COMPONENTS)

    # Transform all data (or load if already saved)
    print_section("Transforming All Data")

    if os.path.exists(OUTPUT_PCA_LATENTS):
        print(f"Found existing PCA latents: {OUTPUT_PCA_LATENTS}")
        print(f"   Loading instead of recomputing...")
        X_pca_all = np.load(OUTPUT_PCA_LATENTS)
        store_gpu = False
        print(f"   Shape: {X_pca_all.shape}")
        print(f"   Delete {OUTPUT_PCA_LATENTS} to force recompute")
    else:
        if gpu_available:
            X_pca_all, store_gpu = transform_all_gpu(data, N, pca, scaler, N_COMPONENTS, REMOVE_PC1)
        else:
            X_pca_all, store_gpu = transform_all_cpu(data, N, pca, scaler, N_COMPONENTS, REMOVE_PC1)

        # Save PCA latents
        os.makedirs(os.path.dirname(OUTPUT_PCA_LATENTS), exist_ok=True)
        print(f"Saving PCA latents to {OUTPUT_PCA_LATENTS}...")
        save_arr = X_pca_all.get() if store_gpu else X_pca_all
        np.save(OUTPUT_PCA_LATENTS, save_arr)
        print(f"Saved PCA latents: shape={X_pca_all.shape}")

        # Save PCA model
        import pickle
        with open(OUTPUT_PCA_MODEL, 'wb') as f:
            pickle.dump(pca, f)
        # Save scaler as mean/std npy files (works for both GPU and CPU scalers)
        mean_, std_ = scaler if isinstance(scaler, tuple) else (scaler.mean_, scaler.scale_)
        np.save(OUTPUT_SCALER_MODEL.replace('.pkl', '_mean.npy'), mean_)
        np.save(OUTPUT_SCALER_MODEL.replace('.pkl', '_std.npy'), std_)
        print(f"Saved PCA model:    {OUTPUT_PCA_MODEL}")
        print(f"Saved scaler mean:  {OUTPUT_SCALER_MODEL.replace('.pkl', '_mean.npy')}")
        print(f"Saved scaler std:   {OUTPUT_SCALER_MODEL.replace('.pkl', '_std.npy')}")

    # Run KMeans (or load if already saved)
    print_section("Running KMeans")

    if os.path.exists(OUTPUT_LABELS):
        print(f"Found existing cluster labels: {OUTPUT_LABELS}")
        print(f"   Loading instead of recomputing...")
        labels = np.load(OUTPUT_LABELS)
        print(f"   Delete {OUTPUT_LABELS} to force rerun KMeans only")
        print(f"Cluster distribution:")
        for c in range(N_CLUSTERS):
            count = (labels == c).sum()
            print(f"  Cluster {c}: {count:,} ({100*count/len(labels):.1f}%)")
    else:
        if gpu_available:
            labels, kmeans = run_kmeans_gpu(X_pca_all, store_gpu, N_CLUSTERS)
        else:
            labels, kmeans = run_kmeans_cpu(X_pca_all, N_CLUSTERS)
        np.save(OUTPUT_LABELS, labels)
        print(f"Saved cluster labels: {OUTPUT_LABELS}")

    # Load metadata and save results
    print_section("Saving Results")
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Metadata columns: {metadata.columns.tolist()}")
    df = save_results(metadata, labels, N)

    # Save plots
    print_section("Saving Plots")
    save_plots(X_pca_all, store_gpu, labels, metadata, var_explained, N)

    # Cleanup GPU memory
    if gpu_available:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()

    print_section("COMPLETE!")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print(f"  {OUTPUT_PCA_LATENTS}  <- loaded automatically next run")
    print(f"  {OUTPUT_PCA_MODEL}")
    print(f"  {OUTPUT_SCALER_MODEL}")
    print(f"  {OUTPUT_LABELS}  <- loaded automatically next run")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_GEOJSON}")
    print(f"  {OUTPUT_PLOT_PCA}")
    print(f"  {OUTPUT_PLOT_MAP}")
    print(f"\nLoad {OUTPUT_GEOJSON} in QGIS to see spatial cluster distribution!")
    print(f"\nTo force rerun:")
    print(f"  Everything:  delete {OUTPUT_PCA_LATENTS}")
    print(f"  KMeans only: delete {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()