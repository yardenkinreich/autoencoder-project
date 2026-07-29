"""
Lunar Mosaic Brightness Normalization
Applies two methods: WAC Hapke albedo correction and local high-pass filtering
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter


def load_image(filepath):
    with rasterio.open(filepath) as src:
        return src.read(1)  # single band, returns uint8 numpy array
    
    raise ValueError(f"Unsupported file format: {filepath.suffix}")

def hapke_albedo_correction(topo_mosaic, hapke_map):
    """
    Correct topographic mosaic using WAC Hapke-normalized albedo map
    
    Args:
        topo_mosaic: LROC topographic mosaic (with albedo variations)
        hapke_map: WAC Hapke-normalized albedo map (same resolution)
    
    Returns:
        Albedo-corrected mosaic
    """
    # Ensure same dimensions
    if topo_mosaic.shape != hapke_map.shape:
        print(f"Warning: Resizing Hapke map from {hapke_map.shape} to {topo_mosaic.shape}")
        hapke_map = cv2.resize(hapke_map, (topo_mosaic.shape[1], topo_mosaic.shape[0]))
    
    # Convert to float for processing
    topo_float = topo_mosaic.astype(np.float64)
    hapke_float = hapke_map.astype(np.float64)
    
    # Avoid division by zero
    hapke_float = np.where(hapke_float < 1e-6, 1e-6, hapke_float)
    
    # Divide to remove albedo component
    corrected = topo_float / hapke_float
    
    # Normalize to preserve overall brightness range
    corrected = corrected * np.mean(hapke_float)
    
    # Clip to reasonable range (adjust based on your data type)
    corrected = np.clip(corrected, 0, np.percentile(corrected, 99.9))
    
    return corrected

def local_highpass_filter_gpu(image, sigma):
    img_gpu = cp.asarray(image.astype(np.float32))
    low_freq = cp_gaussian_filter(img_gpu, sigma=sigma)
    img_gpu -= low_freq
    del low_freq
    cp.get_default_memory_pool().free_all_blocks()
    
    # Sample for percentile to avoid int32 overflow on large array
    flat = img_gpu.ravel()
    sample_idx = cp.random.choice(flat.size, size=10_000_000, replace=False)
    sample = flat[sample_idx]
    p_low = float(cp.percentile(sample, 0.1))
    p_high = float(cp.percentile(sample, 99.9))
    del sample, sample_idx, flat
    
    cp.clip(img_gpu, p_low, p_high, out=img_gpu)
    return cp.asnumpy(img_gpu)

def local_highpass_filter(image, sigma):
    img_float = image.astype(np.float32)
    low_freq = ndimage.gaussian_filter(img_float, sigma=sigma)
    img_float -= low_freq
    del low_freq
    # Symmetric clip to remove extreme outliers only
    p_low = np.percentile(img_float, 0.1)
    p_high = np.percentile(img_float, 99.9)
    np.clip(img_float, p_low, p_high, out=img_float)
    return img_float


def local_highpass_filter_multiscale(image, sigma):
    """Fast approximation using downsampling"""
    from scipy import ndimage
    import cv2
    
    print("  Downsampling for faster processing...")
    # Downsample by factor of 4
    downsampled = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    
    print(f"  Applying Gaussian blur on smaller image (sigma={sigma/4})...")
    low_freq_small = ndimage.gaussian_filter(downsampled.astype(np.float32), sigma=sigma/4)
    
    print("  Upsampling result...")
    low_freq = cv2.resize(low_freq_small, (image.shape[1], image.shape[0]), 
                          interpolation=cv2.INTER_CUBIC)
    
    print("  Computing high-pass...")
    img_float = image.astype(np.float32)
    high_pass = img_float - low_freq
    
    print("  Normalizing...")
    normalized = high_pass + np.median(img_float)
    normalized = np.clip(normalized, 0, np.percentile(normalized, 99.9))
    
    return normalized

def save_image(array, output_path, reference_path):
    with rasterio.open(reference_path) as src:
        profile = src.profile
    profile.update(dtype=rasterio.float32)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)
 
'''
def visualize_results(original, hapke_corrected, highpass_filtered, 
                     roi=None, save_path=None):
    """
    Visualize original and both normalized versions
    
    Args:
        original: Original mosaic
        hapke_corrected: Hapke albedo corrected version
        highpass_filtered: High-pass filtered version
        roi: Region of interest (x, y, width, height) for zoomed comparison
        save_path: Optional path to save figure
    """
    if roi is not None:
        x, y, w, h = roi
        original_roi = original[y:y+h, x:x+w]
        hapke_roi = hapke_corrected[y:y+h, x:x+w]
        highpass_roi = highpass_filtered[y:y+h, x:x+w]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Full images
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original LROC Mosaic', fontsize=14)
        axes[0, 0].axis('off')
        axes[0, 0].add_patch(plt.Rectangle((x, y), w, h, 
                                           edgecolor='red', facecolor='none', linewidth=2))
        
        axes[0, 1].imshow(hapke_corrected, cmap='gray')
        axes[0, 1].set_title('Hapke Albedo Corrected', fontsize=14)
        axes[0, 1].axis('off')
        axes[0, 1].add_patch(plt.Rectangle((x, y), w, h, 
                                           edgecolor='red', facecolor='none', linewidth=2))
        
        axes[0, 2].imshow(highpass_filtered, cmap='gray')
        axes[0, 2].set_title('High-Pass Filtered', fontsize=14)
        axes[0, 2].axis('off')
        axes[0, 2].add_patch(plt.Rectangle((x, y), w, h, 
                                           edgecolor='red', facecolor='none', linewidth=2))
        
        # ROI zoomed
        axes[1, 0].imshow(original_roi, cmap='gray')
        axes[1, 0].set_title('Original (ROI)', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(hapke_roi, cmap='gray')
        axes[1, 1].set_title('Hapke Corrected (ROI)', fontsize=14)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(highpass_roi, cmap='gray')
        axes[1, 2].set_title('High-Pass Filtered (ROI)', fontsize=14)
        axes[1, 2].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original LROC Mosaic', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(hapke_corrected, cmap='gray')
        axes[1].set_title('Hapke Albedo Corrected', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(highpass_filtered, cmap='gray')
        axes[2].set_title('High-Pass Filtered', fontsize=14)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
'''

def main():
    topo_mosaic_path = Path("data/raw/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif")
    output_dir = Path("data/raw/")
    sigma = 200

    print("Loading image...")
    topo_mosaic = load_image(topo_mosaic_path)
    print(f"Mosaic shape: {topo_mosaic.shape}, dtype: {topo_mosaic.dtype}")

    print(f"Applying high-pass filter (sigma={sigma})...")
    highpass_filtered = local_highpass_filter(topo_mosaic, sigma=sigma)

    output_dir.mkdir(exist_ok=True)

    print("Saving...")
    save_image(highpass_filtered, 
               output_dir / "LROC_highpass_filtered_200.tif", 
               reference_path=topo_mosaic_path)

    print("\n=== Statistics ===")
    print(f"Original  - Mean: {np.mean(topo_mosaic):.2f}, Std: {np.std(topo_mosaic):.2f}")
    print(f"Filtered  - Mean: {np.mean(highpass_filtered):.2f}, Std: {np.std(highpass_filtered):.2f}")

    print(f"\nDone! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()