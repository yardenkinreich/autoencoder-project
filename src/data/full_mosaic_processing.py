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

# ── the clip these two filters used to apply unconditionally ──────────────────
# DO NOT re-enable this without reading this comment.
#
# Both filters below used to end with an unconditional clip to the GLOBAL
# 0.1/99.9 percentiles of the high-pass residual, commented "remove extreme
# outliers only". On a whole-Moon mosaic it does not do that. The window is
# fitted over ~4e9 pixels that are overwhelmingly smooth mare and highland,
# while craters are the high-contrast minority - so crater rims and shadows sit
# far outside a bound set by flat terrain, and get flattened onto it.
#
# Measured on the 642 labeled craters, fraction of pixels pinned to exactly one
# of the two clip bounds:
#
#     product                         overall   v-fr    r-fr    r-dr    v-dr
#     WAC_GLOBAL morphology sigma-100   0.99%   6.04%   2.24%   0.16%   0.00%
#     WAC_EMP_643           sigma-100  11.87%     (flat across classes, but
#                                                  only because 12% of every
#                                                  crop is already destroyed)
#     WAC_GLOBAL morphology raw        0.0003%  (unfiltered - clean)
#
# The morphology numbers are monotonic in degradation and EXACTLY ZERO for the
# most degraded class: the clip removes signal in direct proportion to how
# fresh a crater is, which is the one thing this project is trying to measure.
# EMP is far worse because its photometric normalization narrows the global
# residual distribution, pulling the 99.9th percentile down into the range
# crater interiors occupy.
#
# This compounds with the SEPARATE [0,1] percentile clip in the normalization
# step (see src/data/norm_stats.py): two independent clips, both preferentially
# destroying fresh craters.
#
# A high-pass residual should be stored unclipped. Outlier handling belongs at
# normalization time, where it is a recorded, reversible policy choice - not
# baked irreversibly into a 17 GB derived product.
_LEGACY_CLIP_PCT = (0.1, 99.9)


def local_highpass_filter_gpu(image, sigma, clip_pct=_LEGACY_CLIP_PCT):
    """clip_pct=None (recommended) stores the residual unclipped.
    The (0.1, 99.9) default is retained ONLY so the historical products in
    data/raw/**/sigma/ remain reproducible; see _LEGACY_CLIP_PCT above."""
    img_gpu = cp.asarray(image.astype(np.float32))
    low_freq = cp_gaussian_filter(img_gpu, sigma=sigma)
    img_gpu -= low_freq
    del low_freq
    cp.get_default_memory_pool().free_all_blocks()

    if clip_pct is not None:
        # Sample for percentile to avoid int32 overflow on large array
        flat = img_gpu.ravel()
        sample_idx = cp.random.choice(flat.size, size=10_000_000, replace=False)
        sample = flat[sample_idx]
        p_low = float(cp.percentile(sample, clip_pct[0]))
        p_high = float(cp.percentile(sample, clip_pct[1]))
        del sample, sample_idx, flat
        print(f"  WARNING: clipping residual to global {clip_pct} percentiles "
              f"[{p_low:.6g}, {p_high:.6g}] - this destroys fresh-crater pixels "
              f"in proportion to freshness; see _LEGACY_CLIP_PCT.")
        cp.clip(img_gpu, p_low, p_high, out=img_gpu)
    return cp.asnumpy(img_gpu)


def divisive_normalize_gpu(resid, sigma, floor_pct=1.0):
    """Divide a high-pass residual by its own LOCAL CONTRAST level.

        gain = sqrt(blur_sigma(resid**2))
        out  = resid / max(gain, floor)

    WHY THIS EXISTS, AND WHY THE HIGH-PASS ALONE CANNOT DO IT
    --------------------------------------------------------
    The mosaic's tile seams are a CONTRAST (gain) difference: adjacent source
    tiles were imaged at different incidence angles, so the local dynamic
    range is scaled. `local_highpass_filter` SUBTRACTS a blurred copy, which
    removes an additive offset (regional albedo - maria vs highland) and
    leaves a multiplicative gain completely untouched. Scaling a tile by 1.2x
    scales its residual by 1.2x too. That is exactly why the seam survived
    the sigma-100 filter unchanged, and why the correction has to be
    DIVISIVE rather than subtractive.

    WHY resid**2 AND NOT resid
    --------------------------
    The blur has to act on a NON-NEGATIVE quantity. `resid` is locally
    zero-mean by construction, so blur(resid) ~ 0 and carries no information
    about contrast. resid**2 (equivalently |grad I|) is non-negative, so its
    local average is a meaningful local-RMS contrast estimate. This is also
    why the gain field cannot be approximated from a decimated read of the
    residual - block-averaging drives it to zero.

    CHOOSING sigma
    --------------
    It must sit in the gap between crater scale and tile scale:
      craters    1-12 km  ->   10-120 px at 100 m/px
      tiles      90 deg   ->   ~2700 km at the equator
    sigma = 500 px (50 km) is the default. At that width a 12 km crater is
    ~0.7% of the kernel weight, so a crater cannot meaningfully normalize
    itself away; at 200 px it rises to ~4.5%, which starts to matter for the
    largest craters. Measured independently on 100k craters, a scattered
    estimate of the same field found its optimum at a 100 km radius, which
    corresponds to a Gaussian of sigma ~50 km - the same answer.

    THE FLOOR
    ---------
    Where the terrain is genuinely smooth, `gain` approaches zero and the
    division would amplify sensor noise without limit. `floor_pct` clamps the
    divisor at that percentile of the gain field, so the flattest regions are
    divided by a constant instead of being blown up. This is the one place
    the operation is not reversible."""
    img = cp.asarray(resid, dtype=cp.float32)
    sq = img * img
    cp_gaussian_filter(sq, sigma=sigma, output=sq)
    cp.sqrt(sq, out=sq)                      # sq now holds the gain field

    flat = sq.ravel()
    idx = cp.random.choice(flat.size, size=min(10_000_000, flat.size), replace=False)
    floor = float(cp.percentile(flat[idx], floor_pct))
    del idx, flat
    print(f"  divisive: sigma={sigma}, gain floor (p{floor_pct}) = {floor:.6g}")
    cp.maximum(sq, floor, out=sq)
    img /= sq
    del sq
    cp.get_default_memory_pool().free_all_blocks()
    return cp.asnumpy(img)


def local_highpass_filter(image, sigma, clip_pct=_LEGACY_CLIP_PCT):
    """CPU counterpart of local_highpass_filter_gpu; same clip_pct semantics."""
    img_float = image.astype(np.float32)
    low_freq = ndimage.gaussian_filter(img_float, sigma=sigma)
    img_float -= low_freq
    del low_freq
    if clip_pct is not None:
        p_low = np.percentile(img_float, clip_pct[0])
        p_high = np.percentile(img_float, clip_pct[1])
        print(f"  WARNING: clipping residual to global {clip_pct} percentiles "
              f"[{p_low:.6g}, {p_high:.6g}] - see _LEGACY_CLIP_PCT.")
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