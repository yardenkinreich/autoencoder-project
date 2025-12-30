import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def display_npy_image(npy_path, denormalize=True, stats_path=None):
    """
    Display a single .npy image file
    
    Args:
        npy_path: Path to the .npy file
        denormalize: If True, denormalize using stats if available
        stats_path: Path to the _stats.npz file (optional)
    """
    # Load the image
    img = np.load(npy_path)
    
    print(f"File: {os.path.basename(npy_path)}")
    print(f"Shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Value range: [{img.min():.4f}, {img.max():.4f}]")
    print(f"Mean: {img.mean():.4f}, Std: {img.std():.4f}")
    
    # Denormalize if requested and stats available
    display_img = img.copy()
    if denormalize and stats_path and os.path.exists(stats_path):
        stats = np.load(stats_path)
        mean = stats['mean']
        std = stats['std']
        print(f"Denormalizing with mean={mean}, std={std}")
        
        # Denormalize: x_original = x_normalized * std + mean
        for c in range(img.shape[0]):
            display_img[c] = img[c] * std[c] + mean[c]
        
        print(f"Denormalized range: [{display_img.min():.4f}, {display_img.max():.4f}]")
    
    # Display the image
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normalized version
    if img.shape[0] == 1:  # Grayscale
        axes[0].imshow(img[0], cmap='gray')
        axes[0].set_title('Normalized (Grayscale)')
    else:  # RGB
        # Transpose from (C, H, W) to (H, W, C) for display
        img_display = np.transpose(img, (1, 2, 0))
        # Clip to reasonable range for display
        img_display = np.clip(img_display, -3, 3)
        # Scale to [0, 1] for display
        img_display = (img_display + 3) / 6
        axes[0].imshow(img_display)
        axes[0].set_title('Normalized (RGB, scaled)')
    axes[0].axis('off')
    
    # Denormalized version
    if display_img.shape[0] == 1:  # Grayscale
        axes[1].imshow(display_img[0], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Denormalized (Grayscale)')
    else:  # RGB
        display_rgb = np.transpose(display_img, (1, 2, 0))
        display_rgb = np.clip(display_rgb, 0, 1)
        axes[1].imshow(display_rgb)
        axes[1].set_title('Denormalized (RGB)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_npy_grid(npy_dir, num_images=16, denormalize=True, stats_path=None):
    """
    Display a grid of .npy images from a directory
    
    Args:
        npy_dir: Directory containing .npy files
        num_images: Number of images to display in grid
        denormalize: If True, denormalize using stats if available
        stats_path: Path to the _stats.npz file (optional)
    """
    # Get all .npy files
    npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    
    if len(npy_files) == 0:
        print(f"No .npy files found in {npy_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    # Select subset
    selected_files = npy_files[:num_images]
    
    # Load stats if available
    mean, std = None, None
    if denormalize and stats_path and os.path.exists(stats_path):
        stats = np.load(stats_path)
        mean = stats['mean']
        std = stats['std']
        print(f"Using stats: mean={mean}, std={std}")
    
    # Determine grid size
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, npy_path in enumerate(selected_files):
        img = np.load(npy_path)
        
        # Denormalize if requested
        if denormalize and mean is not None and std is not None:
            for c in range(img.shape[0]):
                img[c] = img[c] * std[c] + mean[c]
        
        # Display
        if img.shape[0] == 1:  # Grayscale
            axes[idx].imshow(img[0], cmap='gray', vmin=0, vmax=1)
        else:  # RGB
            img_display = np.transpose(img, (1, 2, 0))
            img_display = np.clip(img_display, 0, 1)
            axes[idx].imshow(img_display)
        
        axes[idx].set_title(os.path.basename(npy_path)[:15], fontsize=8)
        axes[idx].axis('off')
    
    # Hide remaining subplots
    for idx in range(len(selected_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_memmap_samples(memmap_path, num_samples=16, shape_per_image=(3, 224, 224)):
    """
    Display samples from a memmap file
    
    Args:
        memmap_path: Path to the .dat memmap file
        num_samples: Number of samples to display
        shape_per_image: Shape of each image (C, H, W)
    """
    # Try to infer total number of images
    file_size = os.path.getsize(memmap_path)
    bytes_per_image = np.prod(shape_per_image) * 4  # float32 = 4 bytes
    total_images = file_size // bytes_per_image
    
    print(f"Memmap file: {memmap_path}")
    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Estimated total images: {total_images}")
    print(f"Shape per image: {shape_per_image}")
    
    # Load memmap
    data = np.memmap(memmap_path, dtype=np.float32, mode='r', 
                     shape=(total_images, *shape_per_image))
    
    # Sample indices
    indices = np.linspace(0, total_images - 1, num_samples, dtype=int)
    
    # Determine grid size
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for plot_idx, img_idx in enumerate(indices):
        img = data[img_idx]
        
        # Display
        if img.shape[0] == 1:  # Grayscale
            axes[plot_idx].imshow(img[0], cmap='gray')
        else:  # RGB - scale for display
            img_display = np.transpose(img, (1, 2, 0))
            # Clip normalized values
            img_display = np.clip(img_display, -3, 3)
            # Scale to [0, 1]
            img_display = (img_display + 3) / 6
            axes[plot_idx].imshow(img_display)
        
        axes[plot_idx].set_title(f"Index {img_idx}", fontsize=8)
        axes[plot_idx].axis('off')
    
    # Hide remaining subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# ==============================================================================
# JUPYTER NOTEBOOK TESTING FUNCTIONS
# ==============================================================================

def quick_test_preprocessing(npy_dir, stats_path=None, num_samples=9):
    """
    Quick test function for Jupyter notebooks to verify preprocessing
    
    Args:
        npy_dir: Directory containing .npy files
        stats_path: Path to _stats.npz file (optional)
        num_samples: Number of samples to display
    
    Example usage in Jupyter:
        quick_test_preprocessing('mae_crops/', 'mae_craters_stats.npz', num_samples=9)
    """
    print("="*60)
    print("PREPROCESSING VERIFICATION TEST")
    print("="*60)
    
    # Get files
    npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    if len(npy_files) == 0:
        print(f"❌ No .npy files found in {npy_dir}")
        return
    
    print(f"✓ Found {len(npy_files)} .npy files")
    
    # Load stats if available
    if stats_path and os.path.exists(stats_path):
        stats = np.load(stats_path)
        mean = stats['mean']
        std = stats['std']
        print(f"✓ Loaded stats: mean={mean}, std={std}")
    else:
        mean, std = None, None
        print("⚠ No stats file provided")
    
    # Sample random files
    sample_files = np.random.choice(npy_files, min(num_samples, len(npy_files)), replace=False)
    
    # Load first file to get info
    first_img = np.load(sample_files[0])
    print(f"\n--- Image Properties ---")
    print(f"Shape: {first_img.shape}")
    print(f"Dtype: {first_img.dtype}")
    print(f"Channels: {first_img.shape[0]} ({'RGB' if first_img.shape[0] == 3 else 'Grayscale'})")
    print(f"Size: {first_img.shape[1]}x{first_img.shape[2]}")
    
    # Collect statistics
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    
    for f in sample_files[:100]:  # Check up to 100 files
        img = np.load(f)
        all_means.append(img.mean())
        all_stds.append(img.std())
        all_mins.append(img.min())
        all_maxs.append(img.max())
    
    print(f"\n--- Normalized Data Statistics (from {len(all_means)} files) ---")
    print(f"Mean range: [{np.min(all_means):.4f}, {np.max(all_means):.4f}]")
    print(f"Mean of means: {np.mean(all_means):.4f} (should be ~0)")
    print(f"Std range: [{np.min(all_stds):.4f}, {np.max(all_stds):.4f}]")
    print(f"Mean of stds: {np.mean(all_stds):.4f} (should be ~1)")
    print(f"Global min: {np.min(all_mins):.4f}")
    print(f"Global max: {np.max(all_maxs):.4f}")
    
    # Visual check
    print(f"\n--- Displaying {num_samples} Random Samples ---")
    
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, npy_path in enumerate(sample_files):
        img = np.load(npy_path)
        
        # Denormalize if possible
        if mean is not None and std is not None:
            img_display = img.copy()
            for c in range(img.shape[0]):
                img_display[c] = img[c] * std[c] + mean[c]
        else:
            img_display = img
        
        # Display
        if img_display.shape[0] == 1:  # Grayscale
            axes[idx].imshow(img_display[0], cmap='gray', vmin=0, vmax=1)
        else:  # RGB
            img_rgb = np.transpose(img_display, (1, 2, 0))
            img_rgb = np.clip(img_rgb, 0, 1)
            axes[idx].imshow(img_rgb)
        
        crater_id = os.path.basename(npy_path).replace('.npy', '')
        axes[idx].set_title(f"{crater_id[:10]}", fontsize=8)
        axes[idx].axis('off')
    
    # Hide remaining subplots
    for idx in range(len(sample_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING TEST COMPLETE")
    print("="*60)


def quick_test_memmap(memmap_path, stats_path=None, num_samples=9, num_channels=3, image_size=224):
    """
    Quick test function for memmap files in Jupyter notebooks
    
    Example usage:
        quick_test_memmap('mae_craters.dat', 'mae_craters_stats.npz', num_samples=9, num_channels=3)
    """
    print("="*60)
    print("MEMMAP PREPROCESSING VERIFICATION TEST")
    print("="*60)
    
    # Check file
    if not os.path.exists(memmap_path):
        print(f"❌ File not found: {memmap_path}")
        return
    
    file_size = os.path.getsize(memmap_path)
    bytes_per_image = num_channels * image_size * image_size * 4  # float32
    total_images = file_size // bytes_per_image
    
    print(f"✓ File: {memmap_path}")
    print(f"✓ Size: {file_size / (1024**3):.2f} GB")
    print(f"✓ Total images: {total_images}")
    print(f"✓ Shape per image: ({num_channels}, {image_size}, {image_size})")
    
    # Load stats
    if stats_path and os.path.exists(stats_path):
        stats = np.load(stats_path)
        mean = stats['mean']
        std = stats['std']
        print(f"✓ Loaded stats: mean={mean}, std={std}")
    else:
        mean, std = None, None
        print("⚠ No stats file")
    
    # Load memmap
    data = np.memmap(memmap_path, dtype=np.float32, mode='r',
                     shape=(total_images, num_channels, image_size, image_size))
    
    # Sample indices
    sample_indices = np.random.choice(total_images, min(num_samples, total_images), replace=False)
    
    # Statistics
    print(f"\n--- Normalized Data Statistics (from {min(1000, total_images)} samples) ---")
    sample_data = data[:min(1000, total_images)]
    print(f"Mean: {sample_data.mean():.4f} (should be ~0)")
    print(f"Std: {sample_data.std():.4f} (should be ~1)")
    print(f"Min: {sample_data.min():.4f}")
    print(f"Max: {sample_data.max():.4f}")
    print(f"Per-channel means: {sample_data.mean(axis=(0,2,3))}")
    print(f"Per-channel stds: {sample_data.std(axis=(0,2,3))}")
    
    # Display
    print(f"\n--- Displaying {len(sample_indices)} Random Samples ---")
    
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for plot_idx, img_idx in enumerate(sample_indices):
        img = data[img_idx]
        
        # Denormalize if possible
        if mean is not None and std is not None:
            img_display = img.copy()
            for c in range(num_channels):
                img_display[c] = img[c] * std[c] + mean[c]
        else:
            img_display = img
            # Scale for display if normalized
            img_display = np.clip(img_display, -3, 3)
            img_display = (img_display + 3) / 6
        
        # Display
        if num_channels == 1:  # Grayscale
            axes[plot_idx].imshow(img_display[0], cmap='gray', vmin=0, vmax=1)
        else:  # RGB
            img_rgb = np.transpose(img_display, (1, 2, 0))
            img_rgb = np.clip(img_rgb, 0, 1)
            axes[plot_idx].imshow(img_rgb)
        
        axes[plot_idx].set_title(f"Index {img_idx}", fontsize=8)
        axes[plot_idx].axis('off')
    
    # Hide remaining
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("✅ MEMMAP TEST COMPLETE")
    print("="*60)


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Display .npy crater images')
    parser.add_argument('--mode', choices=['single', 'grid', 'memmap'], required=True,
                        help='Display mode: single file, grid of files, or memmap samples')
    parser.add_argument('--npy_path', help='Path to single .npy file (for single mode)')
    parser.add_argument('--npy_dir', help='Directory containing .npy files (for grid mode)')
    parser.add_argument('--memmap_path', help='Path to .dat memmap file (for memmap mode)')
    parser.add_argument('--stats_path', help='Path to _stats.npz file for denormalization')
    parser.add_argument('--num_images', type=int, default=16, 
                        help='Number of images to display in grid/memmap mode')
    parser.add_argument('--no_denormalize', action='store_true',
                        help='Do not denormalize images')
    parser.add_argument('--num_channels', type=int, default=3, choices=[1, 3],
                        help='Number of channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size (assumes square images)')
    
    args = parser.parse_args()
    
    denormalize = not args.no_denormalize
    
    if args.mode == 'single':
        if not args.npy_path:
            print("Error: --npy_path required for single mode")
        else:
            display_npy_image(args.npy_path, denormalize, args.stats_path)
    
    elif args.mode == 'grid':
        if not args.npy_dir:
            print("Error: --npy_dir required for grid mode")
        else:
            display_npy_grid(args.npy_dir, args.num_images, denormalize, args.stats_path)
    
    elif args.mode == 'memmap':
        if not args.memmap_path:
            print("Error: --memmap_path required for memmap mode")
        else:
            shape = (args.num_channels, args.image_size, args.image_size)
            display_memmap_samples(args.memmap_path, args.num_images, shape)