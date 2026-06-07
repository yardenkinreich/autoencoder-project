import numpy as np
import matplotlib.pyplot as plt
from src.cluster.cluster import *
import os
from src.test.evaluate import *
from src.test.display_npy import *
from src.helper_functions import *
from src.data.preprocess import *
from src.data.full_mosaic_processing import *
import pandas as pd
import rasterio

# --- Main Execution ---
if __name__ == "__main__":
    input_path = 'data/raw/wac_mosaic_new_version/WAC_GLOBAL_100m_180.tif'
    output_path = 'data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif'
    SIGMA = 100

    print(f"Starting processing with Sigma={SIGMA}")
    print("Loading image with rasterio...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(input_path) as src:
        print(f"Image shape: {src.shape}, dtype: {src.dtypes[0]}")
        
        # Read the first band
        # WARNING: This loads ~6GB into RAM instantly. 
        # Conversion to float32 later will jump this to ~30GB+.
        image = src.read(1)
        
        # Copy metadata
        profile = src.profile.copy()

    print("Applying high-pass filter...")
    # Apply the filter
    highpass_filtered = local_highpass_filter_gpu(image, sigma=SIGMA)
    print(f"Highpass output range: min={highpass_filtered.min():.4f}, max={highpass_filtered.max():.4f}, mean={highpass_filtered.mean():.4f}")

    # Convert to uint8
    #output_uint8 = highpass_filtered.astype(np.f)

    # Update profile for saving
    profile.update(
        dtype=rasterio.float32,  # Save as float32 to preserve precision
        count=1, 
        compress='lzw',      # Saves disk space
        driver='GTiff',
        bigtiff='YES'        # MANDATORY: Required for files > 4GB
    )

    print(f"Saving to {output_path}...")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(highpass_filtered.astype(np.float32), 1)

    print("Processing complete.")