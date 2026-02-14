import numpy as np
import matplotlib.pyplot as plt
from src.cluster.cluster import *
import os
import pandas as pd
from src.test.evaluate import *
from src.test.display_npy import *
from src.helper_functions import *
from src.data.preprocess import *
from src.data.full_mosaic_processing import *
import pandas as pd
import rasterio
import rasterio

# --- Main Execution ---
if __name__ == "__main__":
    input_path = 'data/raw/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif'
    output_path = 'data/raw/highpass_filtered_lunar_mosaic_sigma_100.tif'
    SIGMA = 100

    print(f"Starting processing with Sigma={SIGMA}")
    print("Loading image with rasterio...")
    
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
    highpass_filtered = local_highpass_filter(image, sigma=SIGMA)

    # Convert to uint8
    output_uint8 = highpass_filtered.astype(np.uint8)

    # Update profile for saving
    profile.update(
        dtype=rasterio.uint8, 
        count=1, 
        compress='lzw',      # Saves disk space
        driver='GTiff',
        bigtiff='YES'        # MANDATORY: Required for files > 4GB
    )

    print(f"Saving to {output_path}...")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output_uint8, 1)

    print("Processing complete.")