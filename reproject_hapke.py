import numpy as np
import rasterio
from rasterio.transform import from_origin

def convert_0_360_to_180_180(input_path, output_path):
    with rasterio.open(input_path) as src:
        # Read the image (Channels, Height, Width)
        data = src.read()
        profile = src.profile.copy()
        
        # Calculate the shift (Half the width)
        # We assume the image covers the full 360 degrees width
        shift_amount = data.shape[2] // 2
        
        print(f"Rolling image by {shift_amount} pixels...")
        
        # ROLL THE ARRAY
        # This moves the right half (180-360) to the left side (-180-0)
        # and pushes the left half (0-180) to the right side.
        new_data = np.roll(data, shift=shift_amount, axis=2)
        
        # FIX THE COORDINATES (GEO-TRANSFORM)
        # The old map started at 0. The new map starts at -180.
        # We assume the pixel resolution (transform[0]) stays the same.
        current_transform = src.transform
        new_transform = from_origin(
            west=-180.0,              # New left edge
            north=current_transform.f, # Top edge stays same (usually 90)
            xsize=current_transform.a, # Pixel width stays same
            ysize=-current_transform.e # Pixel height (make sure it's positive for from_origin)
        )
        
        profile.update({
            'transform': new_transform,
            'height': data.shape[1],
            'width': data.shape[2],
            'crs': 'EPSG:4326',  # Standard WGS84 or Moon equivalent
            'BIGTIFF': 'YES'
        })
        
        # Save
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(new_data)
            
    print(f"Saved corrected map to {output_path}")

# Usage
# convert_0_360_to_180_180('upper_layer_0_360.tif', 'fixed_layer.tif')