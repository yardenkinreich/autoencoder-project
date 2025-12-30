import os
import numpy as np
import pandas as pd
import pyproj
import rasterio
import matplotlib.pyplot as plt
from src.helper_functions import *
import cv2
from transformers import AutoImageProcessor
import torchvision.transforms as transforms
from PIL import Image


def load_and_filter_craters(craters_csv, min_diameter, max_diameter, latitude_bounds, craters_to_output):
    craters = pd.read_csv(craters_csv)
    filtered = craters[
        (craters['DIAM_CIRC_IMG'] >= min_diameter) &
        (craters['DIAM_CIRC_IMG'] <= max_diameter) &
        (craters['LAT_CIRC_IMG'] >= latitude_bounds[0]) &
        (craters['LAT_CIRC_IMG'] <= latitude_bounds[1])
    ]
    if craters_to_output > 0:
        filtered = filtered.sample(n=craters_to_output, random_state=42)
    return filtered


def get_craters_crs():
    craters_wkt = """
        GEOGCS["GCS_Moon",
        DATUM["D_Moon_2000",
        SPHEROID["Moon_2000_IAU_IAG",1737151.3,0, LENGTHUNIT["metre",1]]],
        PRIMEM["Reference_Meridian",0],
        UNIT["metre",1]],
        PROJECTION["Equirectangular"],
        PARAMETER["standard_parallel_1",0],
        PARAMETER["central_meridian",0],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["metre",1, AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["ESRI","103881"]]
    """
    return pyproj.CRS.from_wkt(craters_wkt)


def process_and_save_crater_crops(
    filtered_craters,
    map_file,
    output_dir,
    offset,
    save_raw_crops=True,
    save_np_array=True,
    output_path=None,
    target_size=224,  # Final size for both CAE and MAE
    num_channels=3    # 3 for RGB (MAE), 1 for grayscale (CAE)
):
    """
    Unified preprocessing pipeline for both CAE and MAE:
    1. Crop crater from map with offset
    2. Resize to 256x256
    3. Center crop to 224x224
    4. Convert to target channels (1 for CAE, 3 for MAE)
    5. Compute dataset statistics (mean, std)
    6. Normalize using dataset statistics
    7. Save individual crops and numpy array
    """
    os.makedirs(output_dir, exist_ok=True)
    
    craters_crs = get_craters_crs()
    N = len(filtered_craters)
    
    # Transform pipeline (no normalization yet)
    transform_no_norm = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(target_size),
        transforms.ToTensor()  # Converts to [0, 1]
    ])
    
    # Create temporary memmap for storing unnormalized data
    temp_memmap = None
    if save_np_array and output_path is not None:
        temp_shape = (N, num_channels, target_size, target_size)
        temp_memmap = np.memmap(
            output_path + ".temp", dtype=np.float32, mode="w+", shape=temp_shape
        )
    
    print(f"=== PASS 1: Loading and cropping {N} craters ===")
    print(f"Target size: {target_size}x{target_size}, Channels: {num_channels}")
    
    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )
        
        for i, (_, crater) in enumerate(filtered_craters.iterrows()):
            if i % 10000 == 0:
                print(f"Processed {i}/{N}")
            
            # Crop crater from map
            crater_img = crop_crater(
                map_ref,
                crater["LAT_CIRC_IMG"],
                crater["LON_CIRC_IMG"],
                crater["DIAM_CIRC_IMG"],
                offset,
                transformer
            )
            
            # Ensure proper dimensions
            if crater_img.ndim == 2:
                crater_img = np.stack([crater_img] * 3, axis=-1)  # Convert grayscale to RGB
            
            # Convert to PIL and apply transforms
            pil_img = Image.fromarray(crater_img.astype(np.uint8), mode="RGB")
            tensor_img = transform_no_norm(pil_img)  # Shape: (3, 224, 224), range [0, 1]
            
            # Convert to target number of channels
            if num_channels == 1:
                # Convert RGB to grayscale for CAE
                tensor_img = tensor_img.mean(dim=0, keepdim=True)  # Shape: (1, 224, 224)
            
            # Store in temporary memmap
            if temp_memmap is not None:
                temp_memmap[i] = tensor_img.numpy()
    
    if temp_memmap is None:
        print("Error: temp_memmap is None")
        return
    
    temp_memmap.flush()
    
    # === Step 2: Compute dataset statistics ===
    print("\n=== PASS 2: Computing dataset statistics ===")
    
    chunk_size = 1000
    
    # Compute mean
    print("Computing mean...")
    sum_pixels = np.zeros(num_channels, dtype=np.float64)
    total_pixels = 0
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = temp_memmap[start_idx:end_idx]
        sum_pixels += chunk.sum(axis=(0, 2, 3))  # Sum per channel
        total_pixels += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
    
    dataset_mean = sum_pixels / total_pixels
    
    # Compute std
    print("Computing std...")
    sum_squared_diff = np.zeros(num_channels, dtype=np.float64)
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = temp_memmap[start_idx:end_idx]
        
        for c in range(num_channels):
            diff = chunk[:, c, :, :] - dataset_mean[c]
            sum_squared_diff[c] += (diff ** 2).sum()
    
    dataset_std = np.sqrt(sum_squared_diff / total_pixels)
    
    print(f"\n=== DATASET STATISTICS ===")
    print(f"Mean per channel: {dataset_mean}")
    print(f"Std per channel: {dataset_std}")
    print(f"Total pixels: {total_pixels:,}")
    
    # Sanity checks
    if np.any(dataset_std < 0.01):
        print("⚠️  WARNING: Very low std detected.")
    if np.any(dataset_std > 0.5):
        print("⚠️  WARNING: Very high std detected.")
    
    # === Step 3: Apply normalization and save ===
    print("\n=== PASS 3: Normalizing and saving ===")
    
    # Create final memmap for normalized data
    crater_memmap = np.memmap(
        output_path, dtype=np.float32, mode="w+", 
        shape=(N, num_channels, target_size, target_size)
    )
    
    for start_idx in range(0, N, chunk_size):
        if start_idx % 10000 == 0:
            print(f"Normalizing {start_idx}/{N}")
        
        end_idx = min(start_idx + chunk_size, N)
        chunk = temp_memmap[start_idx:end_idx].copy()
        
        # Debug: print before normalization
        if start_idx == 0:
            print(f"  Before normalization - chunk mean: {chunk.mean():.6f}, std: {chunk.std():.6f}")
        
        # Apply normalization: (x - mean) / std
        '''
        for c in range(num_channels):
            chunk[:, c, :, :] = (chunk[:, c, :, :] - dataset_mean[c]) / (dataset_std[c] + 1e-8)
            '''
        
        # Debug: print after normalization
        if start_idx == 0:
            print(f"  After normalization - chunk mean: {chunk.mean():.6f}, std: {chunk.std():.6f}")
        
        # Write normalized data to memmap
        crater_memmap[start_idx:end_idx] = chunk
        
        # Save individual normalized crops if requested
        if save_raw_crops:
            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                crater_id = filtered_craters.iloc[global_idx]["CRATER_ID"]
                crop_data = chunk[local_idx]  # Shape: (C, H, W), already normalized
                np.save(os.path.join(output_dir, f"{crater_id}.npy"), crop_data)
    
    crater_memmap.flush()
    
    # Clean up temp file
    del temp_memmap
    os.remove(output_path + ".temp")
    
    print(f"\n✅ Finished processing {N} craters. Data saved to {output_path}")
    
    # === Validation - reload from disk to verify ===
    print("\n=== VALIDATION: Normalized dataset statistics ===")
    
    # IMPORTANT: Reload the memmap from disk to ensure we're reading what was actually saved
    del crater_memmap  # Close the write memmap
    
    # Open read-only for validation
    validation_memmap = np.memmap(
        output_path, dtype=np.float32, mode='r',
        shape=(N, num_channels, target_size, target_size)
    )
    
    sample_size = min(1000, N)
    sample_data = validation_memmap[:sample_size]
    print(f"Sample size: {sample_size}")
    print(f"Range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
    print(f"Mean: {sample_data.mean():.4f} (should be ~0)")
    print(f"Std: {sample_data.std():.4f} (should be ~1)")
    print(f"Per-channel means: {sample_data.mean(axis=(0,2,3))} (should be ~0)")
    print(f"Per-channel stds: {sample_data.std(axis=(0,2,3))} (should be ~1)")
    
    # Outlier check
    print(f"\n=== OUTLIER CHECK ===")
    print(f"Values < -3: {(sample_data < -3).sum()} pixels")
    print(f"Values > 3: {(sample_data > 3).sum()} pixels")
    print(f"99th percentile: {np.percentile(sample_data, 99):.4f}")
    print(f"1st percentile: {np.percentile(sample_data, 1):.4f}")
    
    # Save statistics
    stats_path = output_path.replace('.dat', '_stats.npz')
    np.savez(stats_path, mean=dataset_mean, std=dataset_std)
    print(f"\nSaved normalization statistics to {stats_path}")


def save_crater_metadata(filtered_craters, map_file, output_path):
    craters_crs = get_craters_crs()
    buffer = []
    batch_size = 10000

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(craters_crs, map_ref.crs.to_string(), always_xy=True)

        for i, crater in filtered_craters.iterrows():
            lon = crater['LON_CIRC_IMG']
            lat = crater['LAT_CIRC_IMG']
            diam = crater['DIAM_CIRC_IMG']
            crater_id = crater['CRATER_ID']
            if lon > 180:
                lon -= 360
            x, y = transformer.transform(lon, lat)
            diam *= 1000  # convert to meters

            buffer.append({'id': crater_id, 'lat': lat, 'lon': lon, 'x': x, 'y': y, 'diam': diam})

            if (i + 1) % batch_size == 0:
                print(f"Processed {i + 1} craters")
                df = pd.DataFrame(buffer)
                if os.path.exists(output_path):
                    df.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(output_path, index=False)
                buffer.clear()

        # Write any remaining craters
        if buffer:
            df = pd.DataFrame(buffer)
            if os.path.exists(output_path):
                df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                df.to_csv(output_path, index=False)


# Example usage:
# For MAE (RGB):
# process_and_save_crater_crops_unified(
#     filtered_craters, map_file, output_dir, offset,
#     save_raw_crops=True, save_np_array=True, output_path="mae_craters.dat",
#     target_size=224, num_channels=3
# )

# For CAE (Grayscale):
# process_and_save_crater_crops_unified(
#     filtered_craters, map_file, output_dir, offset,
#     save_raw_crops=True, save_np_array=True, output_path="cae_craters.dat",
#     target_size=224, num_channels=1
# )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', required=True)
    parser.add_argument('--craters_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--np_output_path', required=True)
    parser.add_argument('--info_output_path', required=True)
    parser.add_argument('--min_diameter', type=float, default=1.0)
    parser.add_argument('--max_diameter', type=float, default=10.0)
    parser.add_argument('--latitude_bounds', type=float, nargs=2, default=[-60, 60])
    parser.add_argument('--offset', type=float, default=0.5)
    parser.add_argument('--craters_to_output', type=int, default=-1)
    parser.add_argument('--save_raw_crops', action='store_true')
    parser.add_argument('--save_np_array', action='store_true')
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--autoencoder_model', type=str, choices=['cae', 'mae'], default='cae')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filtered = load_and_filter_craters(
        args.craters_csv,
        args.min_diameter,
        args.max_diameter,
        args.latitude_bounds,
        args.craters_to_output
    )

    print(f"Filtered {len(filtered)} craters")

    if args.autoencoder_model == 'cae':
        process_and_save_crater_crops(
            filtered,
            args.map_file,
            args.output_dir,
            args.offset,
            save_raw_crops=args.save_raw_crops,
            save_np_array=args.save_np_array,
            output_path=args.np_output_path,
            target_size=args.target_size,
            num_channels=1
        )
            # Save crater metadata
        save_crater_metadata(filtered, args.map_file, args.info_output_path)
    
        print("Processing complete!")

    elif args.autoencoder_model == 'mae':   
        process_and_save_crater_crops(
            filtered,
            args.map_file,
            args.output_dir,
            args.offset,
            save_raw_crops=args.save_raw_crops,
            save_np_array=args.save_np_array,
            output_path=args.np_output_path,
            target_size=args.target_size,
            num_channels=3
        )
        
        save_crater_metadata(filtered, args.map_file, args.info_output_path)
        print("Processing complete!")
