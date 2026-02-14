import os
import numpy as np
import pandas as pd
import pyproj
import rasterio
import matplotlib.pyplot as plt
from src.helper_functions import *
import cv2
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
    target_size=224,
    num_channels=3
):
    """
    Simplified preprocessing pipeline WITHOUT normalization:
    1. Crop crater from map with offset
    2. Flip crater (shadow to right) - done inside crop_crater()
    3. Resize directly to 224x224 (no center crop needed)
    4. Convert to target channels (1 for CAE, 3 for MAE)
    5. Compute and save statistics (NO normalization applied)
    6. Save individual crops and numpy array
    """
    os.makedirs(output_dir, exist_ok=True)
    
    craters_crs = get_craters_crs()
    N = len(filtered_craters)
    
    # Simplified transform: just resize to target size
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=Image.BICUBIC),
        transforms.ToTensor()  # Converts to [0, 1], shape (C, H, W)
    ])
    
    # Create memmap for storing data
    crater_memmap = None
    if save_np_array and output_path is not None:
        shape = (N, num_channels, target_size, target_size)
        crater_memmap = np.memmap(
            output_path, dtype=np.float32, mode="w+", shape=shape
        )
    
    print(f"=== Processing {N} craters (NO normalization) ===")
    print(f"Target size: {target_size}x{target_size}, Channels: {num_channels}")
    print(f"Pipeline: Crop → Flip (inside crop_crater) → Resize to {target_size}")
    
    # Track flipping statistics
    flip_check_count = 0
    flipped_correctly = 0
    
    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )
        
        for i, (_, crater) in enumerate(filtered_craters.iterrows()):
            if i % 10000 == 0:
                print(f"Processed {i}/{N}")
            
            # Crop crater from map
            # NOTE: crop_crater() already calls flip_crater() internally!
            crater_img = crop_crater(
                map_ref,
                crater["LAT_CIRC_IMG"],
                crater["LON_CIRC_IMG"],
                crater["DIAM_CIRC_IMG"],
                offset,
                transformer
            )
            
            # === FLIP VERIFICATION (first 100 craters) ===
            # Check AFTER crop_crater (which includes flip)
            if i < 100:
                qtr = crater_img.shape[1] // 4
                half = crater_img.shape[1] // 2
                left_side = crater_img[:, qtr:half]
                right_side = crater_img[:, half:-qtr]
                left_mean = left_side.mean()
                right_mean = right_side.mean()
                
                flip_check_count += 1
                if right_mean < left_mean:  # Shadow should be on right (darker)
                    flipped_correctly += 1
                
                if i < 10:  # Print details for first 10
                    print(f"  Crater {i} (after flip): Left={left_mean:.2f}, Right={right_mean:.2f}, "
                          f"Shadow on {'RIGHT ✓' if right_mean < left_mean else 'LEFT ✗'}")
            
            # Ensure proper dimensions for PIL (convert grayscale to RGB if needed)
            if crater_img.ndim == 2:
                crater_img = np.stack([crater_img] * 3, axis=-1)
            
            # Convert to PIL and apply transforms
            pil_img = Image.fromarray(crater_img.astype(np.uint8), mode="RGB")
            tensor_img = transform(pil_img)  # Shape: (3, 224, 224), range [0, 1]
            
            # Convert to target number of channels
            if num_channels == 1:
                # Convert RGB to grayscale for CAE
                tensor_img = tensor_img.mean(dim=0, keepdim=True)  # Shape: (1, 224, 224)
            
            # Store in memmap
            if crater_memmap is not None:
                crater_memmap[i] = tensor_img.numpy()
            
            # Save individual crop
            if save_raw_crops:
                crater_id = filtered_craters.iloc[i]["CRATER_ID"]
                crop_data = tensor_img.numpy()  # Shape: (C, H, W), NO normalization
                np.save(os.path.join(output_dir, f"{crater_id}.npy"), crop_data)
    
    if crater_memmap is not None:
        crater_memmap.flush()
    
    # Print flip statistics
    print(f"\n=== FLIP VERIFICATION (first {flip_check_count} craters) ===")
    print(f"Correctly flipped (shadow on right): {flipped_correctly}/{flip_check_count} "
          f"({100*flipped_correctly/flip_check_count:.1f}%)")
    if flipped_correctly < flip_check_count * 0.8:
        print("⚠️  WARNING: Less than 80% of craters have shadow on right!")
        print("   Check if flip_crater() logic is correct or if illumination is unusual")
    
    # === Compute dataset statistics (for reference only, NOT applied) ===
    print("\n=== Computing dataset statistics (for reference) ===")
    
    if crater_memmap is None:
        print("No memmap to compute statistics from")
        return
    
    chunk_size = 1000
    
    # Compute mean
    print("Computing mean...")
    sum_pixels = np.zeros(num_channels, dtype=np.float64)
    total_pixels = 0
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = crater_memmap[start_idx:end_idx]
        sum_pixels += chunk.sum(axis=(0, 2, 3))
        total_pixels += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
    
    dataset_mean = sum_pixels / total_pixels
    
    # Compute std
    print("Computing std...")
    sum_squared_diff = np.zeros(num_channels, dtype=np.float64)
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = crater_memmap[start_idx:end_idx]
        
        for c in range(num_channels):
            diff = chunk[:, c, :, :] - dataset_mean[c]
            sum_squared_diff[c] += (diff ** 2).sum()
    
    dataset_std = np.sqrt(sum_squared_diff / total_pixels)
    
    print(f"\n=== DATASET STATISTICS (NOT APPLIED) ===")
    print(f"Mean per channel: {dataset_mean}")
    print(f"Std per channel: {dataset_std}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Value range: [0, 1] (from ToTensor)")
    
    # === Validation ===
    print("\n=== VALIDATION: Dataset value ranges ===")
    
    sample_size = min(1000, N)
    sample_data = crater_memmap[:sample_size]
    print(f"Sample size: {sample_size}")
    print(f"Range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
    print(f"Mean: {sample_data.mean():.4f}")
    print(f"Std: {sample_data.std():.4f}")
    
    if num_channels == 3:
        print(f"Per-channel means: {sample_data.mean(axis=(0,2,3))}")
        print(f"Per-channel stds: {sample_data.std(axis=(0,2,3))}")
    
    # Save statistics (for reference if needed later)
    stats_path = output_path.replace('.dat', '_stats.npz')
    np.savez(stats_path, mean=dataset_mean, std=dataset_std)
    print(f"\nSaved statistics to {stats_path} (for reference only)")
    
    print(f"\n✅ Finished processing {N} craters. Data saved to {output_path}")
    print("NOTE: Data is in range [0, 1] with NO normalization applied!")


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', required=True, help='Path to lunar mosaic (LROC or albedo-corrected)')
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