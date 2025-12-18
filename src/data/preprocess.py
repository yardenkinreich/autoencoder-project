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


def process_and_save_crater_crops(filtered_craters, map_file, output_dir, offset, dst_height, dst_width,
                                   save_crops, save_np_array, output_path):
    craters_list = []
    craters_crs = get_craters_crs()

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(craters_crs, map_ref.crs.to_string(), always_xy=True)

        for i, crater in filtered_craters.iterrows():
            if i % 10000 == 0:
                print(f"Processed {i} craters")
            crater_img = crop_crater(
                map_ref,
                crater['LAT_CIRC_IMG'],
                crater['LON_CIRC_IMG'],
                crater['DIAM_CIRC_IMG'],
                offset,
                transformer
            )

            craters_list.append((crater_img / 255).astype(np.float16))

            if save_crops:
                filename = os.path.join(output_dir, f"{crater['CRATER_ID']}.jpeg")
                plt.imsave(filename, crater_img, cmap='gray')

    if save_np_array:
        craters_array = np.array(craters_list)
        craters_array = craters_array.reshape(craters_array.shape[0], dst_height * dst_width)
        print(craters_array.shape, craters_array.dtype)
        np.save(output_path, craters_array)

def process_and_save_crater_crops_mae(
    filtered_craters,
    map_file,
    output_dir,
    offset,
    save_raw_crops=True,
    save_np_array=True,
    output_path=None,
    batch_size=64,
    autoencoder_model="mae"
):
    os.makedirs(output_dir, exist_ok=True)

    craters_crs = get_craters_crs()
    N = len(filtered_craters)

    # === Step 1: First pass - collect all images in [0, 1] range ===
    mae_transform_no_norm = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()  # Just converts to [0, 1]
    ])

    temp_memmap = None
    if save_np_array and output_path is not None:
        temp_memmap = np.memmap(
            output_path + ".temp", dtype=np.float32, mode="w+", shape=(N, 3, 224, 224)
        )
    
    print("=== PASS 1: Loading images ===")
    
    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )

        for i, (_, crater) in enumerate(filtered_craters.iterrows()):
            if i % 10000 == 0:
                print(f"Processed {i}/{N}")

            crater_img = crop_crater(
                map_ref,
                crater["LAT_CIRC_IMG"],
                crater["LON_CIRC_IMG"],
                crater["DIAM_CIRC_IMG"],
                offset,
                transformer,
                autoencoder_model
            )

            if crater_img.ndim == 2:
                crater_img = np.stack([crater_img] * 3, axis=-1)
            
            if save_raw_crops:
                crater_id = crater["CRATER_ID"]
                np.save(os.path.join(output_dir, f"{crater_id}.npy"), crater_img)

            pil_img = Image.fromarray(crater_img.astype(np.uint8), mode="RGB")
            tensor_img = mae_transform_no_norm(pil_img)

            if temp_memmap is not None:
                temp_memmap[i] = tensor_img.numpy()

    if temp_memmap is None:
        print("Error: temp_memmap is None")
        return
    
    temp_memmap.flush()
    
    # === Step 2: Compute dataset mean and std ===
    print("\n=== Computing dataset statistics ===")
    
    chunk_size = 1000
    
    # STEP 2A: Compute overall mean
    print("Computing mean...")
    sum_pixels = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = temp_memmap[start_idx:end_idx]
        sum_pixels += chunk.sum(axis=(0, 2, 3))  # Sum per channel
        total_pixels += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
    
    dataset_mean = sum_pixels / total_pixels
    
    # STEP 2B: Compute overall std using the mean
    print("Computing std...")
    sum_squared_diff = np.zeros(3, dtype=np.float64)
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = temp_memmap[start_idx:end_idx]
        
        for c in range(3):
            diff = chunk[:, c, :, :] - dataset_mean[c]
            sum_squared_diff[c] += (diff ** 2).sum()
    
    dataset_std = np.sqrt(sum_squared_diff / total_pixels)
    
    print(f"\n=== DATASET STATISTICS (CORRECT) ===")
    print(f"Mean per channel (R, G, B): {dataset_mean}")
    print(f"Std per channel (R, G, B): {dataset_std}")
    print(f"Total pixels used: {total_pixels:,}")
    
    # Sanity check
    if np.any(dataset_std < 0.01):
        print("⚠️  WARNING: Very low std detected. Craters might be too similar.")
    if np.any(dataset_std > 0.5):
        print("⚠️  WARNING: Very high std detected. Might have outliers.")
    
    # === Step 3: Second pass - apply dataset normalization ===
    print("\n=== PASS 2: Applying dataset normalization ===")
    
    crater_memmap = np.memmap(
        output_path, dtype=np.float32, mode="w+", shape=(N, 3, 224, 224)
    )
    
    for start_idx in range(0, N, chunk_size):
        if start_idx % 10000 == 0:
            print(f"Normalizing {start_idx}/{N}")
        
        end_idx = min(start_idx + chunk_size, N)
        chunk = temp_memmap[start_idx:end_idx].copy()  # Make a copy
        
        # Apply normalization: (x - mean) / std
        for c in range(3):
            chunk[:, c, :, :] = (chunk[:, c, :, :] - dataset_mean[c]) / (dataset_std[c] + 1e-8)
        
        crater_memmap[start_idx:end_idx] = chunk
    
    crater_memmap.flush()
    
    # Clean up temp file
    del temp_memmap
    os.remove(output_path + ".temp")
    
    print(f"\n✅ Finished processing {N} craters. Data saved to {output_path}")
    
    # === FINAL DIAGNOSTIC ===
    print("\n=== FINAL NORMALIZED DATASET STATISTICS ===")
    sample_size = min(1000, N)
    sample_data = crater_memmap[:sample_size]
    print(f"Sample size: {sample_size}")
    print(f"Overall range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
    print(f"Overall mean: {sample_data.mean():.4f} (should be ~0)")
    print(f"Overall std: {sample_data.std():.4f} (should be ~1)")
    print(f"Per-channel means: {sample_data.mean(axis=(0,2,3))} (should be ~[0, 0, 0])")
    print(f"Per-channel stds: {sample_data.std(axis=(0,2,3))} (should be ~[1, 1, 1])")
    
    # Check for outliers
    print(f"\n=== OUTLIER CHECK ===")
    print(f"Values < -3: {(sample_data < -3).sum()} pixels")
    print(f"Values > 3: {(sample_data > 3).sum()} pixels")
    print(f"99th percentile: {np.percentile(sample_data, 99):.4f}")
    print(f"1st percentile: {np.percentile(sample_data, 1):.4f}")
    
    # Save statistics for later use
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
    parser.add_argument('--dst_height', type=int, default=100)
    parser.add_argument('--dst_width', type=int, default=100)
    parser.add_argument('--autoencoder_model', type=str, choices=['cnn', 'mae'], default='cnn')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default='facebook/vit-mae-large')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filtered = load_and_filter_craters(
        args.craters_csv,
        args.min_diameter,
        args.max_diameter,
        args.latitude_bounds,
        args.craters_to_output
    )
    if args.autoencoder_model == 'cnn':
        process_and_save_crater_crops(
            filtered,
            args.map_file,
            args.output_dir,
            args.offset,
            args.dst_height,
            args.dst_width,
            args.save_crops,
            args.save_np_array,
            args.np_output_path
        )
    elif args.autoencoder_model == 'mae':   
        process_and_save_crater_crops_mae(
            filtered,
            args.map_file,
            args.output_dir,
            args.offset,
            save_raw_crops=args.save_raw_crops,
            save_np_array=args.save_np_array,
            output_path=args.np_output_path,
            batch_size=args.batch_size,
            autoencoder_model=args.autoencoder_model
        )

    save_crater_metadata(
        filtered,
        args.map_file,
        args.info_output_path
    )
