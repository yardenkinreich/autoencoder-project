import os
import numpy as np
import pandas as pd
import pyproj
import rasterio
import matplotlib.pyplot as plt
from src.helper_functions import *
import cv2
import argparse

# ── constants ────────────────────────────────────────────────────────────────
EXTRACT_SIZE = 320   # extract larger tile; rotation needs ~320/1.414 ≈ 226px
                     # of valid pixels before the 224 center-crop


def load_and_filter_craters(craters_csv, min_diameter, max_diameter,
                             latitude_bounds, craters_to_output):
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
        SPHEROID["Moon_2000_IAU_IAG",1737400,0, LENGTHUNIT["metre",1]]],
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
    num_channels=1,        # 1 for CAE, 3 for MAE
):
    """
    Preprocessing pipeline:
      1. Crop crater from map with offset  (crop_crater, NO internal flip)
      2. Resize to EXTRACT_SIZE × EXTRACT_SIZE  (320 px)
         — larger than the 224 model input so any rotation during training
           can be center-cropped to 224 with zero border artifacts
      3. Normalize to [0, 1]
      4. Save raw .npy files and/or a memory-mapped array

    Shadow orientation is NOT fixed here.
    Random rotation + flips are applied on-the-fly in the DataLoader.
    """
    os.makedirs(output_dir, exist_ok=True)
    craters_crs = get_craters_crs()
    N = len(filtered_craters)

    # ── memmap ───────────────────────────────────────────────────────────────
    crater_memmap = None
    if save_np_array and output_path is not None:
        shape = (N, num_channels, EXTRACT_SIZE, EXTRACT_SIZE)
        crater_memmap = np.memmap(output_path, dtype=np.float32,
                                  mode="w+", shape=shape)

    print(f"=== Processing {N} craters ===")
    print(f"Extract size : {EXTRACT_SIZE}×{EXTRACT_SIZE}  (model will crop to 224 at train time)")
    print(f"Channels     : {num_channels}")
    print(f"Shadow flip  : DISABLED — randomised in DataLoader")

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )

        for i, (_, crater) in enumerate(filtered_craters.iterrows()):
            if i % 10_000 == 0:
                print(f"  {i}/{N}")

            # ── crop (no internal flip) ───────────────────────────────────
            crater_img = crop_crater(
                map_ref,
                crater["LAT_CIRC_IMG"],
                crater["LON_CIRC_IMG"],
                crater["DIAM_CIRC_IMG"],
                offset,
                transformer
            )

            # ── resize to EXTRACT_SIZE (uniform scaling, crater is square) ─
            crater_resized = cv2.resize(
                crater_img,
                (EXTRACT_SIZE, EXTRACT_SIZE),
                interpolation=cv2.INTER_CUBIC
            )

            # ── normalize to [0, 1] ──────────────────────────────────────
            c_min = crater_resized.min()
            c_max = crater_resized.max()
            if c_max > c_min:
                crater_norm = (crater_resized - c_min) / (c_max - c_min)
            else:
                crater_norm = np.zeros_like(crater_resized, dtype=np.float32)

            # ── channel layout  (C, H, W) ────────────────────────────────
            if num_channels == 1:
                tensor = crater_norm[np.newaxis, :, :].astype(np.float32)   # (1,320,320)
            else:
                tensor = np.stack([crater_norm] * 3, axis=0).astype(np.float32)  # (3,320,320)

            # ── store ────────────────────────────────────────────────────
            if crater_memmap is not None:
                crater_memmap[i] = tensor

            if save_raw_crops:
                crater_id = filtered_craters.iloc[i]["CRATER_ID"]
                np.save(os.path.join(output_dir, f"{crater_id}.npy"), tensor)

    if crater_memmap is not None:
        crater_memmap.flush()

    # ── dataset statistics (informational only) ───────────────────────────
    if crater_memmap is not None:
        print("\n=== Dataset statistics (first 5000 samples) ===")
        sample = crater_memmap[:min(5000, N)]
        print(f"  Range : [{sample.min():.4f}, {sample.max():.4f}]")
        print(f"  Mean  : {sample.mean():.4f}")
        print(f"  Std   : {sample.std():.4f}")
        if num_channels == 3:
            print(f"  Per-channel mean : {sample.mean(axis=(0,2,3))}")
            print(f"  Per-channel std  : {sample.std(axis=(0,2,3))}")

        stats_path = output_path.replace('.dat', '_stats.npz')
        np.savez(stats_path,
                 mean=sample.mean(axis=(0, 2, 3)),
                 std=sample.std(axis=(0, 2, 3)))
        print(f"  Saved stats → {stats_path}")

    print(f"\n✅ Done. {N} craters saved to {output_path}")
    print(f"   Shape per sample : ({num_channels}, {EXTRACT_SIZE}, {EXTRACT_SIZE})")
    print(f"   Training will center-crop to 224×224 after random rotation.")


def save_crater_metadata(filtered_craters, map_file, output_path):
    craters_crs = get_craters_crs()
    buffer = []
    batch_size = 10_000

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )
        for i, crater in filtered_craters.iterrows():
            lon = crater['LON_CIRC_IMG']
            lat = crater['LAT_CIRC_IMG']
            diam = crater['DIAM_CIRC_IMG']
            crater_id = crater['CRATER_ID']
            if lon > 180:
                lon -= 360
            x, y = transformer.transform(lon, lat)
            diam *= 1000
            buffer.append({'id': crater_id, 'lat': lat, 'lon': lon,
                            'x': x, 'y': y, 'diam': diam})
            if (i + 1) % batch_size == 0:
                print(f"Metadata: processed {i + 1}")
                df = pd.DataFrame(buffer)
                mode = 'a' if os.path.exists(output_path) else 'w'
                df.to_csv(output_path, mode=mode,
                          header=not os.path.exists(output_path), index=False)
                buffer.clear()

        if buffer:
            df = pd.DataFrame(buffer)
            mode = 'a' if os.path.exists(output_path) else 'w'
            df.to_csv(output_path, mode=mode,
                      header=not os.path.exists(output_path), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file',         required=True)
    parser.add_argument('--craters_csv',      required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--np_output_path',   required=True)
    parser.add_argument('--info_output_path', required=True)
    parser.add_argument('--min_diameter',     type=float, default=3.0)
    parser.add_argument('--max_diameter',     type=float, default=10.0)
    parser.add_argument('--latitude_bounds',  type=float, nargs=2, default=[-60, 60])
    parser.add_argument('--offset',           type=float, default=0.5)
    parser.add_argument('--craters_to_output',type=int,   default=-1)
    parser.add_argument('--save_raw_crops',   action='store_true')
    parser.add_argument('--save_np_array',    action='store_true')
    parser.add_argument('--autoencoder_model',type=str,
                        choices=['cae', 'mae'], default='mae')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filtered = load_and_filter_craters(
        args.craters_csv, args.min_diameter, args.max_diameter,
        args.latitude_bounds, args.craters_to_output
    )
    print(f"Filtered {len(filtered)} craters")

    num_channels = 1

    process_and_save_crater_crops(
        filtered, args.map_file, args.output_dir, args.offset,
        save_raw_crops=args.save_raw_crops,
        save_np_array=args.save_np_array,
        output_path=args.np_output_path,
        num_channels=num_channels,
    )
    save_crater_metadata(filtered, args.map_file, args.info_output_path)
    print("All done.")