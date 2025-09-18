import os
import numpy as np
import pandas as pd
import pyproj
import rasterio
import matplotlib.pyplot as plt
from src.helper_functions import *
import cv2


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
                transformer,
                dst_height,
                dst_width
            )
            
            # Flip crater so shadow is always on the right
            crater_img = flip_crater(crater_img)

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
    filtered_craters, map_file, output_dir, offset,
    mae_input_size=224, save_crops=False, save_np_array=True, output_path=None
):

    os.makedirs(output_dir, exist_ok=True)

    craters_crs = get_craters_crs()
    N = len(filtered_craters)

    if save_np_array and output_path is not None:
        # preallocate memmap file
        crater_memmap = np.memmap(
            output_path, dtype=np.float32, mode='w+', shape=(N, mae_input_size, mae_input_size, 3)
        )

    with rasterio.open(map_file) as map_ref:
        transformer = pyproj.Transformer.from_crs(
            craters_crs, map_ref.crs.to_string(), always_xy=True
        )

        for i, (_, crater) in enumerate(filtered_craters.iterrows()):
            if i % 1000 == 0:
                print(f"Processed {i} craters")

            crater_img = crop_crater(
                map_ref,
                crater['LAT_CIRC_IMG'],
                crater['LON_CIRC_IMG'],
                crater['DIAM_CIRC_IMG'],
                offset,
                transformer,
                mae_input_size,
                mae_input_size
            )

            crater_img = flip_crater(crater_img)
            crater_img = np.stack([crater_img]*3, axis=-1).astype(np.float32) / 255.0

            # Save to memmap
            if save_np_array and output_path is not None:
                crater_memmap[i] = crater_img

            if save_crops:
                filename = os.path.join(output_dir, f"{crater['CRATER_ID']}.jpeg")
                plt.imsave(filename, (crater_img*255).astype(np.uint8))

    if save_np_array and output_path is not None:
        crater_memmap.flush()
        print(f"Finished processing craters. Data saved to {output_path}")


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
    parser.add_argument('--save_crops', action='store_true')
    parser.add_argument('--save_np_array', action='store_true')
    parser.add_argument('--dst_height', type=int, default=100)
    parser.add_argument('--dst_width', type=int, default=100)
    parser.add_argument('--autoencoder_model', type=str, choices=['cnn', 'mae'], default='cnn')
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
            mae_input_size=224,
            save_crops=args.save_crops,
            save_np_array=args.save_np_array,
            output_path=args.np_output_path
        )

    save_crater_metadata(
        filtered,
        args.map_file,
        args.info_output_path
    )
