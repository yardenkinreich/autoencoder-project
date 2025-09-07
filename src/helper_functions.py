from rasterio.windows import from_bounds
import numpy as np
import cv2
from numpy import cos, radians
import pyproj
import random
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Data Preprocessing Functions

def crop_crater(map_ref, lat, lon, diameter, offset, transformer, dst_h, dst_w):

    if lon > 180:
        lon -= 360
    # Convert latitude and longitude to map's coordinate system
    x, y = transformer.transform(lon, lat)
    # Define bounding box in projected coordinates
    radius = (diameter / 2) * 1000  # Convert km to meters
    radius_with_offset_x = (radius + radius * offset) / cos(radians(lat))
    radius_with_offset_y = radius + radius * offset
    min_x, min_y = x - radius_with_offset_x, y - radius_with_offset_y
    max_x, max_y = x + radius_with_offset_x, y + radius_with_offset_y

    # Get the window for cropping
    window = from_bounds(min_x, min_y, max_x, max_y, transform=map_ref.transform)

    # Read and crop the data
    cropped_image = map_ref.read(window=window)

    cropped_image = cropped_image.reshape((cropped_image.shape[1], cropped_image.shape[2]))

    projected_height = int(cropped_image.shape[0] / cos(radians(abs(lat))))
    if projected_height > cropped_image.shape[0]:
        cropped_image_projected = cv2.resize(cropped_image, (cropped_image.shape[1], projected_height))
    else:
        cropped_image_projected = cropped_image

    resized_image = cv2.resize(cropped_image_projected, (dst_w, dst_h))

    flipped_image = flip_crater(resized_image)


    # plt.subplot(1, 4, 1)
    # plt.imshow(cropped_image, cmap='gray')
    # plt.title(f'cylindrical')
    # plt.subplot(1, 4, 2)
    # plt.imshow(cropped_image_projected, cmap='gray')
    # plt.title(f'conformal')
    # plt.subplot(1, 4, 3)
    # plt.imshow(resized_image, cmap='gray')
    # plt.title(f'resized')
    # plt.subplot(1, 4, 4)
    # plt.imshow(flipped_image, cmap='gray')
    # plt.title(f'shadow flipped')
    # plt.suptitle(f'diamitter:{round(diameter, 0)}, lat:{round(lat, 0)}')
    # plt.show()

    return flipped_image



def flip_crater(img):
    '''
    Flips crater s.t. the shadow will always be on the r.h.s
    '''
    qtr_img_width = np.int16(img.shape[1] / 4)
    half_img_width = np.int16(img.shape[1] / 2)

    left_crater_side = img[:, qtr_img_width:half_img_width]
    right_crater_side = img[:, half_img_width:-qtr_img_width]

    if left_crater_side.mean() > right_crater_side.mean():
        pass
    else:
        img = np.fliplr(img)

    return img

def plot_latent_space(latents, technique, clusters=None):

    if latents.shape[1] > 2:
        technique = PCA(n_components=2)
        latents_2d = technique.fit_transform(latents)
    else:
        latents_2d = latents

    plt.figure(figsize=(8, 6))
    if clusters is not None:
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                        c=clusters, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter, label="Cluster ID")
    else:
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7)

    plt.title("Latent Space Visualization")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.tight_layout()
    plt.show()