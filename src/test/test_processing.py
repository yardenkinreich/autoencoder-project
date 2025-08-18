import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

mine_dir = "data/test_processing"
daniel_dir = "/Users/yardenkinreich/Documents/Projects/Masters/daniel_crater_autoencoder/OneDrive_2025-08-03/Craters Classifier/craters_dataset"

mine_files = sorted(f for f in os.listdir(mine_dir) if f.endswith(".jpeg"))
daniel_files = sorted(f for f in os.listdir(daniel_dir) if f.endswith(".jpeg"))

common = set(mine_files) & set(daniel_files)
print(f"Found {len(common)} common crater crops to compare.")

for fname in sorted(common):
    mine_img = np.array(Image.open(os.path.join(mine_dir, fname)).convert("L"))
    daniel_img = np.array(Image.open(os.path.join(daniel_dir, fname)).convert("L"))

    if mine_img.shape != daniel_img.shape:
        print(f"⚠️ Shape mismatch for {fname}: mine {mine_img.shape}, daniel {daniel_img.shape}")
        continue

    diff = np.abs(mine_img.astype(int) - daniel_img.astype(int))
    max_diff = diff.max()
    mean_diff = diff.mean()

    if max_diff == 0:
        print(f"✅ {fname}: identical")
    else:
        print(f"❌ {fname}: max diff {max_diff}, mean diff {mean_diff:.2f}")
        
        # Plot heatmap
        plt.figure(figsize=(6, 6))
        plt.imshow(diff, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Pixel difference')
        plt.title(f'Difference Heatmap: {fname}')
        plt.show()
