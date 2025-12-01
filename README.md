# Convolutional Neural Network for Lunar Crater Identification

**Author:** Yarden Kinreich  

This repository contains a pipeline for processing lunar crater data, training a convolutional autoencoder, clustering latent features, and visualizing crater clusters on the lunar mosaic. The pipeline is fully managed with **Snakemake** for reproducible and automated runs.

---

## Project Overview

The goal of this project is to:

- Preprocess lunar crater images from the USGS Robbins dataset and the LRO LROC WAC mosaic.
- Train a convolutional autoencoder with configurable latent dimensions and regularization.
- Extract latent representations of craters for clustering.
- Visualize crater clusters as dots and images on the lunar mosaic.
- Maintain reproducibility through run names, fixed random seeds, and Snakemake workflows.

---

## Directory Structure
```bash
.
├── data/
│   ├── raw/                  # Original data files (craters CSV and mosaic)
│   └── processed/            # Preprocessed crater images and metadata
├── models/                   # Trained models and loss curves
├── results/                  # Clustering results and visualization figures
├── logs/                     # Trained models, loss curves and plotting results
├── src/
│   ├── data/                 # Preprocessing scripts
│   ├── train/                # Training scripts for autoencoder
│   ├── cluster/              # Clustering and plotting scripts
│   └── display/              # Run model on all craters and Visulization on Moon Mosiac
├── Snakefile                 # Snakemake workflow
├── display.py                # Optional visualization script
└── README.md
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone [text](https://github.com/yardenkinreich/autoencoder-project/tree/main)
    cd autoencoder-project
    ```
2. **Create a virtual environment and install dependencies:** # Add explanation
    ```bash
    # Using pixi
    ```

3. Prepare the data:
    - Download the Robbins crater database: 
        [Robbins Moon Crater Database](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins
    - Download the LRO LROC WAC Mosaic 100m: 
        [LRO LROC WAC Mosaic](https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif)
    Place the files in data/raw/.

## Running the Pipeline (with Snakemake)
1. Configure run name and toggles
- Edit the top of the Snakefile:
```python
RUN_NAME = "cnn_latent20_l2_sched"
RUN_PREPROCESS = True        # Set to False to skip preprocessing
RUN_DISPLAY = True           # Set to False to skip display step
TECHNIQUE = "pca"            # Choose dimensionality reduction technique: "pca" or "tsne"
LATENT_DIM = 20              # Set latent dimension size
```
- Adjust parameters in the different rules as needed: (for example, in the `train` rule)
```python
    params:
        latent_dim=LATENT_DIM,
        lr=1e-5,
        weight_decay=1e-5,
        lr_patience=5,
        min_lr=1e-8,
        lr_factor=0.5,
        num_samples = 50000
```

2. Execute the full Snakemake workflow:
```bash
snakemake --cores all
``` 
3. Execute specific steps (optional):
- Preprocess data only:
```bash
snakemake preprocess --cores all
```
- Train the model only:
```bash
snakemake train_autoencoder --cores all
```

## Pipeline Features
- Reproducible training: fixed random seed; can limit number of craters for quick experiments.
- Configurable parameters: Easily adjust latent dimensions, learning rates, and clustering methods.
- Clustering & visualization: PCA or other techniques to group craters; displayed as dots or images on the mosaic.
- Toggle steps: Preprocessing, training, and display can be turned on/off independently.
- Run snapshots: Each Snakemake run creates a snapshot of the workflow for reproducibility.

## MAE Model
facebook/vit-mae structure
Blocks 0-7   (Early):   Edges, textures, basic patterns → KEEP FROZEN
Blocks 8-15  (Middle):  Part-level features, shapes → MAYBE UNFREEZE
Blocks 16-23 (Late):    Domain-specific, high-level → DEFINITELY UNFREEZE
Decoder:                Task-specific reconstruction → ALWAYS UNFREEZE

## Output
All files are saved under logs/ in a folder with the run name. 
Key outputs include:
- Models: models/autoencoder.pth
- Loss plots: models/loss_curve.png
- Reconstructions: models/reconstructions.png
- Latent vectors: results/latents.npy
- Clustering Labeled Data: results/clustering_dots.png
- Clustering Labeled Data: results/clustering_imgs.png
- Classify Unlabled for all Craters: results/crater_clusters_kmeans.csv
- Classify Unlabled for all Craters: results/crater_clusters_kmeans_clusters.geojson


5. Very small raw sizes (53×53, 83×84) → heavy upsampling

This is something to pay attention to.

A crater that is 53×53 becomes 224×224:

You are upsampling by 4×

Interpolation smooths details

Model sees blurry inputs

This usually causes:

Worse reconstructions

Lower MAE patch accuracy

Blurry outputs

Not a pipeline error — just the nature of your data.






Facebook MAE Pretrained Models
base: https://huggingface.co/facebook/vit-mae-base
large: https://huggingface.co/facebook/vit-mae-large
huge: https://huggingface.co/facebook/vit-mae-huge

