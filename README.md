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
├── configs/                   # eval_suite.yaml, dino_craters.yaml, crater-ID registries, global scaling
├── data/
│   ├── raw/                   # Downloaded/derived source files (craters CSV, mosaic, labeled sets)
│   └── processed_wac_100m_new/  # Preprocessed crater crops + memmaps (per sigma/FOV/offset variant)
├── logs/                      # Every training run + its eval results, one folder per run
│   └── {family}/.../{run}/
│       ├── models/            # autoencoder.pth, loss_curve.png
│       ├── run_manifest.json  # training params, read by the dashboard
│       └── eval_results/      # per-checkpoint eval.evaluate() output (see "Evaluation" below)
├── notebooks/                 # Interactive review/eval notebooks
├── src/
│   ├── data/                  # Preprocessing scripts (preprocess_2.py, preprocess_dino.py)
│   ├── train/                  # Training entry points (train.py, train_dino.py)
│   ├── models/                 # Model definitions (autoencoder, MAE, vendored dinov2/)
│   ├── cluster/                 # Clustering and encoding helpers
│   ├── display/                 # Run a trained model on all craters + visualize on the mosaic
│   ├── eval/                    # Eval harness (evaluate.py, metrics.py, visualize.py, history.py, ...)
│   └── dashboard/                # Streamlit dashboard (see "Dashboard" below)
├── Snakefile                  # Snakemake workflow (preprocessing, training, DINO, eval test sets)
└── README.md
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yardenkinreich/autoencoder-project.git
   cd autoencoder-project
   ```
2. **Install dependencies** (this project uses [pixi](https://pixi.sh), not a plain venv — `pixi.toml` pins everything, including CUDA-enabled PyTorch):
    ```bash
    pixi install
    # then run any command via:
    pixi run -e default <command>
    # e.g.:
    pixi run -e default python -m eval.evaluate --help
    ```

3. Prepare the data — place these under `data/raw/`:
    - Robbins crater database:
        [Robbins Moon Crater Database](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins) → `lunar_crater_database_robbins_2018.csv`
    - LRO LROC WAC Mosaic 100m:
        [LRO LROC WAC Mosaic](https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif)

    The Snakefile doesn't read that mosaic file directly — it expects a locally sigma/high-pass-filtered version at `data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif` (see `src/data/full_mosaic_processing.py` for the filtering step) and a matching `configs/global_scaling.json` (frozen normalization range).

## Running the Pipeline (with Snakemake)
1. Configure the run — edit the top of the `Snakefile`:
```python
AUTOENCODER_MODEL = "mae"       # "cae" or "mae"
LATENT_DIM        = 64          # only used by "cae"
EPOCHS            = 50
MASK_RATIO        = 0.75        # only used by "mae"
MIN_DIAMETER      = 3.0         # crater diameter filter, km
MAX_DIAMETER      = 10.0
NUM_SAMPLES       = 25000       # how much of INPUT_DIR to actually train on
PRETRAINED_WEIGHTS = ""         # path to a .pth checkpoint to fine-tune from; "" = scratch
NUM_GPUS          = 1           # >1 launches training via torchrun (single node)
```
`RUN_DIR` (where everything for this run gets written under `logs/`) is derived automatically from these params — runs with identical settings resume/overwrite the same folder instead of fragmenting into new ones. There's no separate run-name variable to set.

2. Execute the full Snakemake workflow (preprocess → train → cluster/display):
```bash
pixi run -e default snakemake --cores all
```
3. Or execute specific steps:
```bash
pixi run -e default snakemake preprocess_craters --cores all     # preprocess only
pixi run -e default snakemake train_autoencoder --cores all      # train only (assumes preprocess_craters already ran)
```
4. DINOv2 is a separate opt-in pipeline (self-distillation, not reconstruction) — not part of `rule all`:
```bash
pixi run -e default snakemake train_dino --cores all
```
Configure it via the `DINO_*` variables near the top of the Snakefile (`DINO_EPOCHS`, `DINO_GPUS`, `DINO_CLEAN_OFFSET`) and `configs/dino_craters.yaml`.

## Pipeline Features
- Reproducible training: fixed random seed; a manifest (`run_manifest.json`) is written alongside every checkpoint recording exactly how it was trained.
- Configurable parameters: latent dimension, epochs, mask ratio, diameter range, pretrained-vs-scratch — all set at the top of the Snakefile.
- Clustering & visualization: PCA or other techniques group craters; displayed as dots or images on the mosaic (`RUN_DISPLAY`/`RUN_CLUSTER_JULIE` toggles).
- Held-out test sets: craters reserved for eval are excluded from training automatically (see "Evaluation" below) — they never leak into the training sample.
- Run snapshots: each Snakemake run copies the `Snakefile` as it was at train time into the run's own log folder, for reproducibility.

## Evaluation

Every trained checkpoint (mae/cae/dino, scratch or pretrained) can be scored against the shared test-set definitions in `configs/eval_suite.yaml`: one or more **labeled** sets (accuracy/F1/QWK/ordinal-MAE/ECE/clustering-agreement/confound checks) plus one **held-out unlabeled** set (reconstruction loss + unsupervised clustering quality).

1. **Materialize the test sets** (one-time, or whenever a set's crater-ID list changes):
    ```bash
    pixi run -e default snakemake preprocess_holdout_set    # unlabeled holdout (configs/holdout_crater_ids.csv)
    pixi run -e default snakemake preprocess_new_test_set   # labeled v-fr/r-fr/r-dr/v-dr set (configs/correct_crater_with_labels_final.csv)
    ```
    Both are cut from the same mosaic/offset as training and are automatically excluded from `preprocess_craters`/`preprocess_craters_dino`'s training data.

2. **Run the eval suite** against a checkpoint:
    ```bash
    PYTHONPATH=src pixi run -e default python -m eval.evaluate \
        --checkpoint logs/mae/scratch/none/na/.../ep50/models/autoencoder.pth \
        --autoencoder-model mae \
        --config configs/eval_suite.yaml
    ```
    `--autoencoder-model` is one of `mae`, `cae`, `dino`, `mae_pretrained`, `dino_pretrained`. Output defaults to `{training_run_dir}/eval_results/<run_id>/` (override with `--out`); useful extra flags: `--skip-holdout` (if the holdout set hasn't been preprocessed yet), `--no-history` (don't append to `logs/eval_history.csv`), `--n-boot` (bootstrap CI resamples, default 2000).

3. **Compare runs by metric**, without the dashboard:
    ```bash
    PYTHONPATH=src pixi run -e default python -m eval.history --metric julie_accuracy --top 10
    ```

## Dashboard

Browse every checkpoint's results (evaluated or not) and compare runs against each other:

```bash
PYTHONPATH=src pixi run -e default streamlit run src/dashboard/app.py
```

Opens one app in your browser with two pages (sidebar-navigable):
- **Run Deep Dive** — pick any trained checkpoint under `logs/`, see its training parameters, artifacts, and full eval results (or a ready-to-run eval command if it hasn't been evaluated yet).
- **Compare Runs** — filter/group runs by architecture or training parameters, chart metrics side by side, and browse a leaderboard across the full run history.

It reads only what `train.py`/`evaluate.py` already wrote (`logs/eval_history.csv`, `logs/**/eval_results/*/summary.json`, `run_manifest.json`) — nothing is recomputed.

## Output
All files are saved under `logs/{family}/.../{run}/` (see "Directory Structure" above — the exact path is derived from the run's own parameters). Key outputs include:
- Model: `models/autoencoder.pth`
- Loss plot: `models/loss_curve.png`
- Reconstructions: `models/reconstructions.png`
- Training params: `run_manifest.json`
- Clustering (Julie's labeled set): `results/clustering_dots_{technique}.png`, `results/clustering_imgs_{technique}.png`
- Clustering (all craters, unlabeled): `results/crater_clusters_{n}.csv`, `results/crater_clusters_{n}.geojson`
- Eval results (once `eval.evaluate` has been run against the checkpoint): `eval_results/<run_id>/` — see "Evaluation" above.

## MAE Model

Structure: `facebook/vit-mae-{base,large,huge}`. When fine-tuning, only unfreeze as many blocks as the target task actually needs — earlier blocks encode more universal, transferable features:

```
┌─────────────────────────────────────────────────────────┐
│ ENCODER BLOCKS 0-7: LOW-LEVEL FEATURES                 │
│ (Keep Frozen ❄️ - Universal Features)                   │
├─────────────────────────────────────────────────────────┤
│ Block 0-1:   Patch embeddings, basic edge detection    │
│ Block 2-3:   Simple textures, local patterns           │
│ Block 4-5:   Corners, curves, basic shapes             │
│ Block 6-7:   Color/intensity patterns, simple combos   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ENCODER BLOCKS 8-15: MID-LEVEL FEATURES                │
│ (Optional to Unfreeze 🔓 - Semi-specific)               │
├─────────────────────────────────────────────────────────┤
│ Block 8-9:   Part-level features (rim sections)        │
│ Block 10-11: Object parts relationships                │
│ Block 12-13: Medium-range spatial context              │
│ Block 14-15: Intermediate semantic features            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ENCODER BLOCKS 16-23: HIGH-LEVEL FEATURES              │
│ (Must Unfreeze 🔓 - Domain-specific)                    │
├─────────────────────────────────────────────────────────┤
│ Block 16-17: Complete object detection                 │
│ Block 18-19: Object-level semantics                    │
│ Block 20-21: Global context & relationships            │
│ Block 22-23: Task-specific abstract features           │
└─────────────────────────────────────────────────────────┘

Decoder: task-specific reconstruction → always unfrozen.
```

Pretrained weights: [base](https://huggingface.co/facebook/vit-mae-base), [large](https://huggingface.co/facebook/vit-mae-large), [huge](https://huggingface.co/facebook/vit-mae-huge).

**Watch out for small raw crops** (e.g. 53×53, 83×84) getting heavily upsampled to the model's input size (224×224 or 128×128 depending on pipeline) — a 53×53 crop is a ~4× upsample, and interpolation smooths out detail the model then sees as blurry. This shows up as worse reconstructions and lower patch accuracy; it's a property of the source data at that crater's diameter, not a pipeline bug.
