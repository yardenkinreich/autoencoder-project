import os
import datetime
import shutil


# --- Parameters of the Run ---
AUTOENCODER_MODEL = "mae"       # "cae" or "mae"
LATENT_DIM        = 64
TECHNIQUE         = "pca"       # "pca" or "tsne" or "umap"
NUM_CLUSTERS      = 4
CLUSTER_METHOD    = "kmeans"    # "kmeans" or "gmm" or "spectral" or "agglomerative" or "hdbscan"
EPOCHS            = 50
MASK_RATIO        = 0.75
MIN_DIAMETER      = 3.0
MAX_DIAMETER      = 10.0
PRETRAINED_WEIGHTS = ""
DATASET = 'processed_wac_100m_new_augmented_no_left_band'  # for logging purposes, set to "" if not using a pretrained model
SIGMA = "100"  # local .pth, set to "" to train from scratch

# --- Run naming ---
if AUTOENCODER_MODEL == "cae":
    RUN_NAME = f"{MIN_DIAMETER}_{MAX_DIAMETER}_{EPOCHS}_{LATENT_DIM}"
    RUN_DIR  = f"logs/{AUTOENCODER_MODEL}/{DATASET}/{SIGMA}/{RUN_NAME}"
elif AUTOENCODER_MODEL == "mae":
    RUN_NAME = f"{MIN_DIAMETER}_{MAX_DIAMETER}_{EPOCHS}_mask{MASK_RATIO}"
    RUN_DIR  = f"logs/{AUTOENCODER_MODEL}/{DATASET}/{SIGMA}/{RUN_NAME}"
else:
    raise ValueError("Unsupported AUTOENCODER_MODEL. Choose 'cae' or 'mae'.")

MODELS_DIR  = f"{RUN_DIR}/models"
RESULTS_DIR = f"{RUN_DIR}/results"
TEST_DIR    = f"{RUN_DIR}/tests/edge_cases"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_DIR,    exist_ok=True)

if os.path.exists("Snakefile"):
    shutil.copy("Snakefile", f"{RUN_DIR}/Snakefile.snapshot")

# --- Config toggles ---
RUN_PREPROCESS    = False
RUN_DISPLAY       = True
RUN_CLUSTER_JULIE = True

# --- Rule all ---
rule all:
    input:
        f"{MODELS_DIR}/autoencoder.pth",
        f"{MODELS_DIR}/loss_curve.png",
        f"{MODELS_DIR}/reconstructions.png",
        *([f"{RESULTS_DIR}/clustering_dots_{TECHNIQUE}.png",
           f"{RESULTS_DIR}/clustering_imgs_{TECHNIQUE}.png"] if RUN_CLUSTER_JULIE else []),
        *([f"{RESULTS_DIR}/crater_clusters_{NUM_CLUSTERS}.csv"] if RUN_DISPLAY else []),
        f"{RUN_DIR}/rules.txt",
        f"{RUN_DIR}/summary.txt",
        f"{RUN_DIR}/dag.pdf"


# --- Preprocessing ---

rule preprocess_craters:
    input:
        map_file    = "data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif",
        craters_csv = "data/raw/lunar_crater_database_robbins_2018.csv"
    output:
        output_dir_clean = directory("data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/crater_crops_clean"),
        output_dir_aug   = directory("data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/crater_crops_aug"),
        np_output_clean  = "data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_clean.dat",
        np_output_aug    = "data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat",
        metadata_output  = "data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/metadata.csv"
    params:
        min_diameter      = MIN_DIAMETER,
        max_diameter      = MAX_DIAMETER,
        lat_min           = -60,
        lat_max           = 60,
        craters_to_output = -1,
        autoencoder_model = AUTOENCODER_MODEL
    shell:
        """
        PYTHONPATH=$(pwd) python src/data/preprocess_2.py \
            --map_file {input.map_file} \
            --craters_csv {input.craters_csv} \
            --output_dir_clean {output.output_dir_clean} \
            --output_dir_aug {output.output_dir_aug} \
            --np_output_path_clean {output.np_output_clean} \
            --np_output_path_aug {output.np_output_aug} \
            --info_output_path {output.metadata_output} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --latitude_bounds {params.lat_min} {params.lat_max} \
            --craters_to_output {params.craters_to_output} \
            --save_raw_crops \
            --save_np_array \
            --autoencoder_model {params.autoencoder_model} \
            --exclude_lon_bounds 180 270

        """

# --- Main training ---
rule train_autoencoder:
    input:
        data = "data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat"
    output:
        model     = f"{MODELS_DIR}/autoencoder.pth",
        loss_plot = f"{MODELS_DIR}/loss_curve.png"
    params:
        autoencoder_model = AUTOENCODER_MODEL,
        epochs            = EPOCHS,
        batch_size        = 128,
        lr                = 1e-3,
        weight_decay      = 1e-5,
        min_lr            = 1e-8,
        val_split         = 0.2,
        mask_ratio        = MASK_RATIO,
        latent_dim        = LATENT_DIM,
        pretrained        = PRETRAINED_WEIGHTS if AUTOENCODER_MODEL == "mae" else "",
        num_samples        = 25000 
    shell:
        """
        PYTHONPATH=$(pwd) python src/train/train.py \
            --autoencoder_model {params.autoencoder_model} \
            --num_samples {params.num_samples} \
            --input {input.data} \
            --model_output {output.model} \
            --loss_plot {output.loss_plot} \
            --epochs {params.epochs} \
            --batch_size {params.batch_size} \
            --lr {params.lr} \
            --weight_decay {params.weight_decay} \
            --min_lr {params.min_lr} \
            --val_split {params.val_split} \
            --mask_ratio {params.mask_ratio} \
            --latent_dim {params.latent_dim} \
            $([ -n "{params.pretrained}" ] && echo "--pretrained_weights {params.pretrained}")
        """


# --- Reconstruction ---
rule reconstruct_craters:
    input:
        npy   = f"data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat",
        model = f"{MODELS_DIR}/autoencoder.pth"
    output:
        reconstructions = f"{MODELS_DIR}/reconstructions.png"
    params:
        autoencoder_model = AUTOENCODER_MODEL,
        device            = "cpu",
        num_images        = 8,
        latent_dim        = LATENT_DIM,
        mask_ratio        = MASK_RATIO
    shell:
        """
        PYTHONPATH=$(pwd) python src/train/reconstruct.py \
            --autoencoder_model {params.autoencoder_model} \
            --input {input.npy} \
            --model {input.model} \
            --device {params.device} \
            --file_out {output.reconstructions} \
            --num_images {params.num_images} \
            --latent_dim {params.latent_dim} \
            --mask_ratio {params.mask_ratio}
        """


# --- Clustering ---
rule encode_latents:
    input:
        imgs_dir = "data/raw/craters_for_danny",
        model    = f"{MODELS_DIR}/autoencoder.pth"
    output:
        latents = f"{RESULTS_DIR}/latents_julie.npy",
        states  = f"{RESULTS_DIR}/states_julie.npy"
    params:
        autoencoder_model = AUTOENCODER_MODEL,
        bottleneck        = LATENT_DIM,
        mask_ratio        = MASK_RATIO
    shell:
        """
        PYTHONPATH=$(pwd) python src/cluster/cluster.py encode \
            --imgs-dir {input.imgs_dir} \
            --model {input.model} \
            --bottleneck {params.bottleneck} \
            --out-latents {output.latents} \
            --out-states {output.states} \
            --autoencoder-model {params.autoencoder_model} \
        """

rule plot_latent_dots:
    input:
        latents = f"{RESULTS_DIR}/latents_julie.npy",
        states  = f"{RESULTS_DIR}/states_julie.npy"
    output:
        f"{RESULTS_DIR}/clustering_dots_{TECHNIQUE}.png"
    params:
        technique  = TECHNIQUE,
        model_name = AUTOENCODER_MODEL
    shell:
        """
        PYTHONPATH=$(pwd) python src/cluster/cluster.py plot-dots \
            --latents {input.latents} \
            --states {input.states} \
            --out-png {output} \
            --model-name {params.model_name} \
            --technique {params.technique}
        """

rule plot_latent_imgs:
    input:
        latents  = f"{RESULTS_DIR}/latents_julie.npy",
        imgs_dir = "data/raw/craters_for_danny"
    output:
        f"{RESULTS_DIR}/clustering_imgs_{TECHNIQUE}.png"
    params:
        technique  = TECHNIQUE,
        model_name = AUTOENCODER_MODEL
    shell:
        """
        PYTHONPATH=$(pwd) python src/cluster/cluster.py plot-imgs \
            --latents {input.latents} \
            --imgs-dir {input.imgs_dir} \
            --out-png {output} \
            --model-name {params.model_name} \
            --technique {params.technique}
        """

rule display_clusters:
    input:
        model    = f"{MODELS_DIR}/autoencoder.pth",
        dataset  = f"data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/craters_aug.dat",
        metadata = f"data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band/metadata.csv"
    output:
        df             = f"{RESULTS_DIR}/crater_clusters_{NUM_CLUSTERS}.csv",
        clustering_png = f"{RESULTS_DIR}/crater_clusters_{NUM_CLUSTERS}.png"
    params:
        autoencoder_model = AUTOENCODER_MODEL,
        num_clusters      = NUM_CLUSTERS,
        batch_size        = 128,
        device            = "cuda",
        latent_dim        = LATENT_DIM,
        cluster_method    = CLUSTER_METHOD,
        technique         = TECHNIQUE,
        latent_output     = f"{MODELS_DIR}/latent_vectors.npy"
        
    run:
        if RUN_DISPLAY:
            shell("""
            PYTHONPATH=$(pwd) python src/display/display.py \
                --model_path {input.model} \
                --dataset_path {input.dataset} \
                --metadata_path {input.metadata} \
                --num_clusters {params.num_clusters} \
                --batch_size {params.batch_size} \
                --device {params.device} \
                --latent_dim {params.latent_dim} \
                --out_df {output.df} \
                --cluster_method {params.cluster_method} \
                --technique {params.technique} \
                --latent_output {params.latent_output} \
                --autoencoder_model {params.autoencoder_model} \
                --use_gpu \
                --clustering_png {output.clustering_png}
            """)
        else:
            print("Skipping display_clusters rule")


# --- Workflow snapshot ---
rule snapshot_workflow:
    output:
        dag     = f"{RUN_DIR}/dag.pdf",
        summary = f"{RUN_DIR}/summary.txt",
        rules   = f"{RUN_DIR}/rules.txt"
    shell:
        """
        snakemake --dag | dot -Tpdf > {output.dag}
        snakemake --summary > {output.summary}
        snakemake --list > {output.rules}
        """
