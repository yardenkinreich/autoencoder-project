import os
import datetime
import shutil


# --- Parameters of the Run ---
AUTOENCODER_MODEL = "mae"  # "cnn" or "mae"
LATENT_DIM = 64
PRETRAINED_MODEL = "facebook/vit-mae-large"
FREEZE_UNTIL = -2  # number of encoder transformer blocks to freeze from the end (negative number)
TECHNIQUE = "pca" # "pca" or "tsne"
NUM_CLUSTERS = 4
CLUSTER_METHOD = "kmeans"  # "kmeans" or "gmm"
EPOCHS = 200 # number of training epochs

# --- Define the run name once ---
#RUN_NAME = f"{AUTOENCODER_MODEL}_{}" # fr for freeze_until, l2 for weight decay, 1_10 for diameter range, 500 for epochs
RUN_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

RUN_DIR = f"logs/{AUTOENCODER_MODEL}/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)

# Create subfolders inside the run directory
MODELS_DIR = f"{RUN_DIR}/models"
RESULTS_DIR = f"{RUN_DIR}/results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Snapshot Snakefile ---
if os.path.exists("Snakefile"):
    shutil.copy("Snakefile", f"{RUN_DIR}/Snakefile.snapshot")

# --- Config toggle ---
RUN_PREPROCESS = True  # Set to true to run preprocessing
RUN_DISPLAY = True # Set to true to display clusters on mosaic
RUN_CLUSTER_JULIE = True  # Set to true to run clustering on Julie's dataset


# --- Rule all ---
rule all:
    input:
        # always required outputs
        f"{MODELS_DIR}/autoencoder.pth",
        f"{MODELS_DIR}/loss_curve.png",
        f"{MODELS_DIR}/reconstructions.png",
        # optional preprocessing
        *([f"data/processed/{AUTOENCODER_MODEL}/craters.npy", f"data/processed/{AUTOENCODER_MODEL}/metadata.csv"] if RUN_PREPROCESS else []),
        # optional cluster Julie's dataset
        *([f"{RESULTS_DIR}/clustering_dots_{TECHNIQUE}.png",
           f"{RESULTS_DIR}/clustering_imgs_{TECHNIQUE}.png"] if RUN_CLUSTER_JULIE else []),
        # optional display
        *([f"{RESULTS_DIR}/crater_clusters_{NUM_CLUSTERS}.csv"]) if RUN_DISPLAY else [],
        *([f"{RESULTS_DIR}/crater_clusters_{NUM_CLUSTERS}.png"]) if RUN_DISPLAY else [],
        f"{RUN_DIR}/rules.txt",
        f"{RUN_DIR}/summary.txt",
        f"{RUN_DIR}/dag.pdf"



rule preprocess_craters:
    input:
        map_file="data/raw/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
        craters_csv="data/raw/lunar_crater_database_robbins_2018.csv"
    output:
        output_dir=directory(f"data/processed/{AUTOENCODER_MODEL}"),
        np_output=f"data/processed/{AUTOENCODER_MODEL}/craters.npy",
        metadata_output=f"data/processed/{AUTOENCODER_MODEL}/metadata.csv"
    params:
        min_diameter=1.0,
        max_diameter=10.0,
        lat_min=-60,
        lat_max=60,
        offset=0.5,
        craters_to_output=-1,   # -1 for all craters
        dst_height=100,
        dst_width=100,
        autoencoder_model=AUTOENCODER_MODEL, # "cnn" or "mae"
        batch_size = 64,
        pretrained_model = PRETRAINED_MODEL
    shell:
        """
        PYTHONPATH=$(pwd) python src/data/preprocess.py \
            --map_file {input.map_file} \
            --craters_csv {input.craters_csv} \
            --output_dir {output.output_dir} \
            --np_output_path {output.np_output} \
            --info_output_path {output.metadata_output} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --latitude_bounds {params.lat_min} {params.lat_max} \
            --offset {params.offset} \
            --craters_to_output {params.craters_to_output} \
            --save_crops \
            --save_np_array \
            --dst_height {params.dst_height} \
            --dst_width {params.dst_width} \
            --autoencoder_model {params.autoencoder_model} \
            --batch_size {params.batch_size} \
            --pretrained_model {params.pretrained_model}
        """

rule train_autoencoder:
    input:
        npy=f"data/processed/{AUTOENCODER_MODEL}/craters.npy"
    output:
        model=f"{MODELS_DIR}/autoencoder.pth",
        loss=f"{MODELS_DIR}/loss_curve.png",
        latent=f"{MODELS_DIR}/latent_vectors.npy"
    params:
        autoencoder_model=AUTOENCODER_MODEL,
        epochs= EPOCHS,
        batch_size=400,
        latent_dim=LATENT_DIM,
        lr=1e-3,
        weight_decay=1e-5,
        lr_patience=3,
        min_lr=1e-8,
        lr_factor=0.5,
        num_samples = 50000,
        freeze_until= FREEZE_UNTIL,  # For MAE: number of encoder transformer blocks to freeze from the end (negative number)
        masked_ratio=0.75,  # Masking ratio for MAE training        
        pretrained_model=PRETRAINED_MODEL

    shell:
        """
        PYTHONPATH=$(pwd) python src/train/train.py \
            --autoencoder_model {params.autoencoder_model} \
            --input {input.npy} \
            --model_output {output.model} \
            --loss_plot {output.loss} \
            --latent_output {output.latent} \
            --epochs {params.epochs} \
            --batch_size {params.batch_size} \
            --latent_dim {params.latent_dim} \
            --lr {params.lr} \
            --lr_patience {params.lr_patience} \
            --min_lr {params.min_lr} \
            --lr_factor {params.lr_factor} \
            --num_samples {params.num_samples} \
            --weight_decay {params.weight_decay} \
            --freeze_until {params.freeze_until} \
            --mask_ratio {params.masked_ratio}
        """


rule reconstruct_craters:
    input:
        npy=f"data/processed/{AUTOENCODER_MODEL}/craters.npy",
        model=f"{MODELS_DIR}/autoencoder.pth"
    output:
        reconstructions=f"{MODELS_DIR}/reconstructions.png"
    params:
        autoencoder_model=AUTOENCODER_MODEL,
        device="cpu",
        num_images=8,
        latent_dim=LATENT_DIM,
        freeze_until = FREEZE_UNTIL  # For MAE: number of encoder transformer blocks to freeze from the end (negative number)
        
    shell:
        """
        PYTHONPATH=$(pwd) python src/train/reconstruct.py \
            --autoencoder_model {params.autoencoder_model} \
            --input {input.npy} \
            --model {input.model} \
            --device {params.device} \
            --file_outq {output.reconstructions} \
            --num_images {params.num_images} \
            --latent_dim {params.latent_dim} \
            --freeze_until {params.freeze_until}
        """


rule encode_latents:
    input:
        imgs_dir="data/raw/craters_for_danny",
        model=f"{MODELS_DIR}/autoencoder.pth"
    output:
        latents=f"{RESULTS_DIR}/latents_julie.npy",
        states=f"{RESULTS_DIR}/states_julie.npy"
    params:
        autoencoder_model=AUTOENCODER_MODEL,
        bottleneck=LATENT_DIM,
        freeze_until = FREEZE_UNTIL,  # For MAE: number of encoder transformer blocks to freeze from the end (negative number)
        pretrained_model=PRETRAINED_MODEL
    shell:
        """
        PYTHONPATH=$(pwd) python src/cluster/cluster.py encode \
            --imgs-dir {input.imgs_dir} \
            --model {input.model} \
            --bottleneck {params.bottleneck} \
            --out-latents {output.latents} \
            --out-states {output.states} \
            --freeze-until {params.freeze_until} \
            --autoencoder-model {AUTOENCODER_MODEL} \
            --pretrained-model {params.pretrained_model}
        """

rule plot_latent_dots:
    input:
        latents=f"{RESULTS_DIR}/latents_julie.npy",
        states=f"{RESULTS_DIR}/states_julie.npy"
    output:
        f"{RESULTS_DIR}/clustering_julie_dots_{TECHNIQUE}.png"
    params:
        technique = TECHNIQUE,
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
        latents=f"{RESULTS_DIR}/latents_julie.npy",
        imgs_dir="data/raw/craters_for_danny"
    output:
        f"{RESULTS_DIR}/clustering_julie_imgs_{TECHNIQUE}.png"
    params:
        technique = TECHNIQUE,
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
        model=f"{MODELS_DIR}/autoencoder.pth",
        dataset=f"data/processed/{AUTOENCODER_MODEL}/craters.npy",
        metadata=f"data/processed/{AUTOENCODER_MODEL}/metadata.csv",
    output:
        df=f"{RESULTS_DIR}/crater_clusters_{NUM_CLUSTERS}.csv",
        clustering_png=f"{RESULTS_DIR}/crater_clusters_{NUM_CLUSTERS}.png"
    params:
        num_clusters = NUM_CLUSTERS,
        autoencoder_model=AUTOENCODER_MODEL,
        batch_size = 400,
        device="cuda",  # or "cpu"
        freeze_until = FREEZE_UNTIL,  # For MAE: number of encoder transformer blocks to freeze from the end (negative number)
        latent_dim = LATENT_DIM, 
        cluster_method= CLUSTER_METHOD,  # "kmeans" or "gmm"
        technique = TECHNIQUE,
        latent_output = f"{RESULTS_DIR}/latents_all.npy"
        pretrained_model = PRETRAINED_MODEL
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
                --freeze_until {params.freeze_until}\
                --use_gpu \
                --pretrained_model {params.pretrained_model}
                """)
        else:
            print("Skipping display_clusters rule")


rule snapshot_workflow:
    output:
        dag=f"{RUN_DIR}/dag.pdf",
        summary=f"{RUN_DIR}/summary.txt",
        rules=f"{RUN_DIR}/rules.txt"
    shell:
        """
        snakemake --dag | dot -Tpdf > {output.dag}
        snakemake --summary > {output.summary}
        snakemake --list > {output.rules}
        """

