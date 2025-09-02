import os
import datetime
import shutil

# --- Define the run name once ---
RUN_NAME = "cnn_latent40_l2_sched_pca"
# Or make it dynamic with timestamp
# RUN_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

RUN_DIR = f"logs/{RUN_NAME}"
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
RUN_PREPROCESS = False   # Set to true to run preprocessing
RUN_DISPLAY = True # Set to true to display clusters on mosaic


# --- Rule all ---
rule all:
    input:
        # always required outputs
        f"{MODELS_DIR}/autoencoder.pth",
        f"{MODELS_DIR}/loss_curve.png",
        f"{MODELS_DIR}/reconstructions.png",
        f"{RESULTS_DIR}/clustering_dots.png",
        f"{RESULTS_DIR}/clustering_imgs.png",
        f"{RESULTS_DIR}/latents.npy",
        f"{RESULTS_DIR}/states.npy",
        # optional preprocessing
        *(["data/processed/craters.npy", "data/processed/metadata.csv"] if RUN_PREPROCESS else []),
        # optional display
        *(["results/crater_clusters_on_mosaic.png"]) if RUN_DISPLAY else []


rule preprocess_craters:
    input:
        map_file="data/raw/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
        craters_csv="data/raw/lunar_crater_database_robbins_2018.csv"
    output:
        crops_dir=directory("data/processed"),
        np_output="data/processed/craters.npy",
        metadata_output="data/processed/metadata.csv"
    params:
        output_dir="data/processed",
        min_diameter=3.0,
        max_diameter=10.0,
        lat_min=-60,
        lat_max=60,
        offset=0.5,
        craters_to_output=-1,   # -1 for all craters
        dst_height=100,
        dst_width=100
    shell:
        """
        PYTHONPATH=$(pwd) python src/data/preprocess.py \
            --map_file {input.map_file} \
            --craters_csv {input.craters_csv} \
            --output_dir {params.output_dir} \
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
            --dst_width {params.dst_width}
        """

rule train_autoencoder:
    input:
        npy="data/processed/craters.npy"
    output:
        model=f"{MODELS_DIR}/autoencoder.pth",
        loss=f"{MODELS_DIR}/loss_curve.png",
        latent=f"{MODELS_DIR}/latent_vectors.npy"
    params:
        epochs= 50,
        batch_size=32,
        latent_dim=40,
        lr=1e-5,
        weight_decay=1e-5,
        lr_patience=5,
        min_lr=1e-8,
        lr_factor=0.5,
        num_samples = 50000
    shell:
        """
        PYTHONPATH=$(pwd) python src/train/train.py \
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
            --weight_decay {params.weight_decay}
        """


rule reconstruct_craters:
    input:
        npy="data/processed/craters.npy",
        model=f"{MODELS_DIR}/autoencoder.pth"
    output:
        reconstructions=f"{MODELS_DIR}/reconstructions.png"
    params:
        device="cpu",
        num_images=8,
        latent_dim=40
        
    shell:
        """
        PYTHONPATH=$(pwd) python src/train/reconstruct.py \
            --input {input.npy} \
            --model {input.model} \
            --device {params.device} \
            --file_outq {output.reconstructions} \
            --num_images {params.num_images} \
            --latent_dim {params.latent_dim}
        """


rule encode_latents:
    input:
        imgs_dir="data/raw/craters_for_danny",
        model=f"{MODELS_DIR}/autoencoder.pth"
    output:
        latents=f"{RESULTS_DIR}/latents.npy",
        states=f"{RESULTS_DIR}/states.npy"
    params:
        bottleneck=40
    shell:
        """
        PYTHONPATH=$(pwd) python src/cluster/cluster.py encode \
            --imgs-dir {input.imgs_dir} \
            --model {input.model} \
            --bottleneck {params.bottleneck} \
            --out-latents {output.latents} \
            --out-states {output.states}
        """

rule plot_latent_dots:
    input:
        latents=f"{RESULTS_DIR}/latents.npy",
        states=f"{RESULTS_DIR}/states.npy"
    output:
        f"{RESULTS_DIR}/clustering_dots.png"
    params:
        technique="pca",
        model_name = "CNN"
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
        latents=f"{RESULTS_DIR}/latents.npy",
        imgs_dir="data/raw/craters_for_danny"
    output:
        f"{RESULTS_DIR}/clustering_imgs.png"
    params:
        technique="pca",
        model_name = "CNN"
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
        dataset="data/processed/craters.npy",
        metadata="data/processed/metadata.csv",
    output:
        df=f"{RESULTS_DIR}/crater_clusters_kmeans.csv"
    params:
        num_clusters=5,
        batch_size=32,
        device="cuda",  # or "cpu"
        latent_dim=40
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
                --out_df {output.df}
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

