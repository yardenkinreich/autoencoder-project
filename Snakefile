
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
        craters_to_output=10000,
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
        model="models/conv_autoencoder_6.pth",
        loss="models/loss_curve_6.png",
        latent="models/latent_vectors_6.npy"
    params:
        epochs=50,
        batch_size=32,
        latent_dim=6,
        lr=1e-3
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
            --lr {params.lr}
        """

rule encode_latents:
    input:
        imgs_dir="data/raw/craters_for_danny",
        model="models/conv_autoencoder_6.pth"
    output:
        latents="results/latents_clustering_cnn_6.npy",
        states="results/states.npy"
    params:
        bottleneck=6
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
        latents="results/latents_clustering_cnn_6.npy",
        states="results/states.npy"
    output:
        "results/clustering_cnn_6_dots.png"
    params:
        technique="pca",
        model_name = "CNN_6"
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
        latents="results/latents_clustering_cnn_6.npy",
        imgs_dir="data/raw/craters_for_danny"
    output:
        "results/clustering_cnn_6_imgs.png"
    params:
        technique="pca",
        model_name = "CNN_6"
    shell:
        """
        PYTHONPATH=$(pwd) python src/cluster/cluster.py plot-imgs \
            --latents {input.latents} \
            --imgs-dir {input.imgs_dir} \
            --out-png {output} \
            --model-name {params.model_name} \
            --technique {params.technique}
        """
