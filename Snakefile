rule all:
    input:
        "results/train_val_loss.png",
        "results/test_metrics.json",
        "results/clustering/embeddings.png",
        "data/processed/craters_images.npy",
        "data/processed/craters_metadata.csv"

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
        model="models/conv_autoencoder.pth",
        loss="models/loss_curve.png",
        latent="models/latent_vectors.npy"
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

rule evaluate_model:
    input:
        model="models/conv_autoencoder.pth",
        data="data/processed/craters.npy"
    output:
        "results/test_metrics.json"
    script:
        "src/test/evaluate.py"

rule cluster_embeddings:
    input:
        latent_vectors="models/latent_vectors.npy",      # saved from training
        metadata_csv="data/processed/metadata.csv",      # contains crater IDs
        images_dir="data/processed"                       # for plotting images
    output:
        latent_output="results/clustering/latent_with_labels.npy",
        dot_plot="results/clustering/embeddings_dots.png",
        image_plot="results/clustering/embeddings_images.png"
    params:
        n_clusters=3                                      # number of clusters
    shell:
        """
        PYTHONPATH=$(pwd) python src/cluster/cluster.py \
            --latent_input {input.latent_vectors} \
            --metadata_csv {input.metadata_csv} \
            --images_dir {input.images_dir} \
            --latent_output {output.latent_output} \
            --dot_plot {output.dot_plot} \
            --image_plot {output.image_plot} \
            --n_clusters {params.n_clusters}
        """