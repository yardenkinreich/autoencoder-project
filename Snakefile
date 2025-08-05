# Snakefile
rule all:
    input:
        "results/train_val_loss.png",
        "results/test_metrics.json",
        "results/clustering/embeddings.png"
        "data/processed/craters_images.npy",
        "data/processed/craters_metadata.csv"

rule download_data:
    output:
        "data/raw/dataset.zip"
    script:
        "src/data/download.py"

rule preprocess_craters:
    input:
        map_file="data/raw/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
        craters_csv="data/raw/lunar_crater_database_robbins_2018.csv"
    output:
        crops_dir=directory("data/processed/crops"),
        np_output="data/processed/craters.npy",
        metadata_output="data/processed/metadata.csv"
     params:
        output_dir="data/processed/crops",
        min_diameter=3.0,
        max_diameter=10.0,
        lat_min=-60,
        lat_max=60,
        offset=0.5,
        craters_to_output=1000,
        dst_height=100,
        dst_width=100
    shell:
        """
        python scripts/process.py \
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

rule preprocess_data:
    input:
        "data/raw/dataset.zip"
    output:
        directory("data/processed/")
    script:
        "src/data/preprocess.py"

rule train_model:
    input:
        data="data/processed/"
    output:
        model="models/autoencoder.pt",
        plot="results/train_val_loss.png"
    script:
        "src/train/train.py"

rule evaluate_model:
    input:
        model="models/autoencoder.pt",
        data="data/processed/"
    output:
        "results/test_metrics.json"
    script:
        "src/test/evaluate.py"

rule cluster_embeddings:
    input:
        model="models/autoencoder.pt",
        data="data/processed/"
    output:
        "results/clustering/embeddings.png"
    script:
        "src/cluster/cluster.py"
