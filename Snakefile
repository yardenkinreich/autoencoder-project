# Snakefile
rule all:
    input:
        "results/train_val_loss.png",
        "results/test_metrics.json",
        "results/clustering/embeddings.png"

rule download_data:
    output:
        "data/raw/dataset.zip"
    script:
        "src/data/download.py"

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
