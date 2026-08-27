import os
import sys
import datetime
import shutil

sys.path.insert(0, "src")
from run_layout import canonical_run_dir


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
NUM_SAMPLES       = 25000       # how much data to actually train on, out of everything in INPUT_DIR
PRETRAINED_WEIGHTS = ""         # path to a .pth checkpoint to init from; "" trains from scratch
NUM_GPUS          = 1           # >1 trains train_autoencoder via torchrun (single node) instead of plain python

# --- Preprocessed data source ---
# Keep INPUT_DIR and DATA_TAG together: DATA_TAG is a short label for INPUT_DIR,
# used to group runs in logs/. If you point INPUT_DIR at different data, update
# DATA_TAG in the same edit so runs on different data never share a folder.
INPUT_DIR = "data/processed_wac_100m_new/sigma/100/test_rotate/without_left_band"
DATA_TAG  = "wac100m_sigma100_noband"

# Random-unlabeled holdout set (configs/holdout_crater_ids.csv) - materialized
# by `snakemake preprocess_holdout_set` but NOT wired into
# preprocess_craters/preprocess_craters_dino's --exclude_crater_ids or
# eval_suite.yaml anymore - the reviewed new-test-set craters (see
# NEW_TEST_SET_IDS below) are the only held-out set training now excludes.
# Rule/files kept (not deleted) in case this is wanted again later.
HOLDOUT_DIR = "data/processed_wac_100m_new/sigma/100/holdout"

# Second held-out test set: configs/correct_crater_with_labels_final.csv
# (notebooks/review_new_test_set.ipynb's output) - a REVIEWED/LABELED set
# (v-fr/r-fr/r-dr/v-dr degree classes), unlike holdout_crater_ids.csv's
# unlabeled random sample. Excluded from training the same way (see
# preprocess_craters/preprocess_craters_dino below); materialized separately
# by `snakemake preprocess_new_test_set` since it needs its own diameter
# bounds (~1-4.2km - this set skews smaller than MIN_DIAMETER/MAX_DIAMETER's
# 3-10km training range) rather than reusing the training config's.
NEW_TEST_SET_IDS = "configs/correct_crater_with_labels_final.csv"
NEW_TEST_SET_DIR = "data/processed_wac_100m_new/sigma/100/new_test_set"
# NEW_TEST_SET_DIR_WIDE (DINO-FOV variant) is defined alongside the
# preprocess_new_test_set rule below, next to NEW_TEST_SET_OFFSETS.

# --- Run naming ---
# RUN_DIR is the same canonical logs/{family}/{source}/{structure}/{frozen}/
# {data_source}/{crater_range}/{num_samples}/{other_metric}/{epochs}/ path
# every other tool in this project computes from a run_manifest.json
# (see src/run_layout.py) - built here from a manifest-shaped dict of these
# same Snakefile variables, so a fresh training run lands exactly where its
# own manifest (written by train.py at the end of the run) says it belongs,
# not at some other ad hoc path. Runs that share every one of these fields
# share a folder, so reruns with identical settings resume/overwrite instead
# of fragmenting into new ones - matches the old RUN_NAME convention's intent.
if AUTOENCODER_MODEL not in ("cae", "mae"):
    raise ValueError("Unsupported AUTOENCODER_MODEL. Choose 'cae' or 'mae'.")

_source = "finetune" if (AUTOENCODER_MODEL == "mae" and PRETRAINED_WEIGHTS) else "scratch"
RUN_DIR = canonical_run_dir({
    "family": AUTOENCODER_MODEL,
    "source": _source,
    "pretrained_weights": PRETRAINED_WEIGHTS if _source == "finetune" else None,
    "epochs": EPOCHS,
    "mask_ratio": MASK_RATIO if AUTOENCODER_MODEL == "mae" else None,
    "latent_dim": LATENT_DIM if AUTOENCODER_MODEL == "cae" else None,
    "num_samples": NUM_SAMPLES,
    "diameter_range": [MIN_DIAMETER, MAX_DIAMETER],
    "data_tag": DATA_TAG,
    "freeze_until": None,
})

MODELS_DIR  = f"{RUN_DIR}/models"
RESULTS_DIR = f"{RUN_DIR}/results"
TEST_DIR    = f"{RUN_DIR}/tests/edge_cases"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_DIR,    exist_ok=True)

if os.path.exists("Snakefile"):
    shutil.copy("Snakefile", f"{RUN_DIR}/Snakefile.snapshot")

# --- Config toggles ---
RUN_DISPLAY       = True
RUN_CLUSTER_JULIE = True

# torchrun launch prefix for train_autoencoder — see src/train/train.py header
# for single-vs-multi-GPU launch details. NUM_GPUS=1 (default) behaves exactly
# as before.
TRAIN_LAUNCHER = (f"torchrun --standalone --nproc_per_node={NUM_GPUS}"
                  if NUM_GPUS > 1 else "python")

# --- DINOv2 (separate opt-in pipeline: `snakemake train_dino`) ---
# DINO isn't part of `rule all` — it's a different training paradigm
# (self-distillation, not reconstruction) with its own checkpoint format;
# see src/train/train_dino.py and configs/dino_craters.yaml.
DINO_GPUS         = 1     # DINOv2's do_train() always needs torchrun, even at 1 (see train_dino.py).
                            # Multi-GPU verified working for real (2-GPU smoke test) - eval/final/
                            # teacher_checkpoint.pth is correctly consolidated across ranks regardless
                            # of DINO_GPUS; model_final.rank_N.pth (resume checkpoint) is per-rank by
                            # design and needs the same GPU count to resume from.
DINO_EPOCHS       = 100
DINO_CLEAN_OFFSET = 1.0   # ~2x diameter FOV (vs MAE's 0.5) — see preprocess_2.py --clean_offset

DINO_INPUT_DIR = f"{os.path.dirname(os.path.dirname(INPUT_DIR))}/dino_wide"
# Same canonical-path computation as RUN_DIR above.
DINO_RUN_DIR = canonical_run_dir({
    "family": "dino",
    "source": "scratch",
    "pretrained_weights": None,
    "epochs": DINO_EPOCHS,
    "mask_ratio": None,
    "latent_dim": None,
    "num_samples": None,
    "diameter_range": [MIN_DIAMETER, MAX_DIAMETER],
    "data_tag": f"{DATA_TAG}_wide",
    "freeze_until": None,
})

# --- DINOv2 finetune (opt-in: `snakemake train_dino_finetune`) ---
# Warm-starts from the stock ImageNet-pretrained checkpoint instead of
# training from scratch - see configs/dino_craters_finetune.yaml for why
# every prior DINO run (including ones nominally called a "finetune") was
# actually from-scratch, and why this needed its own config rather than
# an --opts override of dino_craters.yaml (patch_size/in_chans must match
# the checkpoint exactly, not just add a pretrained_weights path on top of
# the from-scratch architecture).
DINO_FINETUNE_EPOCHS = 10
DINO_FINETUNE_PRETRAINED = "data/raw/pretrained/dinov2_vits14_pretrain.pth"
DINO_FINETUNE_RUN_DIR = canonical_run_dir({
    "family": "dino",
    "source": "finetune",
    "pretrained_weights": DINO_FINETUNE_PRETRAINED,
    "epochs": DINO_FINETUNE_EPOCHS,
    "mask_ratio": None,
    "latent_dim": None,
    "num_samples": None,
    "diameter_range": [MIN_DIAMETER, MAX_DIAMETER],
    "data_tag": f"{DATA_TAG}_wide",
    "freeze_until": None,
})

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
        craters_csv = "data/raw/lunar_crater_database_robbins_2018.csv",
        scaling_json = "configs/global_scaling.json",
        new_test_set_ids = NEW_TEST_SET_IDS
    output:
        output_dir_clean = directory(f"{INPUT_DIR}/crater_crops_clean"),
        output_dir_aug   = directory(f"{INPUT_DIR}/crater_crops_aug"),
        np_output_clean  = f"{INPUT_DIR}/craters_clean.dat",
        np_output_aug    = f"{INPUT_DIR}/craters_aug.dat",
        metadata_output  = f"{INPUT_DIR}/metadata.csv"
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
            --exclude_lon_bounds 180 270 \
            --exclude_crater_ids {input.new_test_set_ids} \
            --norm_mode global \
            --scaling_json {input.scaling_json}

        """


rule preprocess_holdout_set:
    # Materializes configs/holdout_crater_ids.csv's actual image data (it's
    # only an ID list until preprocessed) at the same FOV/offset as INPUT_DIR,
    # for MAE/CAE reconstruction-loss and embedding testing. No aug/rotation
    # branch - this is a test set, not something to train on.
    input:
        map_file    = "data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif",
        craters_csv = "data/raw/lunar_crater_database_robbins_2018.csv",
        scaling_json = "configs/global_scaling.json",
        holdout_ids  = "configs/holdout_crater_ids.csv"
    output:
        output_dir_clean = directory(f"{HOLDOUT_DIR}/crater_crops"),
        np_output_clean  = f"{HOLDOUT_DIR}/craters.dat",
        metadata_output  = f"{HOLDOUT_DIR}/metadata.csv"
    params:
        min_diameter = MIN_DIAMETER,
        max_diameter = MAX_DIAMETER,
        lat_min      = -60,
        lat_max      = 60
    shell:
        """
        PYTHONPATH=$(pwd) python src/data/preprocess_2.py \
            --map_file {input.map_file} \
            --craters_csv {input.craters_csv} \
            --output_dir_clean {output.output_dir_clean} \
            --np_output_path_clean {output.np_output_clean} \
            --info_output_path {output.metadata_output} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --latitude_bounds {params.lat_min} {params.lat_max} \
            --craters_to_output -1 \
            --only_crater_ids {input.holdout_ids} \
            --save_raw_crops \
            --save_np_array \
            --autoencoder_model mae \
            --exclude_lon_bounds 180 270 \
            --norm_mode global \
            --scaling_json {input.scaling_json}
        """


# Suffix -> clean_offset (crop radius as a multiple of crater diameter):
# "" is preprocess_new_test_set's original MAE/CAE-matching 0.5x default;
# "_wide" is DINO's ~1.0x framing (DINO_CLEAN_OFFSET) - needed so the
# planned prototypical-loss auxiliary training (labeled supervision mixed
# into DINO's own training loop) sees craters framed the same way DINO's
# own unlabeled training data (dino_wide) is, not a narrower crop that
# would inject an extra, unintended distribution shift into the one signal
# meant to help. One parameterized rule (preprocess_2.py already exposes
# --clean_offset) rather than two near-duplicate rule blocks.
NEW_TEST_SET_OFFSETS = {"": 0.5, "_wide": DINO_CLEAN_OFFSET}
NEW_TEST_SET_DIR_WIDE = f"{NEW_TEST_SET_DIR}_wide"


rule preprocess_new_test_set:
    # Materializes NEW_TEST_SET_IDS (configs/correct_crater_with_labels_final.csv,
    # notebooks/review_new_test_set.ipynb's output) - same shape as
    # preprocess_holdout_set, but a SEPARATE labeled set (v-fr/r-fr/r-dr/
    # v-dr degree classes) rather than the random unlabeled holdout.
    # Diameter bounds are widened to this set's own range (~1-4.2km) rather
    # than reusing MIN_DIAMETER/MAX_DIAMETER - training keeps its own
    # 3-10km range untouched; this deliberately tests generalization to
    # smaller craters than the model trained on. No --exclude_lon_bounds
    # here (unlike the training rules) - that drops the mosaic-seam band
    # for training-data QUALITY reasons, but this is a fixed, manually-
    # reviewed test set; silently dropping craters to a training heuristic
    # isn't appropriate for a test set.
    #
    # {fov} wildcard picks the crop offset - see NEW_TEST_SET_OFFSETS above.
    #
    # latitude_bounds stays -60/60, UNLIKE the longitude band above - this
    # isn't a policy filter, it's the mosaic's actual data coverage limit:
    # highpass_filtered_lunar_mosaic.tif's raster bounds are exactly
    # +/-1819401.0254489686m, which is +/-60.0000 deg for this equirect.
    # projection's 1737400m lunar radius (y = R * lat_rad). Craters beyond
    # that have no source pixels at all - not a subset we're choosing to
    # skip, a subset that physically isn't in this raster. ~128 of the 770
    # reviewed craters fall outside it and can't be materialized from this
    # mosaic; -90/90 was tried and crashes (cv2 tries to resize a
    # zero-size window read from outside the raster).
    input:
        map_file    = "data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif",
        craters_csv = "data/raw/lunar_crater_database_robbins_2018.csv",
        scaling_json = "configs/global_scaling.json",
        test_ids     = NEW_TEST_SET_IDS
    output:
        output_dir_clean = directory("data/processed_wac_100m_new/sigma/100/new_test_set{fov}/crater_crops"),
        np_output_clean  = "data/processed_wac_100m_new/sigma/100/new_test_set{fov}/craters.dat",
        metadata_output  = "data/processed_wac_100m_new/sigma/100/new_test_set{fov}/metadata.csv"
    wildcard_constraints:
        fov = "|_wide"
    params:
        min_diameter = 1.0,
        max_diameter = 5.0,
        lat_min      = -60,
        lat_max      = 60,
        clean_offset = lambda wildcards: NEW_TEST_SET_OFFSETS[wildcards.fov]
    shell:
        """
        PYTHONPATH=$(pwd) python src/data/preprocess_2.py \
            --map_file {input.map_file} \
            --craters_csv {input.craters_csv} \
            --output_dir_clean {output.output_dir_clean} \
            --np_output_path_clean {output.np_output_clean} \
            --info_output_path {output.metadata_output} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --latitude_bounds {params.lat_min} {params.lat_max} \
            --craters_to_output -1 \
            --only_crater_ids {input.test_ids} \
            --clean_offset {params.clean_offset} \
            --save_raw_crops \
            --save_np_array \
            --autoencoder_model mae \
            --norm_mode global \
            --scaling_json {input.scaling_json}
        """


# --- Main training ---
rule train_autoencoder:
    input:
        data = f"{INPUT_DIR}/craters_aug.dat"
    output:
        model     = f"{MODELS_DIR}/autoencoder.pth",
        loss_plot = f"{MODELS_DIR}/loss_curve.png"
    params:
        launcher          = TRAIN_LAUNCHER,
        autoencoder_model = AUTOENCODER_MODEL,
        epochs            = EPOCHS,
        batch_size        = 128,   # per GPU when NUM_GPUS > 1
        lr                = 1e-3,
        weight_decay      = 1e-5,
        min_lr            = 1e-8,
        val_split         = 0.2,
        mask_ratio        = MASK_RATIO,
        latent_dim        = LATENT_DIM,
        pretrained        = PRETRAINED_WEIGHTS if AUTOENCODER_MODEL == "mae" else "",
        num_samples       = NUM_SAMPLES,
        run_dir           = RUN_DIR,
        min_diameter      = MIN_DIAMETER,
        max_diameter      = MAX_DIAMETER,
        data_tag          = DATA_TAG
    shell:
        """
        PYTHONPATH=$(pwd) {params.launcher} src/train/train.py \
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
            --run_dir {params.run_dir} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --data_tag {params.data_tag} \
            $([ -n "{params.pretrained}" ] && echo "--pretrained_weights {params.pretrained}")
        """


# --- DINOv2 (opt-in: `snakemake train_dino`) ---

rule preprocess_craters_dino:
    input:
        map_file    = "data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif",
        craters_csv = "data/raw/lunar_crater_database_robbins_2018.csv",
        scaling_json = "configs/global_scaling.json",
        new_test_set_ids = NEW_TEST_SET_IDS
    output:
        output_dir_clean = directory(f"{DINO_INPUT_DIR}/crater_crops"),
        np_output_clean  = f"{DINO_INPUT_DIR}/craters_wide.dat",
        metadata_output  = f"{DINO_INPUT_DIR}/metadata.csv"
    params:
        min_diameter      = MIN_DIAMETER,
        max_diameter      = MAX_DIAMETER,
        lat_min           = -60,
        lat_max           = 60,
        craters_to_output = -1,
        clean_offset      = DINO_CLEAN_OFFSET
    shell:
        """
        PYTHONPATH=$(pwd) python src/data/preprocess_2.py \
            --map_file {input.map_file} \
            --craters_csv {input.craters_csv} \
            --output_dir_clean {output.output_dir_clean} \
            --np_output_path_clean {output.np_output_clean} \
            --info_output_path {output.metadata_output} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --latitude_bounds {params.lat_min} {params.lat_max} \
            --craters_to_output {params.craters_to_output} \
            --clean_offset {params.clean_offset} \
            --save_raw_crops \
            --save_np_array \
            --autoencoder_model mae \
            --exclude_lon_bounds 180 270 \
            --exclude_crater_ids {input.new_test_set_ids} \
            --norm_mode global \
            --scaling_json {input.scaling_json}
        """


rule train_dino:
    # train_dino.py saves two things: model_final.rank_N.pth (FSDPCheckpointer,
    # LOCAL_STATE_DICT - for resuming training on the same GPU topology, not
    # portable) and eval/final/teacher_checkpoint.pth (do_test(), a clean
    # portable checkpoint - see src/models/dino_backbone.py). Track the
    # latter, since that's what downstream encode/cluster actually loads.
    input:
        data = f"{DINO_INPUT_DIR}/craters_wide.dat"
    output:
        model = f"{DINO_RUN_DIR}/eval/final/teacher_checkpoint.pth"
    params:
        output_dir   = DINO_RUN_DIR,
        epochs       = DINO_EPOCHS,
        num_gpus     = DINO_GPUS,
        min_diameter = MIN_DIAMETER,
        max_diameter = MAX_DIAMETER,
        data_tag     = f"{DATA_TAG}_wide"   # matches DINO_RUN_DIR - DINO trains on a
                                             # genuinely different (wider-FOV) data export
                                             # than MAE/CAE, not just a differently-named copy
    shell:
        """
        PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node={params.num_gpus} \
            src/train/train_dino.py \
            --input {input.data} \
            --output_dir {params.output_dir} \
            --epochs {params.epochs} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --data_tag {params.data_tag}
        """


rule train_dino_finetune:
    # Warm-starts from the stock ImageNet-pretrained checkpoint - see
    # configs/dino_craters_finetune.yaml for why this needs its own config
    # rather than an --opts override on top of dino_craters.yaml.
    input:
        data = f"{DINO_INPUT_DIR}/craters_wide.dat"
    output:
        model = f"{DINO_FINETUNE_RUN_DIR}/eval/final/teacher_checkpoint.pth"
    params:
        output_dir   = DINO_FINETUNE_RUN_DIR,
        config_file  = "configs/dino_craters_finetune.yaml",
        epochs       = DINO_FINETUNE_EPOCHS,
        num_gpus     = DINO_GPUS,
        pretrained   = DINO_FINETUNE_PRETRAINED,
        min_diameter = MIN_DIAMETER,
        max_diameter = MAX_DIAMETER,
        data_tag     = f"{DATA_TAG}_wide"
    shell:
        """
        PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node={params.num_gpus} \
            src/train/train_dino.py \
            --input {input.data} \
            --output_dir {params.output_dir} \
            --config_file {params.config_file} \
            --epochs {params.epochs} \
            --pretrained_weights {params.pretrained} \
            --min_diameter {params.min_diameter} \
            --max_diameter {params.max_diameter} \
            --data_tag {params.data_tag}
        """


# --- Reconstruction ---
rule reconstruct_craters:
    input:
        npy   = f"{INPUT_DIR}/craters_aug.dat",
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
        dataset  = f"{INPUT_DIR}/craters_aug.dat",
        metadata = f"{INPUT_DIR}/metadata.csv"
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
