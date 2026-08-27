"""
legacy_run_overrides.py — hand-confirmed corrections for existing logs/ runs
whose own recorded metadata is wrong or missing, gathered directly from the
person who ran them (2026-08-16). Consulted by backfill_manifests.py on top
of model_meta.extract_model_params()'s automated result.

Most of what used to need a manual override here turned out to be a genuine
parser gap, not missing information - the confirmations below led to fixing
model_meta.py itself (cnn->cae family aliasing, PRETRAINED_MODEL/
FREEZE_UNTIL support, path-based data_tag inference), so those runs now
resolve correctly automatically. What's left here is only the two cases that
aren't fixable by parsing alone: a genuine historical mislabeling, and a
handful of runs with no recoverable signal on disk at all.
"""

# (path prefix, family_hint) - threaded into extract_model_params() so the
# mask_ratio/latent_dim field-inclusion logic (and the recorded `family`
# itself) are corrected consistently, not just patched after the fact.
FAMILY_HINTS = [
    # These 3 runs' own Snakefile.snapshot says AUTOENCODER_MODEL="cae", but
    # that was a mistake at training time - PRETRAINED_MODEL/FREEZE_UNTIL
    # (both present in the snapshot) are MAE-specific fine-tuning concepts
    # that don't apply to the plain conv autoencoder CAE actually trains.
    # Confirmed by the person who ran them: "the facebook under cae is a
    # mistake - that is supposed to be under mae."
    ("logs/cae/facebook/vit-mae-large/", "mae"),

    # "everything that has cnn in it, its actually the cae (i made the
    # mistake of writing cnn on some)" - covers every logs/cnn_latent*
    # directory uniformly, whether or not its own snapshot happens to record
    # AUTOENCODER_MODEL="cnn" (model_meta.py's cnn->cae alias already
    # handles that case; some of these older snapshots don't have the
    # variable at all, which this prefix rule also covers).
    ("logs/cnn_latent", "cae"),
]

# Explicit field overrides, keyed by exact run directory path - for runs
# whose Snakefile.snapshot predates AUTOENCODER_MODEL/EPOCHS/etc. entirely
# (an even older, unparameterized Snakefile version) and aren't covered by a
# FAMILY_HINTS prefix above.
FIELD_OVERRIDES = {
    # These 3 predate train_dino.py's --min_diameter/--max_diameter flags (or
    # were run manually without them), so nothing on disk records their
    # diameter range - DINOv2's own config.yaml never tracks it at all
    # (diameter filtering is a preprocessing-time concept, not part of
    # DINOv2's training config). Confirmed by the person who ran them: all
    # DINO runs use the same 3-10km range MAE/CAE do (rule train_dino passes
    # the Snakefile's MIN_DIAMETER/MAX_DIAMETER globals unchanged).
    "logs/dino/scratch/none/na/dino_smoke_test/drange_unknown/nunknown/na/ep2":
        {"diameter_range": [3.0, 10.0]},
    "logs/dino/scratch/none/na/dino_wide/drange_unknown/nunknown/na/ep2":
        {"diameter_range": [3.0, 10.0]},
    "logs/dino/scratch/none/na/dino_wide/drange_unknown/nunknown/na/ep150":
        {"diameter_range": [3.0, 10.0]},

    # data_tag corrections (2026-08-23 session) - these runs' Snakefile.snapshot
    # predates the DATA_TAG variable AND the sigma/{N} path convention
    # entirely (an even older `data/processed/{arch}/...` output layout - see
    # model_meta._infer_data_tag_from_snapshot_text's docstring), so nothing
    # parseable records what data they trained on. Confirmed by the person
    # who ran them, one at a time.
    "logs/cae/unknown/none/na/untagged/d1-10/nunknown/lat40/ep50":
        {"data_tag": "wac100m_raw"},  # raw Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif, no sigma filtering
    "logs/cae/unknown/none/na/untagged/d1-10/nunknown/lat40/ep10__cae_test_new_ssh_1.0_10.0_10_40":
        {"data_tag": "wac100m_sigma50"},  # highpass_filtered_lunar_mosaic_sigma_50.tif
    "logs/cae/unknown/none/na/untagged/d1-10/nunknown/lat40/ep10__cae_sigma50_1.0_10.0_10_40":
        {"data_tag": "wac100m_sigma50"},  # highpass_filtered_lunar_mosaic_sigma_50.tif
    "logs/cae/unknown/none/na/untagged/drange_unknown/nunknown/latunknown/epunknown":
        {"data_tag": "wac100m_raw",  # raw Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif, no sigma filtering
         "epochs": 50, "latent_dim": 10},
    "logs/cae/unknown/none/na/untagged/drange_unknown/nunknown/lat20/epunknown":
        {"data_tag": "wac100m_raw",  # raw Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif, no sigma filtering
         "epochs": 20, "diameter_range": [1.0, 10.0]},
    "logs/cae/unknown/none/na/untagged/drange_unknown/nunknown/lat40/epunknown__cnn_latent40_l2_sched_500":
        {"data_tag": "wac100m_raw",  # raw Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif, no sigma filtering
         "epochs": 500, "diameter_range": [3.0, 10.0]},
    "logs/cae/unknown/none/na/untagged/drange_unknown/nunknown/lat40/epunknown__cnn_latent40_l2_sched_pca":
        {"data_tag": "wac100m_raw",  # raw Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif, no sigma filtering
         "epochs": 50, "diameter_range": [3.0, 10.0]},

    # Remaining gaps confirmed 2026-08-23 - none of these are parseable from
    # any file on disk (no PRETRAINED_MODEL/PRETRAINED_WEIGHTS variable for
    # the two MAE runs' source; no Snakefile.snapshot at all for the CAE
    # run's diameter range).
    "logs/cae/unknown/none/na/untagged/drange_unknown/nunknown/latunknown/epunknown":
        {"diameter_range": [3.0, 10.0]},
    "logs/mae/unknown/none/na/untagged/drange_unknown/nunknown/maskunknown/ep500":
        # RUN_NAME comment ("mae_fr2_l2_1_10_500") hints at freeze_until=-2 -
        # confirmed: same facebook/vit-mae-large checkpoint and freeze depth
        # as the other freeze-2 runs above.
        {"source": "finetune", "pretrained_weights": "facebook/vit-mae-large", "freeze_until": -2},
    "logs/mae/unknown/none/na/wac100m_sigma100_noband_global/d3-10/n25000/mask0.75/ep500":
        {"source": "scratch"},

    # Orphan checkpoints (autoencoder.pth sitting directly in run_dir, no
    # models/ wrapper - a pre-convention layout discover_checkpoints() never
    # finds, so these were invisible to the dashboard entirely until the
    # migrate_logs.py dry-run's find_manifests() walk surfaced them,
    # 2026-08-23). No Snakefile.snapshot on disk for either - confirmed by
    # the person who ran them.
    "logs/mae/unknown/none/na/wac100m_sigma100/drange_unknown/nunknown/maskunknown/epunknown":
        {"mask_ratio": 0.75, "diameter_range": [3.0, 10.0], "epochs": 50},
    "logs/mae/unknown/none/na/wac100m_sigma100/drange_unknown/nunknown/maskunknown/epunknown__mae_bottleneck_64_processed_sigma_100":
        # "bottleneck_64" in the dirname = the retired MAE bottleneck
        # architecture variant (see mae_bottleneck.py), bottleneck_dim=64 -
        # reusing the "latent_dim" field for it since there's no separate
        # schema field for MAE's bottleneck dim and the concept is the same
        # (compressed representation size).
        {"mask_ratio": 0.75, "diameter_range": [3.0, 10.0], "epochs": 50, "latent_dim": 64},
}
