"""
Unit tests for dashboard/model_meta.py against real paths already on disk.
Run with: PYTHONPATH=src python -m pytest src/test/test_model_meta.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

from dashboard.model_meta import extract_model_params


def _p(rel: str) -> str:
    return os.path.join(REPO_ROOT, rel)


def test_stock_dino_pretrained():
    r = extract_model_params(_p("data/raw/pretrained/dinov2_vits14_pretrain.pth"), "dino_pretrained")
    assert r["source"] == "stock_pretrained"
    assert r["param_source"] == "stock"


def test_stock_mae_pretrained():
    r = extract_model_params(_p("src/models/mae/pretrain_mae_vit_base_full.pth"), "mae_pretrained")
    assert r["source"] == "stock_pretrained"
    assert r["param_source"] == "stock"


def test_dino_run_with_config_yaml():
    # path is post-migration (logs/ was reorganized by src/dashboard/migrate_logs.py) -
    # this run's config.yaml still lives alongside the checkpoint at its new location
    ckpt = _p("logs/dino/scratch/none/na/wac100m_sigma100/d3-10/nunknown/na/ep150"
             "/20260810-182429/eval/final/teacher_checkpoint.pth")
    r = extract_model_params(ckpt, "dino")
    assert r["param_source"] == "dino_config"
    assert r["family"] == "dino"
    assert r["epochs"] == 150
    assert r["source"] == "scratch"  # pretrained_weights is '' in this run's config.yaml
    # dataset_path in config.yaml is
    # "Craters:root=data/processed_wac_100m_new/sigma/100/dino_wide/craters_wide.dat" -
    # data_tag must resolve to the same "wac100m_sigma100" source MAE/CAE
    # runs sharing that mosaic/filter get, not the DINO-specific wrapper
    # directory name ("dino_wide") - the sigma/{N} source is what actually
    # makes two runs comparable, and this run's own run_manifest.json (see
    # extract_model_params()'s precedence) has been repointed to match.
    assert r["data_tag"] == "wac100m_sigma100"
    assert "/" not in r["data_tag"] and ":" not in r["data_tag"]
    # DINO always crops at a wider FOV (crop radius = 1x diameter) than
    # MAE/CAE (0.5x) - confirmed by the person who ran these (2026-08-23),
    # never swept, so this is fixed per-family, not a per-run guess.
    assert r["fov"] == 1.0


def test_mae_run_with_snakefile_snapshot():
    # This run has both a Snakefile.snapshot AND a run_manifest.json (the
    # manifest wins per extract_model_params()'s precedence - see
    # test_manifest_takes_precedence_over_a_stale_snapshot), but the
    # manifest's own param_source is "snakefile_snapshot" (backfill_manifests.py
    # froze exactly what parsing the snapshot produced), so this still
    # exercises the same real-world values a from-scratch snapshot parse
    # would - just via the manifest short-circuit rather than a live parse.
    ckpt = _p("logs/mae/scratch/none/na/wac100m_sigma100/d3-10/n50000/mask0.75/"
             "ep150/20260505-173914/models/autoencoder.pth")
    r = extract_model_params(ckpt, "mae")
    assert r["param_source"] == "snakefile_snapshot"
    assert r["family"] == "mae"
    assert r["epochs"] == 150
    assert r["mask_ratio"] == 0.75
    assert r["diameter_range"] == [3.0, 10.0]
    assert r["source"] == "scratch"
    assert r["fov"] == 0.5


def test_mae_run_without_snapshot_falls_back_to_dirname(tmp_path):
    """No run_manifest.json, no Snakefile.snapshot, no config.yaml anywhere -
    the only signal left is the dirname's own encoded
    d{min}-{max}_ep{epochs}_mask{ratio}_n{n} tokens.

    Synthetic fixture (no real on-disk run currently exercises this exact
    fallback path - the original fixture directory was removed as part of
    routine cleanup, same situation as
    test_cae_never_reports_pretrained_from_a_stray_snapshot_variable above)."""
    run_dir = tmp_path / "wac100m_sigma100_noband_global" / "d3-10_ep1500_mask0.75_n25000"
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "models" / "autoencoder.pth"
    r = extract_model_params(str(ckpt), "mae")
    assert r["param_source"] == "dirname_regex"
    assert r["epochs"] == 1500
    assert r["mask_ratio"] == 0.75
    assert r["num_samples"] == 25000
    assert r["family"] == "mae"  # preserved from the path hint even though the regex can't recover epochs' siblings


def test_legacy_pretrained_model_and_freeze_until():
    """Older Snakefile.snapshot era: PRETRAINED_MODEL (a HF model id) +
    FREEZE_UNTIL (partial fine-tuning), instead of today's
    PRETRAINED_WEIGHTS - confirmed with the person who ran these."""
    # path is post-migration - this run now lives under its corrected
    # mae/finetune/.../freeze-8/ location
    ckpt = _p("logs/mae/finetune/facebook_vit-mae-large/freeze-8/wac100m_raw"
             "/d1-10/n50000/mask0.75/ep100/20251208-182953/models/autoencoder.pth")
    r = extract_model_params(ckpt, "mae")
    assert r["param_source"] == "snakefile_snapshot"
    assert r["source"] == "finetune"
    assert r["pretrained_weights"] == "facebook/vit-mae-large"
    assert r["freeze_until"] == -8
    assert r["epochs"] == 100
    assert r["mask_ratio"] == 0.75


def test_cae_never_reports_pretrained_from_a_stray_snapshot_variable(tmp_path):
    """A genuine CAE run's Snakefile.snapshot can still have a leftover
    PRETRAINED_MODEL variable (copy-pasted template) that CAE's training
    code never actually consumes (the Snakefile only ever wires pretrained
    init into MAE) - must not be reported as finetune.

    Synthetic fixture (no real on-disk run currently exercises this exact
    edge case - the original fixture directory was removed as part of
    routine cleanup): a CAE snapshot with a stray PRETRAINED_MODEL left over
    from a copy-pasted MAE template."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "Snakefile.snapshot").write_text(
        'AUTOENCODER_MODEL = "cae"\nEPOCHS = 25\nLATENT_DIM = 40\n'
        'PRETRAINED_MODEL = "facebook/vit-mae-large"\n'
    )
    ckpt = run_dir / "models" / "autoencoder.pth"

    r = extract_model_params(str(ckpt), "cae")
    assert r["family"] == "cae"
    # cae is unconditionally "scratch" (never "unknown") - the Snakefile has
    # never wired a pretrained-init path for it, so this isn't a guess.
    assert r["source"] == "scratch"
    assert r["pretrained_weights"] is None


def test_cae_never_reports_freeze_until_from_a_stray_snapshot_variable(tmp_path):
    """Same bug shape as the PRETRAINED_MODEL case above, found via a
    dashboard audit (2026-08-23): a real CAE run's Snakefile.snapshot had
    FREEZE_UNTIL=0 sitting around (the variable's own comment literally says
    "For MAE"), and extract_model_params() reported it as a real
    freeze_until=0 for a CAE run - CAE has never had a freeze-partial-layers
    mechanism, so this was a meaningless leftover masquerading as a fact."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "Snakefile.snapshot").write_text(
        'AUTOENCODER_MODEL = "cae"\nEPOCHS = 50\nLATENT_DIM = 40\nFREEZE_UNTIL = 0\n'
    )
    ckpt = run_dir / "models" / "autoencoder.pth"

    r = extract_model_params(str(ckpt), "cae")
    assert r["family"] == "cae"
    assert r["freeze_until"] is None


def test_family_hint_corrects_a_mislabeled_run():
    """logs/cae/facebook/vit-mae-large/* runs are recorded as AUTOENCODER_
    MODEL="cae" in their own snapshot, but were actually MAE fine-tunes -
    family_hint overrides this, and mask_ratio/latent_dim gating follows the
    corrected family, not the file's original (wrong) one."""
    # path is post-migration - relocated under mae/finetune/... with its own
    # checkpoint-mtime datetime as the leaf (see migrate_logs.py/run_layout.py)
    ckpt = _p("logs/mae/finetune/facebook_vit-mae-large/freeze0/wac100m_raw/d1-10"
             "/n50000/mask0.75/ep50/20251228-193839/models/autoencoder.pth")
    r = extract_model_params(ckpt, "cae", family_hint="mae")
    assert r["family"] == "mae"
    assert r["mask_ratio"] == 0.75
    assert r["latent_dim"] is None
    assert r["source"] == "finetune"
    assert r["pretrained_weights"] == "facebook/vit-mae-large"


def test_manifest_takes_precedence_over_a_stale_snapshot(tmp_path):
    """A run_manifest.json in the run dir must win over Snakefile.snapshot
    even when the two disagree - the manifest was written from the in-memory
    config at save time, so it's authoritative; the snapshot might be stale
    (e.g. hand-edited after the fact, or from a differently-configured re-run
    that reused the same directory)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "Snakefile.snapshot").write_text(
        'AUTOENCODER_MODEL = "mae"\nEPOCHS = 999\nMASK_RATIO = 0.5\n'
    )
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1, "family": "mae", "source": "scratch",
        "pretrained_weights": None, "epochs": 50, "mask_ratio": 0.75,
        "latent_dim": None, "num_samples": 25000, "diameter_range": [3, 10],
        "data_tag": "wac100m_sigma100_noband", "param_source": "run_manifest",
        "checkpoint_path": "models/autoencoder.pth", "written_by": "train.py",
    }))
    ckpt = run_dir / "models" / "autoencoder.pth"

    r = extract_model_params(str(ckpt), "mae")
    assert r["param_source"] == "run_manifest"
    assert r["epochs"] == 50  # manifest's value, not the stale snapshot's 999
    assert r["mask_ratio"] == 0.75
    assert r["num_samples"] == 25000
    assert r["freeze_until"] is None  # not in the manifest schema - defaults, doesn't error
    assert r["fov"] is None  # ditto - an even older manifest schema, predates "fov" too
    assert set(r.keys()) == _EXPECTED_KEYS


_EXPECTED_KEYS = {
    "family", "source", "pretrained_weights", "epochs", "mask_ratio",
    "latent_dim", "num_samples", "diameter_range", "data_tag",
    "freeze_until", "param_source", "iterations", "fov",
}


def test_every_source_returns_the_same_schema(tmp_path):
    """Every extract_model_params() branch must return the same key set -
    callers (the dashboard pages) index every key unconditionally, so a
    parser that omits one (e.g. parse_dino_config once forgot mask_ratio)
    crashes any page that renders a checkpoint from that branch. The last
    two cases (pure dirname_regex fallback, no manifest/snapshot/config at
    all) are synthetic - see test_mae_run_without_snapshot_falls_back_to_dirname."""
    dirname_run = tmp_path / "wac100m_sigma100_noband" / "d3-10_ep50_mask0.75_n25000"
    dirname_run.mkdir(parents=True)
    dirname_run_global = tmp_path / "wac100m_sigma100_noband_global" / "d3-10_ep1500_mask0.75_n25000"
    dirname_run_global.mkdir(parents=True)

    cases = [
        (_p("data/raw/pretrained/dinov2_vits14_pretrain.pth"), "dino_pretrained"),
        (_p("src/models/mae/pretrain_mae_vit_base_full.pth"), "mae_pretrained"),
        (_p("logs/dino/scratch/none/na/wac100m_sigma100/d3-10/nunknown/na/ep150"
           "/20260810-182429/eval/final/teacher_checkpoint.pth"), "dino"),
        (str(dirname_run / "models" / "autoencoder.pth"), "mae"),
        (str(dirname_run_global / "models" / "autoencoder.pth"), "mae"),
    ]
    for ckpt, arch in cases:
        r = extract_model_params(ckpt, arch)
        assert set(r.keys()) == _EXPECTED_KEYS, f"{ckpt} ({arch}): {sorted(r.keys())}"
    assert r["source"] == "unknown"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
