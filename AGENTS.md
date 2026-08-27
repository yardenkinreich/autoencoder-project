# AGENTS.md

Lunar crater autoencoder pipeline (CAE / MAE / DINOv2). Reproducible runs are driven by a **Snakemake** workflow; everything else (eval, dashboard) is standalone Python modules.

## Environment
- Managed by **pixi** (`pixi.toml` / `pixi.lock`, no conda/venv). `pixi.toml` has an *empty* `[tasks]`, so all commands are raw CLI.
- Do **not** rely on a bare `python` / `pixi run` alias. Use the env interpreter explicitly: `.pixi/envs/default/bin/python` (Python 3.11, GPU/CUDA deps). `.vscode/settings.json` points here.
- The code reads `src` via `sys.path` or `PYTHONPATH=$(pwd)`. Snakefile uses `PYTHONPATH=$(pwd)` — replicate this when running modules directly.

## Entrypoints / architecture
- `Snakefile` — edit module-level constants at the top to configure a run. There is **no `RUN_NAME`** anymore; the output dir is `RUN_DIR`, computed from a manifest-shaped dict via `src/run_layout.py:canonical_run_dir()`.
- Run dir schema: `logs/{family}/{source}/{structure}/{frozen}/{data_tag}/{d3-10}/{n<NUM_SAMPLES>}/{maskN|latN}/{epN}/`. Family ∈ `cae|mae|dino`; source ∈ `scratch|finetune`. Keep paths consistent with `run_layout.py` — it is the single source of truth shared by the Snakefile, eval, and dashboard migrations.
- Three model families, different pipelines:
  - **CAE** — dense autoencoder; `latent_dim` is the run metric.
  - **MAE** — vit-mae; runs keyed by `mask_ratio`; optional `PRETRAINED_WEIGHTS` (`source=finetune`, unfrozen later blocks).
  - **DINOv2** — *separate opt-in* `snakemake train_dino` (not in `rule all`). `train_dino.py` **requires `torchrun` even at 1 GPU**; its portable output is `{run}/eval/final/teacher_checkpoint.pth` (per-rank `model_final.rank_N.pth` is resume-only).
- Preprocessing: `src/data/preprocess_2.py`. In the Snakefile, `INPUT_DIR` and `DATA_TAG` **must be edited together** — the tag labels the data dir and groups runs; `DATA_TAG` is also used to build `RUN_DIR`. `MIN_DIAMETER`/`MAX_DIAMETER` etc. flow into both run dir and preprocessing.
- Raw data inputs (large, gitignored): `data/raw/wac_mosaic_new_version/sigma/100/highpass_filtered_lunar_mosaic.tif` + Robbins CSV + `configs/global_scaling.json`.

## Running
- Full/live pipeline: `snakemake --cores all` (Snakemake uses the pixi interpreter; it snapshots the Snakefile + writes `dag.pdf`, `summary.txt`, `rules.txt` into `RUN_DIR`).
- Targeted: `snakemake preprocess_craters`, `train_autoencoder`, `train_dino`, `display_clusters`, etc.
- A stale `.snakemake` lock (e.g. `Error: Directory cannot be locked`) means a prior run crashed — clear `.snakemake/` lock files.
- Eval one trained/stock checkpoint: `PYTHONPATH=src python -m eval.evaluate --checkpoint logs/.../models/autoencoder.pth --autoencoder-model mae --config configs/eval_suite.yaml`
- Dashboard: `PYTHONPATH=src streamlit run src/dashboard/app.py`

## Tests
- **Do not** run bare `pytest` collection — it imports heavy GPU/ML deps and hangs. Run specific files instead:
  - `PYTHONPATH=src python -m pytest src/test/test_history.py src/test/test_model_meta.py` (fast unit tests, pass locally)
- `src/test/test_processing.py` references a local macOS path that does not exist here — treat as non-runnable reference, don't "fix to pass" blindly.

## Gotchas
- `runs/` and `archive/` are legacy / relocated content; `logs/` is the canonical output root now.
- `configs/holdout_crater_ids.csv` is the held-out crater ID list; preprocessing and holdout-set rules both consume it (keep it in sync when the eval holdout changes).
- Checkpoint conventions for eval/dashboard discovery: MAE/CAE `{run}/models/autoencoder.pth`; DINO `{run}/eval/final/teacher_checkpoint.pth` (do not confuse `eval/final/` with eval *results* which live at `{run}/eval_results/<run_id>/`).
