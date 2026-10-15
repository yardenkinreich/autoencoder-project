# DINO/MAE finetuning + prototypical-loss status

Running record of the finetuning/prototypical-loss investigation (started from
"the training I did did not improve the latent separation afterwards"), kept
so this survives independently of any one chat session. Update this file
rather than losing context when priorities shift.

## TL;DR as of 2026-08-27

- **Frozen, off-the-shelf DINOv2 (`dino_pretrained`, tag `off_the_shelf`) is
  still the best-performing model overall.** Neither DINO finetune variant
  has beaten it yet.
- **Root cause identified for why finetuning underperforms**: both DINO
  finetunes (plain and prototypical-loss) collapse toward using raw
  **brightness** as the dominant learned feature (`eta_sq` ~0.85-0.88 on
  Julie's set, vs ~0 for the frozen model on the same set - see "Brightness
  collapse" below). The prototypical loss does NOT fix this - proto and
  no-proto finetunes have statistically identical brightness confound
  severity (0.875 vs 0.879), it just rides on top of an already-collapsed
  embedding.
- Two 1000-epoch **from-scratch MAE** runs (plain vs. prototypical-loss,
  jobs 149/150, 2 GPUs each via DDP) are training now to see whether the
  same collapse pattern shows up outside DINO / outside finetuning.
- A menu of anti-collapse options was discussed (harder freezing, L2-SP
  regularization toward init, adapters/LoRA, slower teacher EMA, stronger
  photometric augmentation, confound-based early stopping) but **none have
  been implemented yet** - this is the open decision point when picking
  this back up.

## What exists now (infrastructure, all committed to git as of 5691b6f8)

- **`src/train/prototypical_loss.py`** - architecture-agnostic prototypical-
  network auxiliary loss (Snell et al. 2017). Class prototype = mean of
  support embeddings, no separate classifier weights, so it doesn't cap the
  model to a fixed class count. Draws episodes from
  `configs/new_test_set_split.csv`'s `train_pool` half only (321 of 642
  craters, stratified by class); the `final_test` half (321 craters) is
  never touched by any training this module drives, by construction - the
  genuine held-out test set for whatever gets trained.
  - Wired into **DINO** via `configs/dino_craters_finetune.yaml`'s `proto:`
    section + `ssl_meta_arch.py`'s `forward_backward()` (single combined
    `loss_accumulator`, one `backprop_loss()` call - required for
    correctness under FSDP). `loss_weight: 0.2` there (~1-2% of DINO's own
    total loss, since DINO's total sits around 18-20).
  - Wired into **MAE** via `train.py --proto_enabled` (new CLI flags:
    `--proto_loss_weight`, `--proto_n_support_per_class`,
    `--proto_dat_path`, `--proto_metadata_csv`, `--proto_split_csv`,
    `--proto_class_names`, `--proto_seed`). Default weight is **0.05**, NOT
    DINO's 0.2 - MAE's reconstruction loss sits on a much smaller scale
    (~1-3) than DINO's combined total, so the same 0.2 gave proto_loss
    ~7-12% of the total loss in a smoke test; 0.05 lands in the same ~1-3%
    "gentle nudge" range DINO's weight gives.
  - Both integrations write `proto_loss_enabled`/`proto_loss_weight`/
    `proto_n_support_per_class`/`proto_split_csv` into `run_manifest.json`,
    surfaced in the dashboard (Run Deep Dive's "More details", Compare
    Runs' param columns).
- **DINO finetuning from the stock checkpoint**
  (`configs/dino_craters_finetune.yaml`) - architecture matches the stock
  ImageNet checkpoint exactly (`vit_small`, `patch_size=14`, `in_chans=3`,
  vs. the from-scratch config's `patch_size=8`/`in_chans=1`), with
  pos_embed bicubic-interpolated at load time
  (`_resize_pretrained_pos_embed` in `ssl_meta_arch.py`). Root cause of the
  *original* "training didn't help" complaint: every prior DINO run was
  actually from-scratch (never loaded pretrained weights) with an
  architecture that couldn't have loaded the stock checkpoint even if it
  tried - not a finetuning failure, a finetuning-never-happened bug.
- **`src/eval/head.py`** - frozen-backbone + lightweight-head comparison
  (raw KMeans+alignment / LDA head / kNN, all under proper out-of-fold CV,
  distinct from `align.py`'s in-sample "alignability" framing). kNN has
  been the single best-performing approach found so far in absolute terms
  (0.767 acc on Julie / 0.807 on new_set_final_test, frozen stock DINO) -
  worth remembering as the standing fallback if finetuning doesn't pan out.
- **Unsupervised over-clustering exploration** (`src/eval/visualize.py`'s
  `plot_unsupervised_cluster_dots`/`_images`, wired into every
  `evaluate.py` run automatically) - KMeans at k=10/k=50 (no true labels
  used), PCA+UMAP dot plots and crater-crop-image plots. Crop zoom is
  0.18 by default, overridable per labeled-set via `eval_suite.yaml`'s
  `unsupervised_crop_zoom` (julie uses 0.28 - its point cloud is smaller,
  150 vs. 321 craters, so 0.18 read too small there).
- **Leak-safe eval set**: `eval_suite.yaml`'s `new_set_final_test` (321
  craters, the `final_test` half of `new_test_set_split.csv`) - the common
  ground for comparing a proto-loss checkpoint (which trained on the OTHER
  half) against a plain finetune/stock checkpoint without leakage.
  `evaluate.py`'s `main()` auto-skips the full, leaky `new_set` for any
  checkpoint whose manifest says `proto_loss_enabled: true`.
- **Full Streamlit dashboard rebuild** (`src/dashboard/`) - Compare Runs,
  Run Deep Dive, Data Sources pages; run registry/manifest backfill;
  leaderboard (`eval.history.best_runs()`) and Run Deep Dive's registry
  both dedupe to the LATEST eval per checkpoint (explicit
  `sort_values("timestamp").drop_duplicates("checkpoint", keep="last")` in
  both now - Run Deep Dive's was hardened to be explicit rather than
  relying on CSV append order). Home page shows unique-checkpoint count,
  not raw eval-run count.
- **SLURM infrastructure** (`slurm/`) - this cluster SIGKILLs unscheduled
  GPU processes outside a job allocation, so every real training/eval run
  goes through `sbatch`, not a bare background process. Also: `sbatch`
  copies the submitted script into a spool dir before running it, so
  `cd "$(dirname "$0")/.."` doesn't work inside a script - use
  `cd "$SLURM_SUBMIT_DIR"` instead. And DINOv2's own
  `distributed.enable()` misreads a bare `sbatch` (no `srun`) allocation
  as `srun`-launched via `SLURM_JOB_ID`'s mere presence - `unset
  SLURM_JOB_ID` before `torchrun` fixes it.

## Brightness collapse - the numbers

`confound_correlation.csv` (Kruskal-Wallis `eta_sq` of the raw cluster
assignment vs. brightness; >=0.14 = "large effect" by this project's own
threshold):

| Set | Stock (frozen) | No-proto finetune | Proto finetune |
|---|---|---|---|
| Julie (150 craters) | -0.001 (p=0.39, no effect) | 0.875 | 0.879 |
| new_set_final_test (321 craters) | 0.356 | 0.849 | 0.845 |

Headline metrics tell the same story - both finetunes underperform frozen
stock on nearly every metric, worst on Julie (QWK collapses to ~0, no
better than random ordinal agreement):

| Set | Stock | No-proto finetune | Proto finetune |
|---|---|---|---|
| Julie: acc / macro_f1 / QWK | 0.493 / 0.483 / 0.462 | 0.367 / 0.310 / -0.009 | 0.387 / 0.326 / 0.009 |
| new_set_final_test: acc / macro_f1 / QWK | 0.636 / 0.573 / 0.747 | 0.586 / 0.543 / 0.729 | 0.592 / 0.548 / 0.733 |

Proto beats no-proto on every metric, consistently but by a small margin -
it's a real, reproducible improvement, just not enough to close the gap
with frozen stock, and it doesn't touch the underlying brightness collapse.

**Likely mechanism**: `src/data/dino_craters_augmentation.py`'s photometric
jitter (`RandomBrightness(p=0.2)`, `RandomContrast(p=0.2)`) - the `p=`
naming is misleading, it's actually the std-dev of a `Normal(1.0, p)`
brightness-scaling factor, ALWAYS applied (no probability gate despite the
name). At std=0.2 this is mild relative to real between-crater brightness
variance, so DINO's self-distillation objective ("same crater -> similar
embedding") can satisfy itself using raw brightness as a free, globally-
consistent shortcut, without learning genuine degradation semantics.
Deliberately NOT trying to eliminate brightness as a signal entirely - the
augmentation module's own docstring argues absolute brightness/contrast
correlates with genuine degradation (fresh = high local contrast, degraded
= washed-out), so per-crop normalization was intentionally skipped. Target
if this gets revisited: bring brightness `eta_sq` down from ~0.85 to
something like 0.14-0.35 (present, on par with what frozen stock itself
shows), not toward zero.

## Options discussed for constraining how much the model is allowed to
## change (not yet implemented - this is the open decision point)

Ordered roughly by implementation cost:

1. **Freeze harder** - `layerwise_decay`/`patch_embed_lr_mult` currently
   soft-slow early layers; could hard-freeze (`requires_grad=False`) all
   but the last 1-2 transformer blocks. Cheapest, config-only.
2. **Freeze the whole backbone, only let the head adapt** - most
   conservative version of (1); zero risk of backbone drift, probably the
   smallest possible improvement over frozen-as-is if any.
3. **L2-SP-style explicit regularization** - add `lambda * sum((theta -
   theta_0)**2)` to the loss, pulling parameters back toward their
   pretrained values. Most literal match to "constrain how much it
   changes" - a single tunable lambda directly bounding drift, rather than
   an indirect hyperparameter like LR. Moderate new code (snapshot theta_0,
   add the penalty in ssl_meta_arch.py's loss accumulation, same pattern as
   the proto-loss hook).
4. **Adapter/LoRA-style finetuning** - freeze the entire backbone, insert
   small trainable low-rank matrices. Backbone's original computation
   provably stays close to unchanged. Most principled, most implementation
   work (not currently in this DINOv2 fork at all).
5. **Slow the teacher's EMA momentum further** - the deployed checkpoint is
   the EMA teacher, not the student; pushing momentum closer to 1.0 damps
   how fast the actually-evaluated representation can drift.
6. **Monitor confound eta_sq during training, stop on collapse onset**
   - operational safeguard rather than an architectural constraint; caps
   drift by outcome instead of by a hyperparameter guess.

Also flagged as a separate, complementary (not alternative) lever:
**strengthening the photometric jitter itself** (std 0.2 -> ~0.4 as a first
step) so brightness stops being a free shortcut in the first place - this
was the plan before the "constrain how much it changes" angle came up, and
is not mutually exclusive with any of the six options above. A natural
first experiment combines a harder freeze (option 1) with the augmentation
fix, checked against `confound_correlation.csv` on a short (~100-200 epoch)
run before committing more GPU budget.

## Known open bug (unrelated to any of the above, not yet fixed)

Evaluating the **stock MAE checkpoint** (`mae_pretrained`,
`src/models/mae/pretrain_mae_vit_base_full.pth`) crashes with `ValueError:
Input X contains NaN` inside the first KMeans call - the embeddings
themselves contain NaN. Likely one degenerate crater image poisoning
`encode_stock_pretrained_mae()`'s cross-batch mean/std standardization
(`src/models/mae_pretrained.py`), since that standardization is computed
across the WHOLE batch, so one bad sample's NaN can propagate to every
sample's value for that feature dimension. Never investigated further -
flagged twice, deferred both times. Blocks a clean stock-vs-scratch-vs-proto
MAE comparison.

Separately, unrelated: the historical MAE finetune pipeline
(`--pretrained_weights src/models/mae/pretrain_mae_vit_base_full.pth`) now
crashes on a `patch_size` mismatch (checkpoint is patch16, current
`train.py`'s hardcoded MAE definition is patch8) - discovered while
smoke-testing MAE proto loss, worked around by testing from-scratch
instead. Not fixed. Blocks re-finetuning MAE from that stock checkpoint
until either the checkpoint or the model definition's patch_size is
reconciled.

## Currently running (check before assuming these finished)

- **Job 149** (`mae-scratch-1000ep`): plain MAE from scratch, 1000 epochs,
  50k samples, 2 GPUs (DDP), diameter 3-10, `wac100m_sigma100_noband` data
  tag. Output: `logs/mae/scratch/none/na/wac100m_sigma100_noband/d3-10/
  n50000/mask0.75/ep1000/`.
- **Job 150** (`mae-scratch-proto-1000ep`): same, with `--proto_enabled
  --proto_loss_weight 0.05`. Output: `.../ep1000_proto/`.
  Both resumable (`checkpoint_last.pth`, `checkpoint_every=20`), ~4-4.5h
  estimated (measured ~30s/epoch single-GPU, GPU-saturated at every batch
  size tested - see `slurm/diagnose_mae_speed.sbatch`'s diagnostic).
