#!/bin/bash
#SBATCH --job-name=catalog-eval-suite
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm-%x-%j-%a.out

# Score one catalog's 5 normalizations with the PROJECT'S OWN eval suite.
# Submit as an array 0-5, one task per catalog.
#
# WHY AN ARRAY AND NOT ONE 30-SET JOB
# -----------------------------------
# The first attempt was a single job covering all 30 sets. It crashed on set
# 20 of 30, and because evaluate.py writes summary.json only at the very end,
# ALL 19 completed results were lost - they existed nowhere but the log. One
# task per catalog means a failure costs one catalog, the six run in parallel
# (~20 min instead of ~1.5 h), and each writes its own summary.json.
#
# WHY NOT THE PROBE (src/eval/catalog_probe.py)
# ---------------------------------------------
# A supervised probe asks "is the label linearly decodable from the
# embedding" - easier than, and different from, the question this project is
# about: does the representation ORGANIZE itself by degradation without
# supervision. On the same data and encoder the probe read accuracy 0.759 /
# QWK 0.833 where this suite reads 0.530 / 0.658, and it cannot report mAP,
# ARI/NMI, silhouette or the confound correlations at all.
#
# The decisive columns for the data question are:
#   separation.silhouette_true      - do the true classes separate at all
#   map                             - mean average precision
#   ari / nmi                       - Hungarian-aligned cluster agreement
#   confound_correlation.lon / lat  - IS THE LATENT SPACE ORGANIZED BY GEOGRAPHY
#   confound_correlation.brightness / sobel_mean / frac_saturated
#   geo_correlation.pc*_vs_lon/lat
#
# All 30 sets hold the SAME 642 craters, so any difference is a difference in
# data presentation and nothing else.
#
# THREADS ARE PINNED TO 1 ON PURPOSE. src/eval/head.py's k-NN has twice hit a
# garbage-index IndexError from sklearn's tree-query backend under a
# multi-CPU allocation - a threading race, not a logic bug. That call now
# forces algorithm="brute" (no tree traversal at all, and faster for 384-dim
# embeddings anyway), and these env vars remove the remaining exposure.
#
# --history is a SEPARATE file: these are data-presentation experiments
# sharing one frozen checkpoint, not model runs, and writing them into
# logs/eval_history.csv would let them compete for a checkpoint's slot in the
# model dashboard's registry.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

CATS=(morph_raw morph_sigma100 morph_sigma200 emp_raw emp_sigma100 emp_sigma200)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"
NORMS="global global_wide zscore robust asinh"
CFG="configs/eval_suite_catalog_${CAT}.yaml"

echo "=== catalog: $CAT ==="
# Idempotent: rebuilds this catalog's eval sets only if missing, and always
# (re)writes the per-catalog config.
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.catalog_to_evalset \
  --catalogs "data/catalogs/$CAT" --norms $NORMS \
  --out-root data/evalsets --config-out "$CFG"

echo
echo "=== eval suite: $CAT x {$NORMS} ==="
PYTHONPATH=src:$(pwd) pixi run -e default python -m eval.evaluate \
  --checkpoint data/raw/pretrained/dinov2_vits14_pretrain.pth \
  --autoencoder-model dino_pretrained \
  --config "$CFG" \
  --out "logs/data_profiles/catalog_eval_suite/$CAT" \
  --history logs/catalog_eval_history.csv \
  --tag "catalog_data_comparison_${CAT}" \
  --n-boot 2000
echo "EXIT=$?"
