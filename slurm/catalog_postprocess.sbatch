#!/bin/bash
#SBATCH --job-name=catalog-postprocess
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Everything that has to happen AFTER a catalog's pixels exist, in the one
# order that is correct:
#
#   1. fit norm_stats.json on each catalog's own 100k-crater pixel sample
#   2. write loader.json  (the policy a model actually loads with)
#   3. re-run the processing comparison + dashboard figures
#
# Step 1 before step 2 because loader.json names a normalization and
# CraterCatalog refuses to construct if the constants for it are absent -
# which is the intended behaviour, not an inconvenience to work around.
#
# WHY norm_stats IS FITTED PER CATALOG AND NOT ONCE
# -------------------------------------------------
# The constants must come from the population a model will actually train on.
# Measured on morph_raw, the 642 labeled craters and the 100k catalog disagree
# substantially - skew 3.53 vs 0.708, std/(1.4826*MAD) 1.86 vs 1.16 - because
# the labeled set was chosen for visual labeling and is therefore enriched in
# large, fresh, bright-rimmed craters. Fitting on the labeled subset would set
# the scale from a biased sample and, worse, would let test data define the
# scaling. Each catalog fits its own.
#
# The testset/ subdirectory deliberately does NOT get its own constants: it
# inherits the parent's via loader.json's "stats_from", so train and test share
# one scaling.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID
export OMP_NUM_THREADS=4

# Override with e.g. CATALOG_LIST="kaguya_tco_raw kaguya_tco_ratio" SKIP_COMPARE=1
# (processing_compare in step 4 is the WAC six-catalog comparison).
read -ra CATALOGS <<< "${CATALOG_LIST:-morph_raw morph_sigma100 morph_sigma200 emp_raw emp_sigma100 emp_sigma200}"

# --- 1. normalization constants ---------------------------------------------
PRESENT=()
for c in "${CATALOGS[@]}"; do
  if [ -f "data/catalogs/$c/index.csv" ]; then PRESENT+=("data/catalogs/$c"); fi
done
echo "catalogs with an index: ${PRESENT[*]}"
if [ ${#PRESENT[@]} -eq 0 ]; then echo "nothing to do" >&2; exit 1; fi

# ALWAYS refit, never keep an existing file. The constants are a function of
# the catalog's pixels, so a norm_stats.json left over from a previous build of
# the same directory is silently wrong - and wrong in the least visible way
# possible, since the stale numbers still look entirely plausible.
for cat in "${PRESENT[@]}"; do
  # NORM_MAX_PX: per-crater pixel cap for native-resolution catalogs (Kaguya:
  # 65536); unset = every pixel, the WAC catalogs' unchanged behaviour.
  PYTHONPATH=$(pwd) pixi run -e default python -m src.data.norm_stats --catalog "$cat" \
    ${NORM_MAX_PX:+--max-px-per-crater $NORM_MAX_PX}
done

# --- 2. loader policy -------------------------------------------------------
# asinh for the raw (unfiltered) mosaics, robust for the high-pass ones. The
# split follows the skew: raw reflectance is bounded below by zero and
# unbounded above so it keeps a bright tail no affine map removes, while a
# high-pass residual has already had that tail subtracted away by
# construction. See src/data/norm_stats.py.
for cat in "${PRESENT[@]}"; do
  case "$cat" in
    *_raw) NORM=asinh ;;
    *)     NORM=robust ;;
  esac
  PYTHONPATH=$(pwd) pixi run -e default python - "$cat" "$NORM" <<'PY'
import json, os, sys
cat, norm = sys.argv[1], sys.argv[2]
policy = {
    "norm": norm,
    # No resize baked in: crops are stored at native size and the model's
    # input size belongs to the model's config, not the data's.
    "resize": None,
    "stats_from": None,          # None -> this catalog's own norm_stats.json
    "diam_min": None, "diam_max": None, "lat_abs_max": None,
    "num_channels": 1,
    "_why": (f"norm={norm}: asinh for raw reflectance (right-skewed, needs a "
             "monotone tail compression), robust for high-pass residuals (tail "
             "already removed by the filter). Neither clips. See "
             "src/data/norm_stats.py and the dashboard's Data Processing page."),
}
json.dump(policy, open(os.path.join(cat, "loader.json"), "w"), indent=2)
print(f"  {cat}/loader.json  norm={norm}")

# The test set borrows the parent's constants rather than fitting its own, so
# that train and test are on one scale. Anything else is a quiet leak.
ts = os.path.join(cat, "testset")
if os.path.isdir(ts):
    p = dict(policy); p["stats_from"] = cat
    p["_why"] += " testset inherits the parent catalog's constants (stats_from)."
    json.dump(p, open(os.path.join(ts, "loader.json"), "w"), indent=2)
    print(f"  {ts}/loader.json  norm={norm}  stats_from={cat}")
PY
done

# --- 3. verify a catalog actually loads -------------------------------------
# A loader.json that does not construct is worse than none at all, so prove it
# before the figures are built on top of it.
for cat in "${PRESENT[@]}"; do
  PYTHONPATH=$(pwd) pixi run -e default python - "$cat" <<'PY'
import sys, numpy as np
from src.data.crater_catalog_dataset import CraterCatalog
cat = sys.argv[1]
ds = CraterCatalog(root=cat)
x = ds[0][0]
a = np.asarray(x)
print(f"  OK {cat}: n={len(ds)} norm={ds.policy['norm']} "
      f"shape={tuple(a.shape)} range=[{a.min():.3f}, {a.max():.3f}]")
assert np.isfinite(a).all(), f"{cat}: non-finite pixels out of the loader"
PY
done

# --- 4. figures + dashboard data --------------------------------------------
if [ "${SKIP_COMPARE:-0}" != "1" ]; then
  PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.processing_compare --no-cache
fi
echo "EXIT=$?"
