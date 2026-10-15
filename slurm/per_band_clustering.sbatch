#!/bin/bash
#SBATCH --job-name=per-band-clust
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/geomap
# Every k the app offers, not just k=10. This used to hardcode --k 10, which
# silently made every regeneration a no-op for the k=4 figures: they kept
# whatever markers, examples and (missing) PCA panels they were first built
# with, while the k=10 ones moved on. The app's k selector offers both, so
# both have to be rebuilt together or the two disagree about what the code says.
KS="${KS:-4 10}"
# EMP is deliberately NOT in the default list. It remains buildable -
# pass CATS/VARIANTS explicitly to include it - but the routine
# comparison is morphology vs its divisive-corrected version.
VARIANTS="${VARIANTS:-morph_sigma100__global_wide morph_sigma100_div500__global_wide}"
for v in $VARIANTS; do
  for k in $KS; do
    echo "=========== $v   k=$k"
    PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.per_band_clustering \
      --map-dir "$D/$v" --k "$k"
  done
done
echo "EXIT=$?"
