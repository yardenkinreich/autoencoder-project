#!/bin/bash
#SBATCH --job-name=regen-examples
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Regenerate every example figure to show CLOSEST + RANDOM members.
#
# Random members alone were uninformative here, and that is itself the finding
# rather than a sampling problem: silhouette sits at ~0.09 in every
# configuration (global 0.0932, per-band 0.083-0.093), so typical members of
# different clusters genuinely are near-indistinguishable. Showing the closest
# members alongside makes the sharper test available - if even the most
# central members of two clusters look alike, the clustering is not capturing
# anything visual at all.
#
# "Closest" is distance to the cluster's own MEAN IMAGE in the global profile
# (the same picture displayed beside it, so the label means what it says) and
# distance to the embedding centroid in the per-band figures.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

S=logs/dino/stock_pretrained/none/na/untagged/drange_unknown/nunknown/na/epunknown/data_comparison/geomap
F=logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10/data_comparison/geomap

echo "########## global cluster profiles"
for d in "$S/morph_sigma100__global_wide" "$S/emp_sigma100__global_wide" \
         "$S/morph_sigma100_div500__global_wide" \
         "$F/morph_sigma100__global_wide" "$F/emp_sigma100__global_wide"; do
  echo "=== $d"
  PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.cluster_profile --dir "$d" --overwrite
done

echo "########## per-band figures"
for v in morph_sigma100__global_wide emp_sigma100__global_wide morph_sigma100_div500__global_wide; do
  echo "=== $v"
  PYTHONPATH=$(pwd) pixi run -e default python -m src.eval.per_band_clustering --map-dir "$S/$v" --k 10
done
echo "EXIT=$?"
