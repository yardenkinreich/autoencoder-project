#!/bin/bash
#SBATCH --job-name=kaguya-register
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Per-tile Kaguya-minus-Robbins offsets from crater rim-symmetry centres
# (1-5 km craters, up to 2000/tile), half-split validated, -> tile_offsets.csv
# consumed by build_catalog --register robbins. See src/data/kaguya_register.py.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.kaguya_register \
  --out_dir data/raw/kaguya_tco_registration --workers 32 "$@"
echo "EXIT=$?"
