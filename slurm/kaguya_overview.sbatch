#!/bin/bash
#SBATCH --job-name=kaguya-overview
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# One pass over all 4,800 Kaguya TCO tiles -> ~253 m/px overview (log
# reflectance, valid / DN0 / DN1 fractions) + per-tile QA table + quick-look.
# Needed by --albedo logratio and by the smoke figures. See
# src/data/kaguya_overview.py.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
PYTHONPATH=$(pwd) pixi run -e default python -m src.data.kaguya_overview \
  --out_dir data/raw/kaguya_tco_overview --workers 32
echo "EXIT=$?"
