#!/bin/bash
#SBATCH --job-name=dino-finetune-noproto
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Same finetune as train_dino_finetune.sbatch (same config, same data, same
# architecture/LR/schedule) but with proto.enabled overridden to false via
# --opts, so this is a controlled A/B: everything held fixed except the
# prototypical-loss auxiliary supervision - the point is to see whether it's
# actually helping vs the same finetune without it.
#
# OUTPUT_DIR below deliberately differs from train_dino_finetune.sbatch's
# (adds the "_noproto" suffix) - canonical_run_dir()'s schema (src/run_layout.py)
# has no axis for the proto flag, so with an identical config/data_tag/epochs
# this run would otherwise land at the EXACT SAME directory as the proto run
# and overwrite its checkpoint. The suffix is purely a path-uniqueness escape
# hatch, not read by anything - the dashboard gets proto_loss_enabled from
# run_manifest.json's own field (param_source="run_manifest" always wins over
# any dirname parsing), so this being harmless to parse is already verified.
#
# Usage:
#   sbatch slurm/train_dino_finetune_noproto.sbatch
#   squeue -u $(whoami)                                      # check status
#   tail -f logs/slurm-dino-finetune-noproto-<jobid>.out      # follow progress

set -euo pipefail
# See train_dino_finetune.sbatch's comment - sbatch runs a copied script from
# a spool dir, so $0 doesn't point back to slurm/ in the repo.
cd "$SLURM_SUBMIT_DIR"

OUTPUT_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10_noproto"
if [ -e "$OUTPUT_DIR/run_manifest.json" ]; then
    echo "Refusing to run: $OUTPUT_DIR already has a completed run (run_manifest.json exists)." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# See train_dino_finetune.sbatch's comment - DINOv2's distributed.enable()
# misreads a bare sbatch (no srun) allocation as an srun-launched process
# via SLURM_JOB_ID's mere presence, and looks for SLURM_NTASKS/SLURM_PROCID
# (never set here) instead of torchrun's own RANK/WORLD_SIZE/etc.
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=1 \
    src/train/train_dino.py \
    --input data/processed_wac_100m_new/sigma/100/dino_wide/craters_wide.dat \
    --output_dir "$OUTPUT_DIR" \
    --config_file configs/dino_craters_finetune.yaml \
    --pretrained_weights data/raw/pretrained/dinov2_vits14_pretrain.pth \
    --min_diameter 3.0 --max_diameter 10.0 --data_tag wac100m_sigma100_noband_wide \
    --opts proto.enabled=false
