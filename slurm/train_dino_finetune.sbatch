#!/bin/bash
#SBATCH --job-name=dino-finetune
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# DINOv2 finetune from the stock pretrained checkpoint - submitted as a
# real SLURM job rather than a raw background process, since this cluster
# kills unscheduled GPU processes outside a job allocation (confirmed: the
# exact same run died at the same point whether launched from Claude Code's
# session or directly in a terminal - not a code bug, a scheduler policy).
#
# Usage:
#   sbatch slurm/train_dino_finetune.sbatch
#   squeue -u $(whoami)                            # check status
#   tail -f logs/slurm-dino-finetune-<jobid>.out    # follow progress

set -euo pipefail
# NOT "cd $(dirname "$0")/.." - sbatch copies this script into a spool dir
# before running it, so $0 doesn't point back to slurm/ in the repo (job 130
# hit this: it cd'd into the spool tree and "mkdir -p logs/..." failed with
# Permission denied there). SLURM sets SLURM_SUBMIT_DIR to the actual
# directory `sbatch` was invoked from - use that instead.
cd "$SLURM_SUBMIT_DIR"

OUTPUT_DIR="logs/dino/finetune/pretrained_dinov2_vits14_pretrain/unknown/wac100m_sigma100_noband_wide/d3-10/nunknown/na/ep10"
mkdir -p "$OUTPUT_DIR"

# DINOv2's own distributed.enable() treats SLURM_JOB_ID's mere presence as
# "this is an srun-launched process" and reads SLURM_NTASKS/SLURM_PROCID/etc
# instead of torchrun's RANK/WORLD_SIZE/etc (see dinov2/distributed/__init__.py
# _is_slurm_job_process()) - true for srun, not for sbatch+torchrun as used
# here, so job 132 died on a missing SLURM_NTASKS. Unset it so torchrun's own
# env vars are used, matching every non-SLURM smoke test that already worked.
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=1 \
    src/train/train_dino.py \
    --input data/processed_wac_100m_new/sigma/100/dino_wide/craters_wide.dat \
    --output_dir "$OUTPUT_DIR" \
    --config_file configs/dino_craters_finetune.yaml \
    --pretrained_weights data/raw/pretrained/dinov2_vits14_pretrain.pth \
    --min_diameter 3.0 --max_diameter 10.0 --data_tag wac100m_sigma100_noband_wide
