#!/bin/bash
#SBATCH --job-name=sv_footpass_taad
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite

# Train the FOOTPASS TAAD baseline (player-centric ball-action spotting).
# See training/FOOTPASS.md. Prereqs:
#   1. HF access set up + data fetched:
#        hf auth login
#        python scripts/footpass_fetch_data.py            # -> $DATA_ROOT
#   2. A FOOTPASS env (py3.11, torch2.1, decord, albumentations, h5py).
#      Build a dedicated env/container; do NOT reuse the soccer-vision env (torch 2.12).
#      Set $FOOTPASS_PY below to that interpreter.

set -euo pipefail
REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
FOOTPASS="$REPO/vendor/FOOTPASS"
DATA_ROOT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass
FOOTPASS_PY=${FOOTPASS_PY:-/blue/ewhite/b.weinstein/envs/footpass/bin/python}

if [ ! -f "$DATA_ROOT/data/train_tactical_data.h5" ]; then
  echo "Missing $DATA_ROOT/data/train_tactical_data.h5 — run scripts/footpass_fetch_data.py first." >&2
  exit 1
fi

cd "$FOOTPASS"
# batch_size=6 OOM'd on an L4 (22GB) at epoch 3 (job 36260512): epochs 1-2 train
# head-only (X3D frozen per set_x3d_freezing_schedule), so autograd skips backbone
# activations; unfreezing at epoch 3 makes it retain them, spiking memory. Halved
# to 3 to fit the unfrozen backward pass — not a leak, so no other change needed.
"$FOOTPASS_PY" train_TAAD_Baseline.py \
  --data_root "$DATA_ROOT" \
  --run_path "$DATA_ROOT/runs/taad_$(date +%d%m%Y_%H%M)" \
  --epochs 20 \
  --batch_size "${BATCH_SIZE:-3}" \
  --num_workers "${SLURM_CPUS_PER_TASK:-6}"
