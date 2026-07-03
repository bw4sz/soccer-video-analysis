#!/bin/bash
#SBATCH --job-name=sv_train_spotting
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite

# Train action spotting (NetVLAD on Baidu features).
# Single-GPU: feature-based training is lightweight.

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

DATA_DIR=/blue/ewhite/b.weinstein/soccer-vision-data/soccernet

# Download data if not present
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "Downloading SoccerNet-v2 labels and Baidu features..."
  "$PY" training/sn_spotting/train_action_spotting.py \
    --download-only \
    --data-dir "$DATA_DIR"
fi

"$PY" training/sn_spotting/train_action_spotting.py \
  --data-dir "$DATA_DIR" \
  --output-dir /blue/ewhite/b.weinstein/soccer-vision-data/training/sn_spotting/runs \
  --features baidu \
  --epochs 50
