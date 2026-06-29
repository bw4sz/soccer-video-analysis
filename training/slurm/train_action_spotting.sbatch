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

cd /blue/ewhite/b.weinstein/src/soccer-video-analysis

DATA_DIR=/orange/ewhite/soccer-vision/soccernet

# Download data if not present
if [ ! -d "$DATA_DIR/train" ]; then
  echo "Downloading SoccerNet-v2 labels and Baidu features..."
  uv run python training/sn_spotting/train_action_spotting.py \
    --download-only \
    --data-dir "$DATA_DIR"
fi

uv run python training/sn_spotting/train_action_spotting.py \
  --data-dir "$DATA_DIR" \
  --output-dir training/sn_spotting/runs \
  --features baidu \
  --epochs 50
