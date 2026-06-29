#!/bin/bash
#SBATCH --job-name=sv_train_calib
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite

# Train sn-calibration DeepLabv3 for field line segmentation.
# Single-GPU job: DeepLabv3-ResNet50 is ~40M params, fits easily on one L4.

cd /blue/ewhite/b.weinstein/src/soccer-video-analysis

DATA_DIR=/orange/ewhite/soccer-vision/sn-calib
PRETRAINED=weights/deeplabv3_sn.pth

# Download pretrained weights if not present
if [ ! -f "$PRETRAINED" ]; then
  mkdir -p weights
  echo "Downloading pretrained DeepLabv3 from Google Drive..."
  uv run python -c "
import gdown
gdown.download(id='1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM', output='$PRETRAINED')
"
fi

# Download SoccerNet calibration data if not present
if [ ! -d "$DATA_DIR/train" ]; then
  echo "Downloading SoccerNet calibration data..."
  uv run python training/sn_calib/train.py --download --data-dir "$DATA_DIR"
fi

uv run python training/sn_calib/train.py \
  --data-dir "$DATA_DIR" \
  --pretrained "$PRETRAINED" \
  --output-dir training/sn_calib/runs \
  --epochs 30 \
  --lr 1e-4 \
  --batch-size 8 \
  --num-workers 4 \
  --comet
