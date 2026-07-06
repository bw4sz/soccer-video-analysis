#!/bin/bash
#SBATCH --job-name=sv_footpass_ours_maskoff
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite

# Mask-OFF counterpart of the live smoke (job 36450376). Same window, same model,
# but --field-mask none, so we can compare TAAD predictions with vs without the
# turf gate on identical frames. Outputs to *_nomask paths.
set -euo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
VENV_PY="$REPO/.venv/bin/python"
FOOTPASS_PY=/blue/ewhite/b.weinstein/envs/footpass/bin/python
VIDEO="$REPO/data/match-saints-16b-pre-mls-next-2026-04-26.mp4"
CKPT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/runs/taad_03072026_1113/checkpoints/best_model.pt
OUTROOT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/ours
H5="$OUTROOT/our_saints_live_nomask.h5"
KEY=our_saints_live_nomask
START=34466
NFRAMES=600

mkdir -p "$OUTROOT"

echo "[1/2] extract tracklets WITHOUT field mask"
cd "$REPO"
"$VENV_PY" scripts/footpass_extract_tracklets.py \
  --video "$VIDEO" --start-frame "$START" --num-frames "$NFRAMES" --stride 1 \
  --game-key "$KEY" --out-h5 "$H5" --device cuda --conf 0.3 \
  --field-mask none

echo "[2/2] run TAAD + visualize (no mask)"
"$FOOTPASS_PY" scripts/footpass_infer_ours.py \
  --h5 "$H5" --game-key "$KEY" --checkpoint "$CKPT" \
  --out-dir "$OUTROOT/taad_smoke_nomask" --conf 0.15 --nms 15

echo "DONE -> $OUTROOT/taad_smoke_nomask"
