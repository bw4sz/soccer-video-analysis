#!/bin/bash
#SBATCH --job-name=sv_footpass_ours_ball
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

# Live window + ball tracking + referee removal + gentle ball-proximity gate.
# Extractor now runs one RF-DETR pass/frame split into ball/players/referees, tracks
# the ball, drops player boxes overlapping a referee. Inference tags events by ball
# distance and drops only far+weak ones (strong off-ball actions survive).
set -euo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
VENV_PY="$REPO/.venv/bin/python"
FOOTPASS_PY=/blue/ewhite/b.weinstein/envs/footpass/bin/python
VIDEO="$REPO/data/match-saints-16b-pre-mls-next-2026-04-26.mp4"
CKPT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/runs/taad_03072026_1113/checkpoints/best_model.pt
OUTROOT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/ours
H5="$OUTROOT/our_saints_ball.h5"
KEY=our_saints_ball
START=34466
NFRAMES=600

mkdir -p "$OUTROOT"

echo "[1/2] extract tracklets + ball + referee removal"
cd "$REPO"
"$VENV_PY" scripts/footpass_extract_tracklets.py \
  --video "$VIDEO" --start-frame "$START" --num-frames "$NFRAMES" --stride 1 \
  --game-key "$KEY" --out-h5 "$H5" --device cuda --conf 0.3 --ball-conf 0.2 \
  --preview "$OUTROOT/tracking_preview_ball.mp4"

echo "[2/2] run TAAD + gentle ball gate + visualize"
"$FOOTPASS_PY" scripts/footpass_infer_ours.py \
  --h5 "$H5" --game-key "$KEY" --checkpoint "$CKPT" \
  --out-dir "$OUTROOT/taad_smoke_ball" --conf 0.15 --nms 15 --ball-gate soft

echo "DONE -> $OUTROOT/taad_smoke_ball"
