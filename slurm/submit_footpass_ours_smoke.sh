#!/bin/bash
#SBATCH --job-name=sv_footpass_ours_smoke
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

# Smoke test: run the trained FOOTPASS TAAD baseline (job 36305311, best_model.pt
# = epoch 18) on a ~20s window of OUR Veo footage (match-saints), and produce an
# annotated clip + key frames. Two envs: extraction uses the soccer-vision .venv
# (RF-DETR/ByteTrack); inference uses the footpass env (X3D TAAD model, GPU-only).
set -euo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
VENV_PY="$REPO/.venv/bin/python"
FOOTPASS_PY=/blue/ewhite/b.weinstein/envs/footpass/bin/python
VIDEO="$REPO/data/match-saints-16b-pre-mls-next-2026-04-26.mp4"
CKPT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/runs/taad_03072026_1113/checkpoints/best_model.pt
OUTROOT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/ours
H5="$OUTROOT/our_saints_setpiece.h5"
KEY=our_saints_setpiece

# Window: restart after the 14s stoppage at 1268.8-1282.8s (trim EDL). 29.97fps:
#   1280s -> frame 38361; 600 frames = ~20s.
START=38361
NFRAMES=600

mkdir -p "$OUTROOT"

echo "[1/2] extract tracklets (RF-DETR + ByteTrack + teams) on GPU"
cd "$REPO"
"$VENV_PY" scripts/footpass_extract_tracklets.py \
  --video "$VIDEO" --start-frame "$START" --num-frames "$NFRAMES" --stride 1 \
  --game-key "$KEY" --out-h5 "$H5" --device cuda --conf 0.3 \
  --preview "$OUTROOT/tracking_preview.mp4"

echo "[2/2] run TAAD + visualize"
"$FOOTPASS_PY" scripts/footpass_infer_ours.py \
  --h5 "$H5" --game-key "$KEY" --checkpoint "$CKPT" \
  --out-dir "$OUTROOT/taad_smoke" --conf 0.15 --nms 15

echo "DONE -> $OUTROOT/taad_smoke (annotated.mp4, keyframes/, predictions.json)"
