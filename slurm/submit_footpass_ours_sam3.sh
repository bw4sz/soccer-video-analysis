#!/bin/bash
#SBATCH --job-name=sv_footpass_ours_sam3
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite

# Controlled re-run of the TAAD-on-our-footage smoke test with SAM3 tracklets.
#
# Same video, same 1150-1170s live window, same checkpoint, same inference flags
# as job 36500443 (taad_smoke_ball). The ONLY change is the tracklet front end:
# RF-DETR + ByteTrack -> SAM3 text-prompt detect+track.
#
# Why: TAAD is track-aware — tracklets are its input and it scores the top-13
# longest tracks per team. RF-DETR finds 5-6 players/frame on this footage vs
# SAM3's 20-22, so the July runs fed it a pitch missing most of its players. The
# in-domain eval (38133841) proved RF-DETR is a strong detector that collapses
# under domain shift, so "TAAD doesn't transfer" and "the input was incomplete"
# are currently confounded. This run separates them.
#
# Read the result against the in-domain class prior (FOOTPASS val, 6070 events):
# pass 50.4%, drive 40.7%, header 2.7%, cross 1.8%, throw-in 1.6%, block 1.3%,
# shot 1.1%, tackle 0.4%. The RF-DETR run inverted it: block 52.8% + shot 22.2%
# = 75% of predictions against a 2.4% prior, with pass+drive at 14% against 91%.
set -euo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
VENV_PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python   # has SAM3 (transformers 5.12)
FOOTPASS_PY=/blue/ewhite/b.weinstein/envs/footpass/bin/python    # torch 2.1 + decord for TAAD
VIDEO="$REPO/data/match-saints-16b-pre-mls-next-2026-04-26.mp4"
CKPT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/runs/taad_03072026_1113/checkpoints/best_model.pt
OUTROOT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/ours
TAG="${1:-refvote}"
H5="$OUTROOT/our_saints_sam3_${TAG}.h5"
KEY="our_saints_sam3_${TAG}"
START=34466
NFRAMES=600

# SAM3 + RF-DETR weights are both cached here and past their gates.
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTROOT"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[1/2] extract tracklets with SAM3 (players + ball + referee concepts)"
PYTHONPATH="$REPO/src" "$VENV_PY" -u scripts/footpass_extract_tracklets.py \
  --video "$VIDEO" --start-frame "$START" --num-frames "$NFRAMES" --stride 1 \
  --game-key "$KEY" --out-h5 "$H5" --device cuda --detector sam3 --sam3-chunk 30 \
  --ref-filter vote --ref-vote 0.6 --ref-iou 0.45 \
  --preview "$OUTROOT/tracking_preview_sam3_${TAG}.mp4"

echo "[2/2] run TAAD + gentle ball gate + visualize (flags identical to 36500443)"
"$FOOTPASS_PY" -u scripts/footpass_infer_ours.py \
  --h5 "$H5" --game-key "$KEY" --checkpoint "$CKPT" \
  --out-dir "$OUTROOT/taad_smoke_sam3_${TAG}" --conf 0.15 --nms 15 --ball-gate soft

echo "DONE -> $OUTROOT/taad_smoke_sam3_${TAG}"
