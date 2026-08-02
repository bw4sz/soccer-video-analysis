#!/bin/bash
#SBATCH --job-name=sv_footpass_ours_u14g
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

# First TAAD run on U14G footage. Every previous TAAD-on-our-footage smoke
# (36450376 / 36460253 / 36500443 / 38162552) used the SAME video —
# match-saints-16b-pre-mls-next-2026-04-26.mp4, the XbotGo camera. The class
# collapse those runs measured is therefore one camera's domain gap, measured
# four times. U14G is a different camera (Veo), a different venue (shared
# multi-pitch), and a different sun angle (low, long shadows, lens flare), and
# TAAD has never been pointed at it.
#
# Protocol is deliberately identical to job 36500443 so the two are comparable:
# 600 frames, stride 1, RF-DETR + ByteTrack front end, --conf 0.3 --ball-conf
# 0.2, and TAAD at --conf 0.15 --nms 15 --ball-gate soft off the same
# taad_03072026_1113 best_model.pt. ONLY the video and window change.
#
# Window: frames 2700-3300 of data/u14g_smoke180.mp4 (~90-110s into the clip,
# which is itself cut from 15:00 of the match). Picked off
# runs/u14g-smoke-rfdetr/ball_track.json as live open play: 92-95% ball
# visibility and 56-120 px median frame-to-frame ball motion, i.e. the ball is
# on screen and moving. July's first smoke (36450376) was invalidated by
# landing in a dead-ball span; this avoids repeating that.
#
# RF-DETR not SAM3: job 38162552 swapped the front end on the XbotGo window and
# TAAD's class distribution did not move, so the detector is not the variable
# under test here. RF-DETR is also 33x faster on this exact footage (0.046 vs
# 1.540 s/frame, job 38180242), so this run costs GPU-minutes.
#
# Read the result against the in-domain class prior (FOOTPASS val, 6070 events):
#   pass 50.4%  drive 40.7%  header 2.7%  cross 1.8%
#   throw-in 1.6%  block 1.3%  shot 1.1%  tackle 0.4%
# The XbotGo runs inverted it: block+shot 71-75% against a 2.4% prior, with
# pass+drive at 14-17% against 91%. The question is whether U14G does the same.
#
# KNOWN CAVEAT: the turf polygon (--field-mask auto) is the largest green blob,
# and this venue is a multi-pitch complex, so the NEIGHBOURING match is also
# turf and will not be excluded. The --pitch-region flag that would have fixed
# this was removed in 24557ed ("re-id is the answer we want"). Expect
# neighbouring-pitch players to compete for TAAD's top-13-longest slots; judge
# the preview before judging the class distribution.
set -euo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
VENV_PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python   # rfdetr + supervision
FOOTPASS_PY=/blue/ewhite/b.weinstein/envs/footpass/bin/python    # torch 2.1 + decord for TAAD
VIDEO="$REPO/data/u14g_smoke180.mp4"
CKPT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/runs/taad_03072026_1113/checkpoints/best_model.pt
OUTROOT=/blue/ewhite/b.weinstein/soccer-vision-data/footpass/ours
TAG="${1:-u14g}"
H5="$OUTROOT/our_u14g_rfdetr_${TAG}.h5"
KEY="our_u14g_rfdetr_${TAG}"
START=2700
NFRAMES=600

export PYTHONUNBUFFERED=1   # SLURM redirects stdout to a file; buffering reads as a hang
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTROOT"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[1/2] extract tracklets (RF-DETR + ByteTrack + team split)"
PYTHONPATH="$REPO/src" "$VENV_PY" -u scripts/footpass_extract_tracklets.py \
  --video "$VIDEO" --start-frame "$START" --num-frames "$NFRAMES" --stride 1 \
  --game-key "$KEY" --out-h5 "$H5" --device cuda \
  --conf 0.3 --ball-conf 0.2 --ref-filter vote \
  --preview "$OUTROOT/tracking_preview_${TAG}.mp4"

echo "[2/2] run TAAD + gentle ball gate + visualize (flags identical to 36500443)"
"$FOOTPASS_PY" -u scripts/footpass_infer_ours.py \
  --h5 "$H5" --game-key "$KEY" --checkpoint "$CKPT" \
  --out-dir "$OUTROOT/taad_smoke_${TAG}" --conf 0.15 --nms 15 --ball-gate soft

echo "DONE -> $OUTROOT/taad_smoke_${TAG}"
