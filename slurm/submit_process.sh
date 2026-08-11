#!/bin/bash
#SBATCH --job-name=sv_process
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_process_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_process_%j.err

# Process any match. Every squad needs its own run before `enroll` can dump
# crops or frames for it, so this takes the video and match id as arguments.
#
# Detection is RF-DETR (examples/process_match.yaml): public weights, no HF gate,
# 0.045 s/detection-frame, and better where we have ground truth — see that file.
# A full 60-min match is ~1.6h, which is what the 4h wall is sized for; most of
# that is video decoding, not detection.
#
# 128GB, not 64: per-frame track boxes are held for the whole run so tracks.json
# can be written per-track at the end, and on a 60-minute match at this venue
# that is a lot of lanes. Job 38313387 was OOM-killed at 64GB 18% in.
#
# Usage: sbatch slurm/submit_process.sh <video> <match_id> [profile] [config]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="${1:?usage: sbatch slurm/submit_process.sh <video> <match_id> [profile] [config]}"
MATCH_ID="${2:?missing match_id}"
PROFILE="${3:-}"
CONFIG="${4:-$REPO/examples/process_match.yaml}"   # detector config, not team-specific
RUN_DIR="$REPO/runs/$MATCH_ID"

# RF-DETR weights are already in this cache, so the job loads fully offline.
export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
# Unbuffered: SLURM redirects stdout to a file, so Python buffers it and a job
# killed at the wall limit leaves a log showing only step 1 — indistinguishable
# from a hang. Progress must land in the log as it happens (job 38162799).
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== soccer-vision process ==="
echo "start:    $(date)"
echo "node:     $(hostname)"
echo "video:    $VIDEO"
echo "match_id: $MATCH_ID"
echo "profile:  ${PROFILE:-<none>}"
echo "config:   $CONFIG"

ARGS=(--match-id "$MATCH_ID" --out-dir "$REPO/runs" --config "$CONFIG")
[ -n "$PROFILE" ] && ARGS+=(--profile "$PROFILE")

"$PY" -m soccer_vision.cli.main process "$VIDEO" "${ARGS[@]}"
RC=$?

if [ $RC -eq 0 ]; then
  "$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
tracks = json.loads((run / "tracks.json").read_text())
teams = tracks.get("teams") or {}
ball = json.loads((run / "ball_track.json").read_text())
vis = sum(1 for s in ball["samples"] if s["visible"])
print(f"tracks: {len(tracks['tracks'])}")
print(f"kit-stamped tracks: {len(teams)}")
print(f"ball: {vis}/{len(ball['samples'])} visible ({100 * vis / max(1, len(ball['samples'])):.1f}%)")
PY
fi

echo ""
echo "done: $(date)  exit=$RC"
echo "run dir: $RUN_DIR"
echo "NEXT: soccer-vision enroll --run $RUN_DIR --dump-frames $RUN_DIR/label_frames --profile <profile>"
exit $RC
