#!/bin/bash
#SBATCH --job-name=sv_process
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_process_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_process_%j.err

# Process any match. Every squad needs its own run before `enroll` can dump
# crops or frames for it, so this takes the video and match id as arguments.
#
# Defaults to RF-DETR (examples/process_match.yaml): public weights, no HF gate,
# ~26x faster than SAM3, and better where we have ground truth — see that file.
# A full 60-min match is ~1.6h, which is what the 4h wall is sized for. SAM3 on
# the same match needs 9.4h and used to silently blow an 8h wall (job 38162800).
#
# To opt into SAM3 instead, pass its config as the 4th argument:
#   sbatch slurm/submit_process.sh <video> <id> <profile> examples/saints-u11-sam3.yaml
# and raise --time to 12:00:00, or it will not finish.
#
# Usage: sbatch slurm/submit_process.sh <video> <match_id> [profile] [config]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="${1:?usage: sbatch slurm/submit_process.sh <video> <match_id> [profile] [config]}"
MATCH_ID="${2:?missing match_id}"
PROFILE="${3:-}"
CONFIG="${4:-$REPO/examples/process_match.yaml}"   # detector config, not team-specific
RUN_DIR="$REPO/runs/$MATCH_ID"

# Both RF-DETR and SAM3 weights are already in this cache, so the job loads
# fully offline (no token, and no HF gate to trip on a compute node).
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
# reclaim fragmentation from SAM3's chunked-session rotations (inert for RF-DETR)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
