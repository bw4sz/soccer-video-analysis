#!/bin/bash
#SBATCH --job-name=sv_sam3_full
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_sam3_full_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_sam3_full_%j.err

# Full Saints U11 match on the SAM3 pipeline (players + ball via text prompts),
# then jersey identify. This is the first full-length SAM3 run — chunked sessions
# were only validated on ~300-frame clips, so wall time is generous (8h) to avoid
# losing hours to a timeout (process writes its JSON only at the end of the loop).
# The #6 proximity reel is built AFTER this, off tracks.json + ball_track.json.
#
# Usage: sbatch slurm/submit_sam3_saints_full.sh

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="$REPO/data/SaintsU11_OVF_Jul192026.MP4"
MATCH_ID="saints-u11-sam3-full"
RUN_DIR="$REPO/runs/$MATCH_ID"
CONFIG="$REPO/examples/saints-u11-sam3.yaml"
PROFILE="$REPO/examples/profiles/saints-u11.yaml"

# SAM3 weights live in the default HF cache (past the gate); load offline.
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
# reclaim fragmentation from the chunked-session rotations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

{
  echo "=== saints-u11 SAM3 full match ==="
  echo "start:    $(date)"
  echo "node:     $(hostname)"
  echo "video:    $VIDEO"
  echo "match_id: $MATCH_ID"
  echo "config:   $CONFIG"
} | tee "$REPO/slurm/logs/saints_sam3_full_status.txt"

echo ""
echo "[1/2] process (SAM3 players + ball) ..."
"$PY" -m soccer_vision.cli.main process "$VIDEO" \
  --match-id "$MATCH_ID" \
  --out-dir "$REPO/runs" \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --device cuda
RC=$?
if [ $RC -ne 0 ]; then echo "process failed (rc=$RC)"; exit $RC; fi

echo ""
echo "[2/2] identify (jersey numbers, roster-mapped) ..."
"$PY" -m soccer_vision.cli.main identify --run "$RUN_DIR" \
  --profile "$PROFILE" --device cuda
RC=$?
if [ $RC -ne 0 ]; then echo "identify failed (rc=$RC)"; exit $RC; fi

echo ""
echo "=== quick post-run summary ==="
"$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
tracks = json.loads((run/"tracks.json").read_text())["tracks"]
ball = json.loads((run/"ball_track.json").read_text())
vis = sum(1 for s in ball["samples"] if s["visible"])
stats = json.loads((run/"stats.json").read_text())
jer = json.loads((run/"jerseys.json").read_text())
jt = jer.get("tracks", jer)
six = [tid for tid, r in jt.items() if str(r.get("jersey")) == "6"]
named = [tid for tid, r in jt.items() if r.get("name")]
print(f"tracks: {len(tracks)}")
print(f"ball: {vis}/{len(ball['samples'])} visible ({100*vis/len(ball['samples']):.1f}%)")
print(f"teams: {stats.get('teams')}")
print(f"jersey-read tracks: {len(named)} named; tracks voting #6: {sorted(six)}")
PY

echo ""
echo "done: $(date)  exit=$RC"
echo "run dir: $RUN_DIR"
echo "NEXT: build #6 proximity reel from $RUN_DIR/{tracks,ball_track,jerseys}.json"
exit $RC
