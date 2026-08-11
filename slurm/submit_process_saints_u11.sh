#!/bin/bash
#SBATCH --job-name=sv_saints_u11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_saints_u11_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_saints_u11_%j.err

# Full individual-player pathway on the Saints U11 vs OVF match (2026-07-19).
# Goal: how many action clips can we cut for the BLACK team's #6, and what
# event labels do they carry?
#
# Stages (all GPU except the final ffmpeg cutting):
#   1. process  — ball track + ByteTrack + team colours + rule-based events
#   2. identify — confidence-weighted per-track jersey-number vote (jerseys.json)
#   3. summary  — count #6/black events by label (before rendering anything)
#   4. extract  — render the #6/black clips with a player halo
#
# The original video is never modified. No --profile: black #6 is the opponent,
# not in the Saints roster, so we select by number only (no player name).
#
# Usage: sbatch slurm/submit_process_saints_u11.sh

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="$REPO/data/SaintsU11_OVF_Jul192026.MP4"
MATCH_ID="saints-u11-ovf-2026-07-19"
RUN_DIR="$REPO/runs/$MATCH_ID"

# Keep the RF-DETR / PARSeq weight caches off the home quota.
export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

{
  echo "=== saints-u11 full pathway ==="
  echo "start:    $(date)"
  echo "node:     $(hostname)"
  echo "gpu:      ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "video:    $VIDEO"
  echo "match_id: $MATCH_ID"
  echo "run_dir:  $RUN_DIR"
} | tee "$REPO/slurm/logs/saints_u11_status.txt"
nvidia-smi 2>&1 | head -15 || true

echo ""
echo "[1/4] process ..."
"$PY" -m soccer_vision.cli.main process "$VIDEO" \
  --match-id "$MATCH_ID" \
  --out-dir "$REPO/runs" \
  --device cuda
RC=$?
if [ $RC -ne 0 ]; then echo "process failed (rc=$RC)"; exit $RC; fi

echo ""
echo "[2/4] identify (jersey numbers) ..."
"$PY" -m soccer_vision.cli.main identify --run "$RUN_DIR"
RC=$?
if [ $RC -ne 0 ]; then echo "identify failed (rc=$RC)"; exit $RC; fi

echo ""
echo "[3/4] summary — #6 / black events by label (pre-render):"
"$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter

run = Path(sys.argv[1])
ann = json.loads((run / "annotations.json").read_text())
events = ann.get("events", [])
jerseys = json.loads((run / "jerseys.json").read_text())

# Which team colours exist, and which tracks vote #6?
teams = Counter((e.get("team") or "unknown") for e in events)
print(f"  team colours in events: {dict(teams)}")

# jerseys.json is keyed by track id -> {jersey, confidence, team?, ...}
six_tracks = set()
for tid, rec in jerseys.items():
    if str(rec.get("jersey")) == "6":
        six_tracks.add(int(tid))
print(f"  tracks voting #6: {sorted(six_tracks)}")

def team_of(e):
    return (e.get("team") or "").lower()

six_events = [e for e in events if e.get("track_id") in six_tracks]
six_black  = [e for e in six_events if team_of(e) == "black"]

print(f"  #6 events (any team): {len(six_events)}")
print(f"  #6 events on BLACK:   {len(six_black)}")
by_label = Counter(e.get("label") for e in six_black)
print(f"  #6/black by label:    {dict(by_label)}")
# Also show any-team #6 breakdown, in case team naming isn't 'black'.
print(f"  #6 any-team by (team,label): "
      f"{dict(Counter((team_of(e), e.get('label')) for e in six_events))}")
PY

echo ""
echo "[4/4] extract — render #6 / black clips with halo ..."
"$PY" -m soccer_vision.cli.main extract --run "$RUN_DIR" \
  --team black --number 6 --halo
RC=$?

echo ""
echo "clips in $RUN_DIR/clips:"
ls -la "$RUN_DIR/clips" 2>/dev/null | tail -30

echo ""
echo "done: $(date)  exit=$RC"
echo "run dir: $RUN_DIR"
exit $RC
