#!/bin/bash
#SBATCH --job-name=sv_trim_empty
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_trim_empty_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_trim_empty_%j.err

# Smoke-test `trim-empty` on a full youth match. The Saints 16B pre-MLS-Next
# clip is ~53 min with a long halftime and many stoppages — exactly the dead
# time this tool should cut. Two stages:
#   1. GPU: build a ball track from the RF-DETR ball detector.
#   2. CPU: cut offscreen/stationary spans >= --min-dead and re-encode the rest.
# The original video is never modified; all artifacts land in LOG_DIR.
#
# Usage: sbatch slurm/submit_trim_empty.sh [VIDEO] [SAMPLE_FPS] [MIN_DEAD_S]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
# Base env already carries torch(+CUDA), rfdetr and ffmpeg deps (run_pipeline uses it).
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="${1:-$REPO/data/match-saints-16b-pre-mls-next-2026-04-26.mp4}"
SAMPLE_FPS="${2:-2}"     # 2 fps → 10 samples per 5s window; plenty of resolution
MIN_DEAD="${3:-5}"       # cut dead spans longer than this many seconds

# Keep the RF-DETR weight cache off the home quota (shared with other jobs).
export HF_HOME=/blue/ewhite/b.weinstein/soccer-vision/hf_cache
mkdir -p "$HF_HOME"

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO/slurm/logs/trim_empty_$TS"
mkdir -p "$LOG_DIR"

BASE=$(basename "${VIDEO%.*}")
TRACK="$LOG_DIR/${BASE}.ball_track.json"
OUT="$LOG_DIR/${BASE}.trimmed.mp4"
EDL="$LOG_DIR/${BASE}.trim.json"

{
  echo "=== trim-empty smoke ==="
  echo "start:      $(date)"
  echo "node:       $(hostname)"
  echo "gpu:        ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "video:      $VIDEO"
  echo "sample_fps: $SAMPLE_FPS"
  echo "min_dead_s: $MIN_DEAD"
  echo "out_dir:    $LOG_DIR"
} | tee "$LOG_DIR/status.txt"
nvidia-smi 2>&1 | tee -a "$LOG_DIR/status.txt" || true

# Build the ball track (GPU) and render the trimmed cut (CPU) in one command.
# --save-track persists the intermediate track so the cut is reproducible and
# the ball-detection rate can be audited afterwards.
echo "[1/2] Building ball track + trimming dead time..." | tee -a "$LOG_DIR/status.txt"
"$PY" -m soccer_vision.cli.main trim-empty "$VIDEO" \
  --sample-fps "$SAMPLE_FPS" \
  --min-dead "$MIN_DEAD" \
  --save-track "$TRACK" \
  --out "$OUT" \
  --edl "$EDL" 2>&1 | tee "$LOG_DIR/trim.log"
RC=${PIPESTATUS[0]}

# Report ball-detection coverage — on overhead youth footage RF-DETR (trained on
# broadcast) may miss the ball, which inflates "offscreen" dead time. This tells
# us whether the cut is driven by real stoppages or by detection dropout.
echo "[2/2] Detection + cut summary:" | tee -a "$LOG_DIR/status.txt"
"$PY" - "$TRACK" "$EDL" <<'PY' 2>&1 | tee -a "$LOG_DIR/status.txt"
import json, sys
track = json.load(open(sys.argv[1]))
s = track["samples"]
vis = sum(1 for x in s if x.get("visible"))
print(f"  samples:        {len(s)}")
print(f"  ball visible:   {vis} ({100*vis/max(1,len(s)):.1f}%)")
try:
    edl = json.load(open(sys.argv[2]))
    src, kept, rem = edl["source_duration_s"], edl["kept_duration_s"], edl["removed_duration_s"]
    reasons = {}
    for r in edl["removed_segments"]:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(f"  source:         {src/60:.1f} min")
    print(f"  kept:           {kept/60:.1f} min")
    print(f"  removed:        {rem/60:.1f} min ({100*rem/max(1,src):.0f}%)")
    print(f"  dead spans:     {len(edl['removed_segments'])}  by reason: {reasons}")
    longest = sorted(edl["removed_segments"], key=lambda r: -r["duration_s"])[:5]
    print("  longest cuts (likely halftime first):")
    for r in longest:
        print(f"    {r['start_s']/60:6.1f}–{r['end_s']/60:5.1f} min  "
              f"{r['duration_s']:6.1f}s  {r['reason']}")
except Exception as e:
    print(f"  (no EDL summary: {e})")
PY

echo "done: $(date)  exit=$RC" | tee -a "$LOG_DIR/status.txt"
echo "Artifacts: $LOG_DIR" | tee -a "$LOG_DIR/status.txt"
echo "  track: $TRACK"
echo "  edl:   $EDL"
echo "  video: $OUT"
exit "$RC"
