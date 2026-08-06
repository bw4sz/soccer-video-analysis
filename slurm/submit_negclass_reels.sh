#!/bin/bash
#SBATCH --job-name=sv_negclass_reels
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-milan
#SBATCH --output=/home/b.weinstein/logs/sv_negclass_reels_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_negclass_reels_%j.err

# Cut the same 13 reels as job 38723488, but off the post-negative-class
# `jerseys.json` (job 38746007), into a parallel directory so the two sets can be
# watched side by side. `reel` has no --jerseys flag — it reads the run's
# jerseys.json — so this is the only way to A/B: the baseline set already sits in
# $RUN_DIR/reels/ and is not touched.
#
# **This is a validation render, not a result.** The headline "2311 -> 100 names"
# from 38746007 is not the negative class: of the 1,425 lanes re-id had named,
# 1,002 were dropped by --min-lane-seconds (median span 0.03s), 223 no longer
# exist after dedup/link changed the lane file, 100 abstained, and only **19**
# lost a name to "not ours". The baseline it was diffed against also carried 886
# propagated names that this run never ran propagation to produce.
#
# So the honest question these reels answer is narrow: with far fewer names, is
# what survives *right* — one player, on the pitch, in our kit?
#
# Usage: sbatch slurm/submit_negclass_reels.sh [run_dir] [profile] [kit]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:-runs/saints-u14g-full-30fps}"
PROFILE="${2:-examples/profiles/saints-u14g.yaml}"
KIT="${3:-black}"
OUT_DIR="$RUN_DIR/reels_negclass"

export PYTHONUNBUFFERED=1
module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true
mkdir -p "$OUT_DIR"

echo "=== negative-class reels ==="
echo "start:    $(date)"
echo "node:     $(hostname)"
echo "run:      $RUN_DIR"
echo "jerseys:  $RUN_DIR/jerseys.json  (post-negclass, job 38746007)"
echo "baseline: $RUN_DIR/reels/        (job 38723488, pre-negclass)"
echo "out:      $OUT_DIR"

mapfile -t PLAYERS < <("$PY" - "$PROFILE" <<'PYEOF'
import sys
from soccer_vision.profiles.loader import load_profile
for p in load_profile(sys.argv[1])["roster"]:
    print(p["name"])
PYEOF
)
echo "roster: ${#PLAYERS[@]} players"

for PLAYER in "${PLAYERS[@]}"; do
  SLUG=$(echo "$PLAYER" | tr '[:upper:] ' '[:lower:]_' | tr -cd 'a-z0-9_-')
  echo
  echo "--- $PLAYER -> $OUT_DIR/reel_${SLUG}.mp4 ---"
  "$PY" -m soccer_vision.cli.main reel \
    --run "$RUN_DIR" --player "$PLAYER" --profile "$PROFILE" \
    --team "$KIT" --halo --out "$OUT_DIR/reel_${SLUG}.mp4"
done

echo
echo "=== done: $(date) ==="
echo "minutes per reel (negclass vs baseline):"
for f in "$OUT_DIR"/*.mp4; do
  B="$RUN_DIR/reels/$(basename "$f")"
  DN=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  DB=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$B" 2>/dev/null)
  printf "  %-40s %6.1f min   (baseline %6.1f min)\n" "$(basename "$f")" \
    "$(echo "${DN:-0}/60" | bc -l)" "$(echo "${DB:-0}/60" | bc -l)"
done
