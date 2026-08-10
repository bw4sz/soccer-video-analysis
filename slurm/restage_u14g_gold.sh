#!/bin/bash
#SBATCH --job-name=sv_restage_u14g
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-default
#SBATCH --output=/home/b.weinstein/logs/sv_restage_u14g_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_restage_u14g_%j.err

# Re-render the U14G gold set with our players on the early pages.
#
# The first staging ranked a window's lanes by how many frames they stayed on
# screen. That ranks by standing still — the annotator opened pass 3 of the
# 2400 s stretch and found not one of our squad in it. Two orderings replace it,
# both measured on these exact nine windows:
#
#   --promote-kit black   our kit first. Page 1 goes from 7 of ours to 16 of 16,
#                         pages 1-2 to 262 of the 313 black-kit lanes.
#   --rank motion         within each group, by distance travelled net of camera
#                         pan, in body-heights. Static lanes leave pages 1-3
#                         (15 -> 5) and pass 7 becomes entirely static.
#
# Neither drops a lane: all 767 still get a pass, because the opposition lanes
# are what measure this pipeline's largest error (naming an opponent as one of
# ours) and a filtered lane cannot be asked about at all.
#
# No GPU: this is video decode and H.264 encode, ~1 min per clip x 51.
#
# Usage: sbatch slurm/restage_u14g_gold.sh

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY="$REPO/.venv/bin/python"
RUN="$REPO/runs/saints-u14g-full-30fps"
PROFILE="$REPO/examples/profiles/saints-u14g.yaml"
GOLD="$RUN/heldout_gold"
BACKUP="$RUN/heldout_gold.length-ranked"

cd "$REPO"
module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true
export PYTHONUNBUFFERED=1

if [ ! -f "$RUN/tracks.json" ]; then
  echo "No $RUN/tracks.json"
  exit 1
fi

# Keep the old slot map. The clips are regenerable in an hour, but tracklets.json
# is the only key to any annotation already sitting in a Label Studio project —
# without it an export of the old pages is uninterpretable.
if [ -f "$GOLD/tracklets.json" ] && [ ! -d "$BACKUP" ]; then
  mkdir -p "$BACKUP"
  cp "$GOLD"/*.json "$GOLD"/*.xml "$GOLD"/*.md "$BACKUP"/ 2>/dev/null
  echo "kept the old slot map in $BACKUP"
fi

# Stale clips would otherwise survive under names the new manifest doesn't use.
rm -f "$GOLD"/clips/*.mp4

echo "=== restaging the gold set: ours first, then busiest ==="
"$PY" -m soccer_vision.cli.main enroll \
  --run "$RUN" \
  --dump-tracklets "$GOLD" \
  --profile "$PROFILE" \
  --at 2400 --window 20 --n-windows 9 \
  --all-lanes --max-lanes 16 --min-track-frames 30 \
  --rank motion --promote-kit black \
  --heldout-mode only \
  --serve-root "$REPO"

cp "$REPO/docs/annotating-u14g-gold.md" "$GOLD/ANNOTATING.md" 2>/dev/null

echo ""
echo "=== what the annotator will meet on each pass ==="
"$PY" - <<'PYEOF'
import collections, json
from pathlib import Path

run = Path("runs/saints-u14g-full-30fps")
teams = {int(k): v for k, v in json.loads((run / "tracks.json").read_text())["teams"].items()}
man = json.loads((run / "heldout_gold" / "tracklets.json").read_text())

by_page = collections.defaultdict(collections.Counter)
for w in man["windows"]:
    for lane in w["lanes"]:
        by_page[w["page"]][teams.get(lane["track_id"], "-")] += 1

print(f"{'pass':>5} {'lanes':>6} {'ours (black)':>13} {'white':>7} {'no kit':>7}")
for page in sorted(by_page):
    c = by_page[page]
    print(f"{page:5d} {sum(c.values()):6d} {c['black']:13d} {c['white']:7d} {c['-']:7d}")
print(f"\n{len(man['windows'])} tasks, "
      f"{sum(len(w['lanes']) for w in man['windows'])} lanes")
PYEOF

echo ""
echo "done: $(date)"
echo "  gold: $GOLD  (read ANNOTATING.md there)"
echo "  old slot map kept at: $BACKUP"
