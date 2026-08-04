#!/bin/bash
#SBATCH --job-name=sv_all_reels
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-milan
#SBATCH --output=/home/b.weinstein/logs/sv_all_reels_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_all_reels_%j.err

# Cut a reel for **every player on the roster** and one for the **whole kit**,
# off saved artefacts, to see what the pipeline actually delivers end to end.
#
# Every reel so far has been Morgan (and once Morrighan), which is the player the
# gallery is deepest on and the one every measurement in `CLAUDE.md` is anchored
# to. Thirteen reels answers a different question: what does a parent of *any*
# child on this squad get today? The census that motivated this run
# (`--on-ball-dist 90`, `--on-ball-min-span 0.4`, on `runs/saints-u14g-full-30fps`):
#
#   selection                 lanes  spans  touch_s  reel_min  signal
#   ALL black lanes            7519    694     1831      59.2   51.6%
#   ALL named black             447    296      365      34.3   17.7%
#   Morgan Lobey (#4)            14     14       22       2.1   17.8%
#   Catherine Conroy (#6)       114     59       58       7.8   12.3%
#   Gia Olson (#7)               86     74       93      10.1   15.3%
#   Izabelle Scott-Snow (#8)     82     77       76       8.9   14.2%
#   Joelle Fontenot (#9)          0      0        0       0.0    0.0%
#   Leire Cabral (#10)           59     39       35       5.1   11.6%
#   Eveleigh Bottorff (#11)      24     14       32       2.6   20.5%
#   Ila Sheets (#17)              6      3        5       0.6   14.0%
#   Morrighan Wright (#21)        5      4        2       0.6    5.5%
#   Lainey Jarvis (#26)           0      0        0       0.0    0.0%
#   Riley McNicholas (#37)       18     19       15       2.0   12.6%
#   Iris McDonald (#50)          23     12        7       1.8    6.3%
#   Quinn Perrin (#88)           16     32       25       3.5   12.0%
#
# Two players get **no reel at all** — not because they did not play, but because
# re-id never named a single one of their lanes. That is the *Identity coverage*
# number of `CLAUDE.md` seen from the parent's side rather than in aggregate.
#
# **The team reel is the control**, and it is why `--team` alone now cuts a reel
# (`cli/extract.py::_on_ball_fallback`, which used to refuse it as "not a player
# query"). It anchors on-ball spans on every black-kit lane and needs no identity
# at all, so it holds 1,831 s of our touches against the 365 s that carry a name.
# Watching it next to the player reels shows the loss is naming, not football.
# It is rendered **without a halo** on purpose: the one-halo-per-player rule
# (issue #28) has no player to be about here, so a spotlight would just be
# ringing whichever lane won an arbitrary pick.
#
# Costs nothing but CPU: no detector, no GPU, no re-`identify`. Reads
# tracks.json / ball_track.json / jerseys.json as they stand.
#
# **Naming predates `--min-lane-seconds`.** This run's jerseys.json was written
# on 2026-08-02 before that gate landed (commit 0461000), so no lane in it
# carries `excluded: "short"`. Simulated, the 1.0 s gate drops 81% of the naming
# decisions and 21% of the named football, so the player reels here are the
# *ungated* — wider and dirtier — selection. Pass RE_IDENTIFY=1 to rebuild
# naming with current defaults first; that needs a GPU partition and about an
# hour, and is the only reason this script would want one.
#
# Usage: sbatch slurm/submit_all_reels.sh [run_dir] [profile] [kit]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:-$REPO/runs/saints-u14g-full-30fps}"
PROFILE="${2:-$REPO/examples/profiles/saints-u14g.yaml}"
KIT="${3:-black}"
OUT_DIR="$RUN_DIR/reels"
RE_IDENTIFY="${RE_IDENTIFY:-0}"

export PYTHONUNBUFFERED=1
module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

mkdir -p "$OUT_DIR"

echo "=== every player + the whole $KIT kit ==="
echo "start:   $(date)"
echo "node:    $(hostname)"
echo "run:     $RUN_DIR"
echo "profile: $PROFILE"
echo "out:     $OUT_DIR"

if [ "$RE_IDENTIFY" = "1" ]; then
  echo
  echo "=== 0. re-identify with current defaults (incl. --min-lane-seconds) ==="
  export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
  [ -f "$RUN_DIR/tracks.unlinked.json" ] && \
    cp "$RUN_DIR/tracks.unlinked.json" "$RUN_DIR/tracks.json"
  "$PY" -m soccer_vision.cli.main identify --run "$RUN_DIR" --method reid \
    --gallery "$REPO/galleries/saints-u14g.fullmatch.npz" \
    --profile "$PROFILE" --team "$KIT" --device cuda || exit 1
  cp "$RUN_DIR/jerseys.json" "$RUN_DIR/jerseys.prelink.json"
  "$PY" -m soccer_vision.cli.main link-tracks --run "$RUN_DIR" --in-place || exit 1
fi

# --- 1. the whole kit, no identity required --------------------------------
# Rendered first: it is the control the player reels are read against, and it is
# the one output here that cannot be empty for want of a name.
echo
echo "=== 1. team reel: all $KIT lanes on the ball ==="
"$PY" -m soccer_vision.cli.main reel \
  --run "$RUN_DIR" --team "$KIT" \
  --out "$OUT_DIR/reel_team_${KIT}.mp4"

# --- 2. one reel per roster player -----------------------------------------
# Every name is attempted, including the ones the census says have nothing, so
# the log records an explicit "no lane carries this name" rather than a gap.
# Slug is the full name: two players can share a first name, and a reel silently
# overwriting another child's is the worst possible failure here.
echo
echo "=== 2. per-player reels ==="
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
  OUT="$OUT_DIR/reel_${SLUG}.mp4"
  echo
  echo "--- $PLAYER -> $OUT ---"
  "$PY" -m soccer_vision.cli.main reel \
    --run "$RUN_DIR" --player "$PLAYER" --profile "$PROFILE" \
    --team "$KIT" --halo --out "$OUT"
done

echo
echo "=== done: $(date) ==="
ls -la "$OUT_DIR"
echo
echo "minutes of footage per reel:"
for f in "$OUT_DIR"/*.mp4; do
  D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  printf "  %-44s %6.1f min\n" "$(basename "$f")" "$(echo "$D/60" | bc -l)"
done
