#!/bin/bash
#SBATCH --job-name=sv_morgan_30fps
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_morgan_30fps_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_morgan_30fps_%j.err

# Rebuild Morgan's reel on the **30 fps** run, with everything the 2026-08-02
# audit changed, and see what the ball-driven selection looks like now.
#
# Every previous Morgan reel came off `runs/saints-u14g-full`, which detects at
# 5 fps. That run fragments her where the 30 fps run holds her: from the
# hand-verified lane, motion linking alone follows her 23.5 s across a handoff at
# 96% frame fill and a 6.6 px jump (`scripts/follow_player.py`). So the reel is
# rebuilt against `runs/saints-u14g-full-30fps`, which had never been through
# `identify`.
#
# Four things are different from the run that produced the 8.7-minute reel:
#
#  1. **The ball track is gated.** That run's ball_track.json predates the median
#     gate. Smoothed offline first (already done, in the repo): p95 frame step
#     635 px -> 46 px, unphysical steps 20.5% -> 3.1%, coverage 82.1% -> 78.5%.
#     On-ball spans are pure ball-to-player geometry, so this is upstream of
#     every clip boundary in the reel.
#  2. **Duplicate-id lanes are collapsed before linking** (`merge_duplicate_lanes`),
#     the failure mode that ends 9.9% of lanes over 10 s and that linking refused
#     by construction.
#  3. **The halo follows one lane at a time.** It used to union every lane
#     carrying the player's name, 2-4 of which are alive at once on 18.4% of her
#     named frames — the jumping halo (issue #28).
#  4. **Linking runs**, so a chain that re-id named anywhere propagates that name
#     along its whole length.
#
# Method is `reid`, not `reid+ocr`. The ledger's own reading of the ungated run
# is that OCR *naming* on this footage is hallucination-shaped (#1 x982), and
# `--player` resolves through re-id names regardless, so the OCR pass would
# double the cost to feed a path this reel does not take. The `crosscheck:
# "agree"` signal is worth having later — it is the most reliable identity
# evidence in the pipeline — but not on the first look.
#
# Cost: identify was 16 min on 17,395 lanes with reid+ocr. This has 29,456
# eligible lanes of 44,350 after the kit gate, but no OCR. Expect well under an
# hour; 8 h wall time is slack for the reel render, which is frame-by-frame.
#
# Usage: sbatch slurm/submit_morgan_reel_30fps.sh [run_dir] [gallery] [profile] [kit]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:-/orange/ewhite/b.weinstein/soccer-video-analysis/runs/saints-u14g-full-30fps}"
GALLERY="${2:-/orange/ewhite/b.weinstein/soccer-video-analysis/galleries/saints-u14g.fullmatch.npz}"
PROFILE="${3:-/orange/ewhite/b.weinstein/soccer-video-analysis/examples/profiles/saints-u14g.yaml}"
KIT="${4:-black}"

export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== Morgan reel, 30 fps run ==="
echo "start:   $(date)"
echo "node:    $(hostname)"
echo "run:     $RUN_DIR"
echo "gallery: $GALLERY"
echo "kit:     $KIT"

# --- 0. ball track ---------------------------------------------------------
# No-op when already gated; prints the before/after either way.
echo
echo "=== 0. ball track ==="
"$PY" scripts/smooth_saved_ball_track.py --run "$RUN_DIR" || exit 1

# --- 1. identify -----------------------------------------------------------
# Re-runnable: linking overwrites tracks.json in place, so start from the
# unlinked file if a previous attempt left one behind. Naming raw lanes and
# *then* linking is the documented order — the chain inherits whichever member
# re-id managed to name, which is the point of linking.
echo
echo "=== 1. identify (re-id, kit gate) ==="
if [ -f "$RUN_DIR/tracks.unlinked.json" ]; then
  echo "restoring raw tracks.json from tracks.unlinked.json (re-run)"
  cp "$RUN_DIR/tracks.unlinked.json" "$RUN_DIR/tracks.json"
fi

"$PY" -m soccer_vision.cli.main identify \
  --run "$RUN_DIR" \
  --method reid \
  --gallery "$GALLERY" \
  --profile "$PROFILE" \
  --team "$KIT" \
  --device cuda || exit 1

cp "$RUN_DIR/jerseys.json" "$RUN_DIR/jerseys.prelink.json"
echo "pre-link identities kept: $RUN_DIR/jerseys.prelink.json"

echo
echo "=== 1b. name concurrency BEFORE linking (issue #28) ==="
"$PY" slurm/report_name_concurrency.py --run "$RUN_DIR" --top 8

# --- 2. dedup + link -------------------------------------------------------
echo
echo "=== 2. dedup + link (names propagate along chains) ==="
"$PY" -m soccer_vision.cli.main link-tracks --run "$RUN_DIR" --in-place || exit 1

# --- 3. how concurrent are the names now? ----------------------------------
# Measured on the 5 fps run, linking made this **worse**: Morgan 18.4% -> 28.4%,
# the worst case 4 concurrent lanes -> 6, all players 17.9% -> 21.9%.
# `propagate_names` pushes a chain's name onto every member, and it has no
# one-player-one-place check either, so it buys named coverage by spreading a
# name onto lanes that are concurrent with other lanes of the same name.
#
# Both numbers are printed because they measure different things and we want
# both: linking is what makes the *halo* smooth (longer chains, fewer handoffs),
# and it is what makes the *selection* worse (more lanes wrongly carrying her
# name, so more clips of other children). If the post-link figure is much worse
# here too, the next move is to gate propagation on non-overlap rather than to
# stop linking.
echo
echo "=== 3. name concurrency AFTER linking (issue #28) ==="
"$PY" slurm/report_name_concurrency.py --run "$RUN_DIR" --top 8

# --- 4. reels --------------------------------------------------------------
# On-ball spans: no action detector exists, so these are touches. The halo now
# follows one coherent lane at a time.
for PLAYER in "Morgan Lobey" "Morrighan Wright"; do
  SLUG=$(echo "$PLAYER" | awk '{print tolower($1)}')
  OUT="$RUN_DIR/reel_${SLUG}_30fps.mp4"
  echo
  echo "=== 4. reel: $PLAYER -> $OUT ==="
  "$PY" -m soccer_vision.cli.main reel \
    --run "$RUN_DIR" \
    --player "$PLAYER" \
    --profile "$PROFILE" \
    --team "$KIT" \
    --halo \
    --out "$OUT"
done

echo
echo "done: $(date)"
echo "run dir: $RUN_DIR"
ls -la "$RUN_DIR"/reel_*_30fps.mp4 2>/dev/null
