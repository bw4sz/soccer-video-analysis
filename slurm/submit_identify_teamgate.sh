#!/bin/bash
#SBATCH --job-name=sv_identify_teamgate
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
#SBATCH --output=/home/b.weinstein/logs/sv_identify_teamgate_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_identify_teamgate_%j.err

# Re-run `identify` with the new kit gate (`--team black`), and test the
# prediction job 38526407 left on the table.
#
# That job found 38 re-id names disproved by legible jersey reads, and the
# numbers those shirts carried (#2 x11, #1 x10, #20 x7, #23 x3) are mostly worn
# by nobody on our roster — the signature of the gallery pulling *opponents* onto
# our squad rather than confusing teammates. The 2026-08-02 audit put a number on
# it: of 1,920 re-id names, 1,163 landed on white-kit lanes against 296 on black,
# and Saints U14G are the black kit.
#
# So the gate is not a tuning knob, it is a missing constraint, and it makes a
# falsifiable prediction: **the veto's conflicts should largely disappear.** If
# they hold steady on our own kit instead, the remaining errors are teammate
# confusions and the gate was never going to touch them — which is worth knowing
# just as much.
#
# Method is reid+ocr, matching the run this replaces so the diff isolates the
# gate. Note the OCR *naming* caveat from the ledger: on the ungated run it named
# 2,589 abstained lanes with a hallucination-shaped distribution (#1 x982), so
# treat `source: "ocr"` names here as unvetted — the reels below are built from
# --player, which resolves through re-id names.
#
# Cost: 16 min on the ungated run. The gate excludes ~3,100 of 17,395 lanes
# before any model runs, so this should be faster, not slower.
#
# Usage: sbatch slurm/submit_identify_teamgate.sh [run_dir] [gallery] [profile] [kit]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:-runs/saints-u14g-full}"
GALLERY="${2:-galleries/saints-u14g.fullmatch.npz}"
PROFILE="${3:-examples/profiles/saints-u14g.yaml}"
KIT="${4:-black}"
BASELINE="$RUN_DIR/jerseys.pre-teamgate.json"

export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== identify with the kit gate ==="
echo "start:   $(date)"
echo "node:    $(hostname)"
echo "run:     $RUN_DIR"
echo "gallery: $GALLERY"
echo "kit:     $KIT"

# Keep the ungated result to diff against. Written once, so a re-run keeps the
# *original* pre-gate file rather than overwriting it with a previous gated one.
if [ ! -f "$BASELINE" ]; then
  cp "$RUN_DIR/jerseys.json" "$BASELINE"
  echo "baseline saved: $BASELINE"
else
  echo "baseline already present: $BASELINE (kept)"
fi

"$PY" -m soccer_vision.cli.main identify \
  --run "$RUN_DIR" \
  --method reid+ocr \
  --gallery "$GALLERY" \
  --profile "$PROFILE" \
  --team "$KIT" \
  --device cuda
RC=$?

if [ $RC -ne 0 ]; then
  echo "identify failed (exit $RC) — leaving $BASELINE in place"
  echo "done: $(date)"
  exit $RC
fi

echo
"$PY" slurm/compare_team_gate.py "$RUN_DIR" --team "$KIT"

# Rebuild the two reels that motivated this, to new names so the old ones stay
# around for comparison. On-ball spans are the selection pathway (no action
# detector), so these are touches, haloed.
for PLAYER in "Morgan Lobey" "Morrighan Wright"; do
  SLUG=$(echo "$PLAYER" | awk '{print tolower($1)}')
  OUT="$RUN_DIR/reel_${SLUG}_teamgate.mp4"
  echo
  echo "=== reel: $PLAYER -> $OUT ==="
  "$PY" -m soccer_vision.cli.main reel \
    --run "$RUN_DIR" \
    --player "$PLAYER" \
    --profile "$PROFILE" \
    --team "$KIT" \
    --halo \
    --out "$OUT"
done

echo
echo "done: $(date)  exit=$RC"
echo "run dir:  $RUN_DIR"
echo "baseline: $BASELINE"
echo "next: sbatch slurm/submit_link_tracks.sh $RUN_DIR, then rebuild the linked reels"
exit $RC
