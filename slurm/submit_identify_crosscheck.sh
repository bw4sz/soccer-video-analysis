#!/bin/bash
#SBATCH --job-name=sv_identify_xcheck
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_identify_xcheck_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_identify_xcheck_%j.err

# Re-run `identify` with jersey OCR cross-checking the re-id names, and diff the
# result against the re-id-only run that produced the current reels.
#
# The question: job 38506138 named 1,920/17,395 lanes off the full-match gallery,
# but **Gia Olson alone took 979 of them (51%)** — an attractor, not a squad. If
# those lanes are really other players, the ones carrying a legible number are
# now provably wrong, and this job counts them.
#
# Cost, versus submit_identify.sh's ~4 min: re-id is unchanged (one decode pass,
# batched forwards on a 2.2M backbone) but OCR is a **per-crop, unbatched PARSeq
# forward** over up to 40 samples of every lane — the named ones to check them and
# the abstained ones to name them. Hours, not minutes; 12h is deliberate headroom.
# `--no-ocr-verify` would skip the checking half.
#
# Deliberately NOT passing --conflict-exclude-jersey 1 on this first run. `1` is
# PARSeq's hallucination class on this footage, so it is the obvious number to
# bar — but the 0.7 per-read confidence floor may already handle it, and the
# comparison prints vetoes broken down by the number read. Measure first, then
# decide whether to bar it.
#
# Usage: sbatch slurm/submit_identify_crosscheck.sh [run_dir] [gallery] [profile]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:-runs/saints-u14g-full}"
GALLERY="${2:-galleries/saints-u14g.fullmatch.npz}"
PROFILE="${3:-examples/profiles/saints-u14g.yaml}"
BASELINE="$RUN_DIR/jerseys.reid-only.json"

export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== identify + jersey cross-check ==="
echo "start:   $(date)"
echo "node:    $(hostname)"
echo "run:     $RUN_DIR"
echo "gallery: $GALLERY"
echo "profile: $PROFILE"

# Keep the re-id-only result to diff against. Written once: on a re-run the
# baseline must stay the *original* reid-only file, not the previous cross-check.
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
  --device cuda
RC=$?

if [ $RC -eq 0 ]; then
  echo
  "$PY" slurm/compare_identify_crosscheck.py "$RUN_DIR" --show 40
fi

echo "done: $(date)  exit=$RC"
echo "run dir: $RUN_DIR"
echo "next: re-run slurm/submit_link_tracks.sh, then rebuild the reels"
exit $RC
