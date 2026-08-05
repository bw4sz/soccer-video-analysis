#!/bin/bash
#SBATCH --job-name=sv_identify_negclass
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
#SBATCH --output=/home/b.weinstein/logs/sv_identify_negclass_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_identify_negclass_%j.err

# Re-run `identify` with a "not ours" class in the gallery, and find out what it
# costs on the whole match rather than on a windowed sample.
#
# The kit gate (job 38526407) fixed naming the *opposing squad*. It does nothing
# about people who are not playing at all: a standing spectator in dark clothes
# is stamped `black` by a classifier that judges lightness against turf, so the
# gate passes them and re-id names them after somebody's daughter. Measured at
# t=2400s on this run, roughly 13 of 33 tracked boxes are adults on the far
# touchline, two of them labelled "Catherine".
#
# The gallery had no way to say "none of the above" — match_track ranks our
# eleven and returns the nearest. NEGATIVE_LABEL gives the score somewhere else
# to land. Negatives are 66 hand-confirmed off-pitch lanes (1,283 crops), class
# capped at 64 like a player.
#
# **The cap is the whole ballgame, and the sweep is counter-intuitive twice over**
# (slurm/sweep_negative_class.py, 189 off-pitch / 425 on-pitch lanes):
#
#   pool    cap   rejects off   rejects on   names off/on
#   base      -             -            -          4/62
#   broad    32       27 (14%)      0 (0.0%)         3/62
#   broad    64       98 (52%)     10 (2.4%)         2/54
#   broad   128      116 (61%)     53 (12%)          0/37
#   hard     64       71 (38%)     25 (5.9%)         0/37
#   hard    256      149 (79%)     86 (20%)          0/19
#
# 1. **Hard negative mining made it worse.** Mining the 99 off-pitch lanes the
#    class failed to reject and enrolling those ("hard") is dominated by the
#    broad sample at every cap — fewer off-pitch rejections *and* more on-pitch
#    ones. In a k-NN gallery there is no boundary to sharpen: the lanes that
#    survived rejection are the ones that look most like players, so enrolling
#    them extends the negative class into player space.
# 2. **Bigger is not better.** Past 64 the class starts eating real players.
#
# So: broad negatives, cap 64. All ten on-pitch rejections at that setting were
# rendered and checked by eye and every one is a spectator at the *near*
# touchline that the foot-y proxy had mis-binned, so the true cost is below what
# the table shows.
#
# Method is `reid` to match the run this replaces (jerseys.json, method=reid,
# gallery saints-u14g.fullmatch.npz) so the diff isolates the negative class.
#
# Usage: sbatch slurm/submit_identify_negclass.sh [run_dir] [gallery] [profile] [kit]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:-runs/saints-u14g-full-30fps}"
GALLERY="${2:-galleries/saints-u14g.fullmatch-neg64.npz}"
PROFILE="${3:-examples/profiles/saints-u14g.yaml}"
KIT="${4:-black}"
BASELINE="$RUN_DIR/jerseys.pre-negclass.json"

export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== identify with a 'not ours' class ==="
echo "start:   $(date)"
echo "node:    $(hostname)"
echo "run:     $RUN_DIR"
echo "gallery: $GALLERY"
echo "kit:     $KIT"

# Keep the pre-negative-class result to diff against. Written once, so a re-run
# keeps the *original* rather than overwriting it with a previous gated one.
if [ ! -f "$BASELINE" ]; then
  cp "$RUN_DIR/jerseys.json" "$BASELINE"
  echo "baseline saved: $BASELINE"
else
  echo "baseline already present: $BASELINE (kept)"
fi

"$PY" -m soccer_vision.cli.main identify \
  --run "$RUN_DIR" \
  --method reid \
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
echo "=== what the negative class changed ==="
"$PY" slurm/compare_negclass_run.py "$RUN_DIR" --baseline "$BASELINE"

# Rebuild Morgan's reel. On-ball spans are the selection pathway (no action
# detector), so these are touches, haloed. The old reel stays for comparison.
echo
echo "=== reel: Morgan Lobey ==="
"$PY" -m soccer_vision.cli.main reel \
  --run "$RUN_DIR" \
  --player "Morgan Lobey" \
  --profile "$PROFILE" \
  --team "$KIT" \
  --halo \
  --out "$RUN_DIR/reel_morgan_negclass.mp4"

echo
echo "done: $(date)  exit=$RC"
echo "run dir:  $RUN_DIR"
echo "baseline: $BASELINE"
exit $RC
