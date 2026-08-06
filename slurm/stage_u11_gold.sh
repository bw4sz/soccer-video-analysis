#!/bin/bash
#SBATCH --job-name=sv_stage_u11
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-default
#SBATCH --output=/home/b.weinstein/logs/sv_stage_u11_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_stage_u11_%j.err

# Stage the two U11 XbotGo Label Studio projects, once `process` has finished.
#
# Between-video generalization needs both halves of the split, and they must not
# touch: Simon is enrolled from the FIRST half and scored on the SECOND, which
# `heldout.yaml` declares off-limits (945-1845 s). Neither dump chooses that
# boundary itself — the registry does, and the enrolment dump would drop a window
# landing past 945 s even if this script asked for one.
#
#   enrol_tracklets/   spread over 0-900 s   -> gallery
#   heldout_gold/      945-1845 s            -> answer sheet, enrolment refused
#
# No GPU: rendering ringed clips is video decode and H.264 encode.
#
# Usage: sbatch slurm/stage_u11_gold.sh [n_gold_stretches]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY="$REPO/.venv/bin/python"
RUN="$REPO/runs/saints-u11-xbotgo-30fps"
PROFILE="$REPO/examples/profiles/saints-u11.yaml"

# 45 x 20 s covers the whole held-out half. Pass a smaller number to cover the
# first N x 20 s of it instead — the stretches run back to back from 945 s, so a
# partial run leaves a contiguous prefix rather than a scattered sample.
N_GOLD="${1:-45}"

cd "$REPO"
module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true
export PYTHONUNBUFFERED=1

if [ ! -f "$RUN/tracks.json" ]; then
  echo "No $RUN/tracks.json — run slurm/submit_process.sh on the U11 video first."
  exit 1
fi

echo "=== how big is this going to be? ==="
"$PY" slurm/size_tracklet_project.py --run "$RUN" \
  --at 945 --window 20 --n-windows "$N_GOLD" --all-lanes --max-lanes 16 \
  --heldout-mode only --sweep-min-lane-seconds

echo ""
echo "=== 1/2  enrolment windows (first half only) ==="
# No --at, so windows spread evenly over the whole video; the held-out gate then
# drops the ones past 945 s. Asking for 24 leaves roughly a dozen in the first
# half, which is the batch size that stopped helping on the U14G (CLAUDE.md:
# "the answer to a thin gallery is more windows and more videos, not longer ones").
"$PY" -m soccer_vision.cli.main enroll \
  --run "$RUN" \
  --dump-tracklets "$RUN/enrol_tracklets" \
  --profile "$PROFILE" \
  --window 20 --n-windows 24 --max-lanes 16 --min-track-frames 30 \
  --serve-root "$REPO"

echo ""
echo "=== 2/2  held-out gold set (second half, every lane) ==="
"$PY" -m soccer_vision.cli.main enroll \
  --run "$RUN" \
  --dump-tracklets "$RUN/heldout_gold" \
  --profile "$PROFILE" \
  --at 945 --window 20 --n-windows "$N_GOLD" --all-lanes --max-lanes 16 \
  --min-track-frames 30 --heldout-mode only \
  --serve-root "$REPO"

cp "$REPO/docs/annotating-u11-simon.md" "$RUN/heldout_gold/ANNOTATING.md" 2>/dev/null

echo ""
echo "done: $(date)"
echo "  enrol:  $RUN/enrol_tracklets"
echo "  gold:   $RUN/heldout_gold   (read ANNOTATING.md there)"
echo "  check:  $PY scripts/audit_heldout.py --run $RUN"
