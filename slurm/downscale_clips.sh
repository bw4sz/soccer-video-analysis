#!/bin/bash
#SBATCH --job-name=sv_clips_720p
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=hpg-milan
#SBATCH --output=/home/b.weinstein/logs/sv_clips_720p_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_clips_720p_%j.err

# Re-encode a run's event clips to 720p for annotation on a laptop.
#
# Clips come out of `process` at the source resolution (1920x1080, ~25 MB per
# 20 s clip = 6.1 GB for a full match), which is far more than Label Studio
# needs to scrub and far more than anyone wants to rsync. This drops them to
# 720p/CRF 30 with no audio — a few MB each — leaving the originals untouched.
#
# Usage: sbatch slurm/downscale_clips.sh runs/<match_id> [out_dir]

set -uo pipefail

RUN_DIR="${1:?usage: sbatch slurm/downscale_clips.sh runs/<match_id> [out_dir]}"
OUT_DIR="${2:-${RUN_DIR}/clips_720p}"

module load ffmpeg

mkdir -p "$OUT_DIR"
echo "Re-encoding $(ls "$RUN_DIR"/clips/*.mp4 | wc -l) clip(s) → $OUT_DIR"

# -P 8 matches --cpus-per-task; each ffmpeg is single-threaded enough that
# running eight of them beats giving one job eight threads.
ls "$RUN_DIR"/clips/*.mp4 | xargs -P 8 -I{} sh -c '
  out="'"$OUT_DIR"'/$(basename "{}")"
  [ -s "$out" ] && exit 0
  ffmpeg -nostdin -loglevel error -y -i "{}" \
    -vf scale=-2:720 -c:v libx264 -crf 30 -preset veryfast -an "$out"
'

echo "Done. $(du -sh "$OUT_DIR" | cut -f1) across $(ls "$OUT_DIR" | wc -l) clip(s)."
