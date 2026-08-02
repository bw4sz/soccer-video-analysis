#!/bin/bash
#SBATCH --job-name=sv_link
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=02:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_link_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_link_%j.err

# Rejoin fragmented lanes, with the appearance veto enabled. The veto is the
# expensive part: it embeds both edges of every lane (2 crops x ~17k lanes) off
# the proxy, which is the same decode-bound pass `identify` makes.
set -uo pipefail
REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"
RUN_DIR="${1:?usage: sbatch slurm/submit_link_tracks.sh <run_dir> [max_gap] [max_dist] [min_appearance]}"
GAP="${2:-3.0}"; DIST="${3:-250}"; MINAPP="${4:-0.70}"
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1
module load ffmpeg/4.3.1 2>/dev/null || true
echo "start: $(date)  node: $(hostname)"
"$PY" -m soccer_vision.cli.main link-tracks --run "$RUN_DIR" \
  --max-gap "$GAP" --max-dist "$DIST" --appearance --min-appearance "$MINAPP" \
  --device cuda
RC=$?
echo "done: $(date) exit=$RC"
exit $RC
