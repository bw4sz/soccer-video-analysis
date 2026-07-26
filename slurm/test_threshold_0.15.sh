#!/bin/bash
#SBATCH --job-name=sv_saints_threshold_test
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=01:30:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_saints_threshold_0.15_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_saints_threshold_0.15_%j.err

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="$REPO/data/SaintsU11_OVF_Jul192026.MP4"
MATCH_ID="saints-u11-ovf-2026-07-19-threshold-0.15"
RUN_DIR="$REPO/runs/$MATCH_ID"

export HF_HOME=/blue/ewhite/b.weinstein/soccer-vision/hf_cache
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== Threshold test: conf=0.15 ==="
echo "start:    $(date)"
echo "video:    $VIDEO"
echo "match_id: $MATCH_ID"

"$PY" -m soccer_vision.cli.main process "$VIDEO" \
  --match-id "$MATCH_ID" \
  --out-dir "$REPO/runs" \
  --device cuda

echo "done: $(date)"
echo "run dir: $RUN_DIR"
echo ""
echo "Compare diagnostics:"
echo "  Original (conf=0.3): runs/saints-u11-ovf-2026-07-19/diagnostics/team_colour_frame17376.png"
echo "  New (conf=0.15):     runs/$MATCH_ID/diagnostics/team_colour_frame17376.png"
