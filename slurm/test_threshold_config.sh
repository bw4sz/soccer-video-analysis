#!/bin/bash
#SBATCH --job-name=sv_saints_threshold_config
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
#SBATCH --output=/home/b.weinstein/logs/sv_saints_threshold_config_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_saints_threshold_config_%j.err

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="$REPO/data/SaintsU11_OVF_Jul192026.MP4"
MATCH_ID="saints-u11-ovf-2026-07-19-conf-0.15"
RUN_DIR="$REPO/runs/$MATCH_ID"
CONFIG="$REPO/examples/saints-u11-0.15-threshold.yaml"

export HF_HOME=/blue/ewhite/b.weinstein/soccer-vision/hf_cache
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

{
  echo "=== Saints U11 with conf_threshold=0.15 ==="
  echo "start:    $(date)"
  echo "video:    $VIDEO"
  echo "match_id: $MATCH_ID"
  echo "config:   $CONFIG"
  echo "Expected: ~25 players vs 19 at default 0.3"
} | tee "$REPO/slurm/logs/saints_conf_status.txt"

echo ""
echo "[1/1] process with conf_threshold=0.15 ..."
python3 -u -m soccer_vision.cli.main process "$VIDEO" \
  --match-id "$MATCH_ID" \
  --out-dir "$REPO/runs" \
  --config "$CONFIG" \
  --device cuda

RC=$?
echo ""
echo "Exit code: $RC"
if [ $RC -eq 0 ]; then
  echo "✓ Process completed successfully"
  echo ""
  echo "Results saved to: $RUN_DIR"
  python3 << 'PY'
import json
from pathlib import Path
run = Path('runs') / 'saints-u11-ovf-2026-07-19-conf-0.15'
if (run / 'stats.json').exists():
    stats = json.loads((run / 'stats.json').read_text())
    print(f"  Events: {stats.get('total_events', 0)}")
    print(f"  Event team breakdown: {stats.get('event_counts_by_team', {})}")
    print(f"  Teams detected: {stats.get('teams', {})}")
PY
else
  echo "✗ Process failed (exit=$RC)"
fi

echo ""
echo "done: $(date)"
exit $RC
