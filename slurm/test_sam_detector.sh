#!/bin/bash
#SBATCH --job-name=sv_saints_sam
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_saints_sam_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_saints_sam_%j.err

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="$REPO/data/SaintsU11_OVF_Jul192026.MP4"
MATCH_ID="saints-u11-ovf-2026-07-19-sam"
RUN_DIR="$REPO/runs/$MATCH_ID"
CONFIG="$REPO/examples/saints-u11-sam.yaml"

export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

{
  echo "=== Saints U11 with SAM Player Detector ==="
  echo "start:    $(date)"
  echo "video:    $VIDEO"
  echo "match_id: $MATCH_ID"
  echo "config:   $CONFIG"
  echo ""
  echo "Installing segment-anything (first run only)..."
} | tee "$REPO/slurm/logs/saints_sam_status.txt"

# Verify SAM is installed
$PY -c "import segment_anything; print('SAM ready')" || {
  echo "ERROR: segment-anything not installed. Run: python -m pip install segment-anything"
  exit 1
}

echo ""
echo "[1/1] process with SAM detector..."
$PY -u -m soccer_vision.cli.main process "$VIDEO" \
  --match-id "$MATCH_ID" \
  --out-dir "$REPO/runs" \
  --config "$CONFIG" \
  --device cuda

RC=$?
echo ""
if [ $RC -eq 0 ]; then
  echo "✓ Process completed successfully"
  $PY << 'PY'
import json
from pathlib import Path

run = Path('runs/saints-u11-ovf-2026-07-19-sam')
stats = json.loads((run / 'stats.json').read_text())

print("\n" + "=" * 70)
print("SAM RESULTS")
print("=" * 70)
print(f"Total events: {stats['total_events']}")

throws = stats['event_counts_by_team'].get('throw_in', {})
unknown = throws.get('unknown', 0)
classified = stats['event_counts'].get('throw_in', 0) - unknown
total = stats['event_counts'].get('throw_in', 0)
pct = 100 * classified / total if total > 0 else 0

print(f"\nThrow-in team classification:")
print(f"  Unknown: {unknown}")
print(f"  Blue:    {throws.get('blue', 0)}")
print(f"  White:   {throws.get('white', 0)}")
print(f"  Classified: {classified}/{total} ({pct:.1f}%)")

print(f"\n✓ Comparison to baseline (RF-DETR conf=0.3):")
print(f"  Before: 8/199 (4%)")
print(f"  After:  {classified}/{total} ({pct:.1f}%)")
print(f"  Δ:      +{classified - 8} events (+{pct - 4:.1f}pp)")

if classified > 100:
    print("\n✅ SAM WORKS! Team classification significantly improved")
else:
    print(f"\n⚠️  SAM result ({classified}/199) - still needs investigation")

print("=" * 70)
PY
else
  echo "✗ Process failed (exit=$RC)"
fi

echo ""
echo "done: $(date)"
echo "run dir: $RUN_DIR"
exit $RC
