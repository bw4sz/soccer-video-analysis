#!/bin/bash
#SBATCH --job-name=sv_sam3_smoke
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:45:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_sam3_smoke_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_sam3_smoke_%j.err

set -uo pipefail
REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

CLIP=/orange/ewhite/b.weinstein/soccer-video-analysis/data/saints_u11_smoke60.mp4
SRC=/orange/ewhite/b.weinstein/soccer-video-analysis/data/SaintsU11_OVF_Jul192026.MP4

# 60s of live play starting at t=500s (the validated window), cut once.
if [ ! -f "$CLIP" ]; then
  echo "[cut] extracting 60s live-play clip..."
  ffmpeg -y -loglevel error -ss 500 -i "$SRC" -t 60 -c copy "$CLIP"
fi
echo "clip: $(ls -lh $CLIP | awk '{print $5}')"

echo "start: $(date)"
$PY -u -m soccer_vision.cli.main process "$CLIP" \
  --match-id sam3-full \
  --out-dir "$REPO/runs" \
  --config "$REPO/examples/saints-u11-sam3.yaml" \
  --device cuda
RC=$?
echo "process rc=$RC"

if [ $RC -eq 0 ]; then
  $PY - <<'PYEOF'
import json
from pathlib import Path
run = Path('/orange/ewhite/b.weinstein/soccer-video-analysis/runs/sam3-full')
stats = json.loads((run / 'stats.json').read_text())
print("\n" + "=" * 60)
print("SAM3 PIPELINE SMOKE — 60s live clip")
print("=" * 60)
print("total_events:", stats.get("total_events"))
tracks = json.loads((run / 'tracks.json').read_text())["tracks"]
print("tracks:", len(tracks))
by_team = stats.get("event_counts_by_team", {})
print("event_counts:", stats.get("event_counts"))
print("event_counts_by_team:", json.dumps(by_team, indent=2)[:800])
# the metric that matters: how many events carry a real team?
tot = unk = 0
for ev, teams in by_team.items():
    for t, n in teams.items():
        tot += n
        if t == "unknown":
            unk += n
print(f"\nteam-classified events: {tot - unk}/{tot} "
      f"({100*(tot-unk)/tot if tot else 0:.1f}%)   [baseline RF-DETR: 3.5%]")
print("=" * 60)
PYEOF
fi
echo "done: $(date)"
exit $RC
