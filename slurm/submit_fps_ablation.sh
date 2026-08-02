#!/bin/bash
#SBATCH --job-name=sv_fps_abl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_fps_abl_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_fps_abl_%j.err

# Does detecting on every frame actually fix track fragmentation?
#
# `runs/saints-u14g-full` was detected at 5 fps and produced 17,395 lanes for 22
# players, median lane 2.0 s. Commit 9e7acfd moved the default to every frame and
# told ByteTrack the truth about its rate, but nothing has measured what that
# buys, and every downstream identity number we have was taken on the 5 fps run.
#
# This runs the SAME 3-min U14G clip through the CURRENT code twice — 30 fps and
# 5 fps, identical field cut, identical detector — so the only difference is the
# detection rate. `runs/u14g-smoke-rfdetr` is not a valid control: it predates the
# field-cut change (a3bb247), which alone moves detections per frame by ~36%.
#
# Reported per rate: lanes, lane-length distribution in *seconds* (not samples —
# 7 samples means 1.4 s at 5 fps and 0.23 s at 30), and how many lanes survive
# past 5 s, which is the length that matters for identity: a lane too short to
# carry a confident re-id vote can never be named, however good the gallery is.

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="$REPO/data/u14g_smoke180.mp4"
PROFILE="$REPO/examples/profiles/saints-u14g.yaml"
CONFIG="$REPO/examples/process_match.yaml"

export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== fps ablation ==="
echo "start: $(date)  node: $(hostname)"

for FPS in 30 5; do
  MATCH_ID="u14g-fps${FPS}"
  echo ""
  echo "--- detect_fps=$FPS -> runs/$MATCH_ID ---"
  ARGS=(--match-id "$MATCH_ID" --out-dir "$REPO/runs" --config "$CONFIG" --profile "$PROFILE")
  # 30 is native here, so pass it explicitly rather than relying on the default
  # meaning "native" — the log should state the rate it actually ran at.
  ARGS+=(--detect-fps "$FPS")
  "$PY" -m soccer_vision.cli.main process "$VIDEO" "${ARGS[@]}"
  echo "rc=$? at $(date)"
done

echo ""
echo "=== comparison ==="
"$PY" - <<'PY'
import json
from pathlib import Path

import numpy as np

print(f"{'rate':>6} {'lanes':>7} {'med_s':>7} {'p90_s':>7} {'>=2s':>6} {'>=5s':>6} "
      f"{'det/frm':>8} {'ball%':>7}")
for fps in (30, 5):
    run = Path(f"/orange/ewhite/b.weinstein/soccer-video-analysis/runs/u14g-fps{fps}")
    d = json.loads((run / "tracks.json").read_text())
    interval = int(d.get("sample_interval", 1)) or 1
    eff = d["fps"] / interval
    lens = np.array([len(v) for v in d["tracks"].values()]) / eff   # seconds
    ball = json.loads((run / "ball_track.json").read_text())["samples"]
    vis = sum(1 for s in ball if s["visible"])
    n_frames = len({s["frame"] for s in ball})
    det = sum(len(v) for v in d["tracks"].values()) / max(1, n_frames)
    print(f"{eff:>6.1f} {len(lens):>7} {np.median(lens):>7.2f} "
          f"{np.percentile(lens, 90):>7.2f} {int((lens >= 2).sum()):>6} "
          f"{int((lens >= 5).sum()):>6} {det:>8.1f} {100 * vis / max(1, len(ball)):>6.1f}%")
PY

echo ""
echo "done: $(date)"
