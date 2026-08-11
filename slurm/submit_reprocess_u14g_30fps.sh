#!/bin/bash
#SBATCH --job-name=sv_u14g_30
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192GB
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_u14g_30_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_u14g_30_%j.err

# Re-process the U14G full match at native frame rate, with the single-forward
# detector.
#
# Two reasons this run exists:
#
#   1. **Every identity number we have is measured on a degraded substrate.**
#      `runs/saints-u14g-full` was detected at 5 fps. Job 38526638 showed native
#      rate puts 75% more of the match under a track and 2.5x more of that time
#      into lanes >=10 s (longest lane 114 s vs 27.6 s). Lane length is what
#      gates identity: a lane too short to carry a confident re-id vote can never
#      be named however good the gallery is. So the linking sweep, the naming
#      yield and the 9.4% on-ball coverage all need re-measuring here, not there.
#
#   2. **It was unaffordable until now.** `process` ran the full RF-DETR forward
#      twice per frame — once at the player threshold, once inside
#      `detect_ball_position` at the looser ball threshold. Native rate cost
#      ~4.3x realtime because of it. `predict_split` does both classes in one
#      pass, which is what makes a 60-min match at 30 fps a reasonable ask.
#
# STAGE 1 guards STAGE 2: re-run the 3-min clip that job 38526638 already
# processed at 30 fps and require byte-identical tracks. `predict_split` is meant
# to be *exactly* the two calls it replaces (scores don't depend on the threshold
# they were requested at), so anything other than an exact match means the
# refactor changed detections and the full match must not run on it.
#
# 192GB, not 128: per-frame track boxes are held for the whole run, and at 6x the
# detection rate there are ~6x as many. Job 38313387 was OOM-killed at 64GB on the
# 5 fps version of this same match.

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO="$REPO/data/wfc-rangers-vs-saints-pcu-cup-2026-07-11.mp4"
PROFILE="$REPO/examples/profiles/saints-u14g.yaml"
CONFIG="$REPO/examples/process_match.yaml"
MATCH_ID=saints-u14g-full-30fps

export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== U14G full match @ native rate, single-forward detector ==="
echo "start: $(date)  node: $(hostname)"

# ---------------------------------------------------------------- stage 1
echo ""
echo "--- STAGE 1: equivalence check on the 3-min clip ---"
T0=$SECONDS
"$PY" -m soccer_vision.cli.main process "$REPO/data/u14g_smoke180.mp4" \
    --match-id u14g-fps30-onepass --out-dir "$REPO/runs" \
    --config "$CONFIG" --profile "$PROFILE" --detect-fps 30
SMOKE_RC=$?
SMOKE_S=$((SECONDS - T0))
echo "stage 1 exit=$SMOKE_RC in ${SMOKE_S}s (two-forward baseline was 780s)"

if [ $SMOKE_RC -ne 0 ]; then
  echo "ABORT: smoke process failed"; exit $SMOKE_RC
fi

"$PY" - <<'PY'
import json
import sys
from pathlib import Path

runs = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/runs")
old = json.loads((runs / "u14g-fps30" / "tracks.json").read_text())
new = json.loads((runs / "u14g-fps30-onepass" / "tracks.json").read_text())

problems = []
if old["tracks"].keys() != new["tracks"].keys():
    problems.append(f"lane ids differ: {len(old['tracks'])} vs {len(new['tracks'])}")
else:
    for tid, samples in old["tracks"].items():
        if samples != new["tracks"][tid]:
            problems.append(f"lane {tid} differs")
            if len(problems) > 5:
                break

ob = json.loads((runs / "u14g-fps30" / "ball_track.json").read_text())["samples"]
nb = json.loads((runs / "u14g-fps30-onepass" / "ball_track.json").read_text())["samples"]
if ob != nb:
    diff = sum(1 for a, b in zip(ob, nb) if a != b)
    problems.append(f"ball track differs on {diff}/{len(ob)} samples")

if problems:
    print("NOT EQUIVALENT:")
    for p in problems[:8]:
        print(f"  - {p}")
    sys.exit(1)
print(f"equivalent: {len(new['tracks'])} lanes and {len(nb)} ball samples identical")
PY
if [ $? -ne 0 ]; then
  echo "ABORT: predict_split changed detections — not running the full match."
  exit 1
fi

# ---------------------------------------------------------------- stage 2
echo ""
echo "--- STAGE 2: full match -> runs/$MATCH_ID ---"
T0=$SECONDS
"$PY" -m soccer_vision.cli.main process "$VIDEO" \
    --match-id "$MATCH_ID" --out-dir "$REPO/runs" \
    --config "$CONFIG" --profile "$PROFILE" --detect-fps 30
RC=$?
echo "stage 2 exit=$RC in $((SECONDS - T0))s"

if [ $RC -eq 0 ]; then
  echo ""
  echo "=== 5 fps (saints-u14g-full) vs 30 fps (this run) ==="
  "$PY" - <<'PY'
import json
from pathlib import Path

import numpy as np

runs = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/runs")
print(f"{'run':<26}{'rate':>6}{'lanes':>8}{'tracked_s':>11}{'>=10s share':>13}{'longest':>9}")
for name in ("saints-u14g-full", "saints-u14g-full-30fps"):
    d = json.loads((runs / name / "tracks.json").read_text())
    interval = int(d.get("sample_interval", 1)) or 1
    eff = d["fps"] / interval
    lens = np.array([len(v) for v in d["tracks"].values()]) / eff
    tot = lens.sum()
    share = lens[lens >= 10].sum() / max(tot, 1e-9)
    print(f"{name:<26}{eff:>6.1f}{len(lens):>8}{tot:>11.0f}{share:>12.1%}{lens.max():>9.1f}")
PY
fi

echo ""
echo "done: $(date)  exit=$RC"
echo "NEXT: re-run the link-gate sweep and the identity measurements against"
echo "      runs/$MATCH_ID — the 5 fps numbers do not transfer."
exit $RC
