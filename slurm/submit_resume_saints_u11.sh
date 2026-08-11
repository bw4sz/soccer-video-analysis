#!/bin/bash
#SBATCH --job-name=sv_saints_u11_resume
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:30:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_saints_u11_resume_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_saints_u11_resume_%j.err

# RESUME the Saints U11 pathway after job 37591635 crashed at stage 2.
# `process` (stage 1) already produced tracks.json / annotations.json / clips /
# broadcast_proxy.mp4 in the run dir on 2026-07-19 — we reuse all of it and only
# run the tail: identify -> summary -> extract. The crash was a missing `nltk`
# (PARSeq import chain); now installed into the env. Nothing from process is recomputed.
#
# Usage: sbatch slurm/submit_resume_saints_u11.sh

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

MATCH_ID="saints-u11-ovf-2026-07-19"
RUN_DIR="$REPO/runs/$MATCH_ID"

export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

{
  echo "=== saints-u11 RESUME (stages 2-4) ==="
  echo "start:    $(date)"
  echo "node:     $(hostname)"
  echo "gpu:      ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "run_dir:  $RUN_DIR"
} | tee "$REPO/slurm/logs/saints_u11_resume_status.txt"

if [ ! -f "$RUN_DIR/tracks.json" ]; then
  echo "ERROR: no tracks.json in $RUN_DIR — process stage output missing, cannot resume."
  exit 2
fi

echo ""
echo "[2/4] identify (jersey numbers) ..."
"$PY" -m soccer_vision.cli.main identify --run "$RUN_DIR"
RC=$?
if [ $RC -ne 0 ]; then echo "identify failed (rc=$RC)"; exit $RC; fi

echo ""
echo "[3/4] summary — #6 / black events by label (pre-render):"
"$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter

run = Path(sys.argv[1])
jerseys = json.loads((run / "jerseys.json").read_text())
tracks = jerseys.get("tracks", {})   # jerseys.json nests per-track records under "tracks"

# Jersey-vote yield. Events in annotations.json carry no track_id, so #6-by-event
# can't be joined here — stage 4 `extract --number 6` does that association via
# tracks.json and is the real answer. Here we just report OCR yield.
votes = Counter(str(r.get("jersey")) for r in tracks.values())
legible = [r for r in tracks.values() if r.get("jersey") is not None]
six = {tid: r for tid, r in tracks.items() if str(r.get("jersey")) == "6"}
print(f"  tracks total:        {len(tracks)}")
print(f"  tracks with a number:{len(legible)}  (unknown: {votes.get('None', 0)})")
print(f"  jersey vote counts:  {dict(votes)}")
print(f"  tracks voting #6:    {sorted(six, key=lambda t: int(t))}")
for tid, r in sorted(six.items(), key=lambda kv: int(kv[0])):
    print(f"    track {tid}: conf={r.get('confidence')} n_obs={r.get('n_obs')} "
          f"legible_frac={r.get('legible_frac')} name={r.get('name')}")
PY

echo ""
echo "[4/4] extract — render #6 / black clips with halo ..."
"$PY" -m soccer_vision.cli.main extract --run "$RUN_DIR" \
  --team black --number 6 --halo
RC=$?

echo ""
echo "clips in $RUN_DIR/clips:"
ls -la "$RUN_DIR/clips" 2>/dev/null | tail -30

echo ""
echo "done: $(date)  exit=$RC"
echo "run dir: $RUN_DIR"
exit $RC
