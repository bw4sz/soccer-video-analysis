#!/bin/bash
#SBATCH --job-name=sv_identify
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_identify_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_identify_%j.err

# Name every track in a processed run — appearance re-id against the squad's
# gallery, with OCR picking up what the gallery abstains on.
#
# Sized off the U14G full match (17,395 lanes, ~164k crops off ~18k distinct
# frames). The cost here is **video decoding, not the model**: embedding is a
# batched forward on a 2.2M-parameter backbone, while the crops are scattered
# over a 2 GB long-GOP h264 file. `embed_tracks` now plans crops per frame and
# makes one forward pass (~66 min at 0.036 s/frame); the old seek-per-crop path
# would have cost ~50 h on this run, which is why this job did not exist before.
#
# GPU is for the embedding batches; 64GB is ample since crops are released as
# each track completes.
#
# Usage: sbatch slurm/submit_identify.sh <run_dir> [profile] [gallery] [method]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:?usage: sbatch slurm/submit_identify.sh <run_dir> [profile] [gallery] [method]}"
PROFILE="${2:-}"
GALLERY="${3:-}"
METHOD="${4:-reid}"

export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== soccer-vision identify ==="
echo "start:   $(date)"
echo "node:    $(hostname)"
echo "run:     $RUN_DIR"
echo "profile: ${PROFILE:-<none>}"
echo "gallery: ${GALLERY:-<from profile>}"
echo "method:  $METHOD"

ARGS=(--run "$RUN_DIR" --method "$METHOD" --device cuda)
[ -n "$PROFILE" ] && ARGS+=(--profile "$PROFILE")
[ -n "$GALLERY" ] && ARGS+=(--gallery "$GALLERY")

"$PY" -m soccer_vision.cli.main identify "${ARGS[@]}"
RC=$?

if [ $RC -eq 0 ]; then
  "$PY" - "$RUN_DIR" <<'PY'
import collections, json, sys
from pathlib import Path

doc = json.loads((Path(sys.argv[1]) / "jerseys.json").read_text())
tracks = doc["tracks"]
named = collections.Counter(
    r["name"] for r in tracks.values() if r.get("name")
)
print(f"\nnamed {sum(named.values())}/{len(tracks)} tracks")
for name, n in named.most_common():
    print(f"  {name}: {n}")
PY
fi

echo "done: $(date)  exit=$RC"
echo "run dir: $RUN_DIR"
exit $RC
