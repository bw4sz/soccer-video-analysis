#!/bin/bash
#SBATCH --job-name=sv_identify_assign
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
#SBATCH --output=/home/b.weinstein/logs/sv_identify_assign_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_identify_assign_%j.err

# First `identify` with naming solved as an **assignment** rather than as twelve
# independent contests: one player holds one lane at a time, and the margin is
# measured against the identities still *available* to a lane instead of against
# whoever happens to sit second in the gallery.
#
# Why this and not a threshold change. Morgan is ranked first on 138 of 541
# eligible lanes -- more than any other player -- and named on one, because the
# four wide-net players all score ~0.64 against every lane and separate at the
# third decimal. Median margin is 0.009 against a gate of 0.05, i.e. the gate
# sits above the p90 of the distribution it filters. A per-lane margin measures
# how crowded the gallery is, not how confident the match is.
#
# Simulated from saved embeddings before building this (541 eligible lanes,
# gallery saints-u14g.fullmatch.npz), worst = most lanes carrying one name at the
# same instant:
#
#   margin  |  per-lane: named / worst  |  assignment: named / worst
#   0.050   |        19 / 2             |        19 / 1
#   0.030   |        45 / 2             |        81 / 1
#   0.020   |       105 / 4             |       200 / 1
#   0.010   |       244 / 9             |       255 / 1
#   0.000   |       541 / 18            |       278 / 1
#
# Two things to read there. The constraint holds concurrency at 1 **at every
# margin**, so the collision that made the halo strobe cannot come back through a
# looser threshold. And naming *saturates* around 270 as the margin goes to zero
# -- the constraint, not the threshold, becomes the binding rule, which is the
# whole argument for moving the decision here.
#
# Concurrency of 1 is true by construction and is NOT evidence of correctness.
# It removes the one error mode measurable without labels. Whether the names are
# right still needs `slurm/validate_reid_frames.py` and watching a reel.
#
# This run uses the default margin 0.05 so the diff against job 38746007 isolates
# the constraint. It also writes **reid_scores.npz**, the per-lane score vector
# over every gallery identity, so the assignment can be re-solved at any margin
# afterwards with no GPU -- the embedding pass is the expensive part and it only
# needs doing once.
#
# Usage: sbatch slurm/submit_identify_assign.sh [run_dir] [gallery] [profile] [kit]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

RUN_DIR="${1:-runs/saints-u14g-full-30fps}"
GALLERY="${2:-galleries/saints-u14g.fullmatch-neg64.npz}"
PROFILE="${3:-examples/profiles/saints-u14g.yaml}"
KIT="${4:-black}"
BASELINE="$RUN_DIR/jerseys.pre-assign.json"

export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== identify with one-player-one-lane assignment ==="
echo "start:   $(date)"
echo "node:    $(hostname)"
echo "run:     $RUN_DIR"
echo "gallery: $GALLERY"
echo "kit:     $KIT"

cp "$RUN_DIR/jerseys.json" "$BASELINE" && echo "baseline saved: $BASELINE"

"$PY" -m soccer_vision.cli.main identify \
  --run "$RUN_DIR" --method reid --gallery "$GALLERY" \
  --profile "$PROFILE" --team "$KIT" --device cuda || exit 1

echo
echo "=== what the constraint changed ==="
"$PY" - "$RUN_DIR" "$BASELINE" <<'PYEOF'
import collections, json, sys
run, base_path = sys.argv[1], sys.argv[2]
new = json.load(open(f"{run}/jerseys.json"))["tracks"]
base = json.load(open(base_path))["tracks"]
td = json.load(open(f"{run}/tracks.json")); fps = td["fps"]

def worst_concurrency(d):
    lanes = collections.defaultdict(list)
    for tid, v in d.items():
        if v.get("name") and tid in td["tracks"]:
            fr = [s["frame"] for s in td["tracks"][tid]]
            lanes[v["name"]].append((min(fr) / fps, max(fr) / fps))
    out = {}
    for name, spans in lanes.items():
        edges = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
        live = peak = 0
        for _, delta in edges:
            live += delta
            peak = max(peak, live)
        out[name] = peak
    return out

for tag, d in (("per-lane (before)", base), ("assignment (after)", new)):
    named = {t: v for t, v in d.items() if v.get("name")}
    wc = worst_concurrency(d)
    over = {k: v for k, v in wc.items() if v > 1}
    print(f"\n{tag}: {len(named)} named lanes")
    print(f"  worst lanes sharing one name at an instant: {max(wc.values(), default=0)}")
    print(f"  names landing on 2+ concurrent lanes: {len(over)}")
    if over:
        print("   ", dict(sorted(over.items(), key=lambda kv: -kv[1])[:6]))
    for n, k in collections.Counter(v["name"] for v in named.values()).most_common():
        print(f"    {n:24s} {k}")

opened = sum(1 for v in new.values()
             if v.get("name") and (v.get("open_margin") is not None)
             and v["margin"] > v["open_margin"])
print(f"\n{opened} lane(s) were named on a margin the constraint opened up")
PYEOF

echo
echo "done: $(date)  exit=$?"
echo "scores kept at: $RUN_DIR/reid_scores.npz  (re-solve any margin, no GPU)"
