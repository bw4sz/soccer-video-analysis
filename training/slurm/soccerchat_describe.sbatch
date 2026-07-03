#!/bin/bash
#SBATCH --job-name=sv_soccerchat
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite

# Run SoccerChat (Qwen2-VL-7B + LoRA) over a processed run's event clips.
# Weights (~16 GB) are pulled from HuggingFace on first use into HF_HOME.
#
# Usage:
#   sbatch training/slurm/soccerchat_describe.sbatch runs/<match_id> [LIMIT]
# e.g. a quick 5-clip smoke test to gauge youth-footage transfer:
#   sbatch training/slurm/soccerchat_describe.sbatch runs/abc123 5

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/soccer-video-analysis
cd "$REPO"

RUN="${1:?usage: sbatch soccerchat_describe.sbatch <run-dir> [limit]}"
LIMIT="${2:-0}"

# Keep the big model cache on /blue, not the tiny home quota.
export HF_HOME=/blue/ewhite/b.weinstein/soccer-vision/hf_cache
mkdir -p "$HF_HOME"

# Install the soccerchat extra into the project venv if not already present.
uv sync --project "$REPO" --extra soccerchat

LIMIT_ARG=()
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

uv run --project "$REPO" soccer-vision describe --run "$RUN" "${LIMIT_ARG[@]}"

echo "Done. Review with:  soccer-vision annotate --run $RUN"
