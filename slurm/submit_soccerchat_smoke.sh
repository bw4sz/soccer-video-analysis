#!/bin/bash
#SBATCH --job-name=sv_soccerchat_smoke
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_soccerchat_smoke_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_soccerchat_smoke_%j.err

# Smoke-test SoccerChat on youth footage: cut a few 10s clips from the match and
# run the VLM (classify + caption) on each. Exercises the real model-load +
# ms-swift inference path from soccer_vision.verify.soccerchat.
#
# Usage: sbatch slurm/submit_soccerchat_smoke.sh [VIDEO] [N_CLIPS]

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
cd "$REPO"

VIDEO="${1:-$REPO/data/match-saints-16b-pre-mls-next-2026-04-26.mp4}"
N="${2:-6}"

# Cached weights live here (already downloaded); keep them off the home quota.
export HF_HOME=/blue/ewhite/b.weinstein/soccer-vision/hf_cache
# Build/run in an ISOLATED env so the dev .venv (base/gpu extras) is untouched.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/soccer-vision/venv_soccerchat

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO/slurm/logs/soccerchat_smoke_$TS"
mkdir -p "$LOG_DIR"

{
  echo "=== SoccerChat smoke ==="
  echo "start:  $(date)"
  echo "node:   $(hostname)"
  echo "gpu:    ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "video:  $VIDEO"
  echo "clips:  $N"
  echo "env:    $UV_PROJECT_ENVIRONMENT"
} | tee "$LOG_DIR/status.txt"
nvidia-smi 2>&1 | tee -a "$LOG_DIR/status.txt" || true

echo "[1/2] Syncing soccerchat env (ms-swift, transformers, decord, av)..." | tee -a "$LOG_DIR/status.txt"
uv sync --extra soccerchat 2>&1 | tee "$LOG_DIR/uv_sync.log"

echo "[2/2] Running SoccerChat on $N clips..." | tee -a "$LOG_DIR/status.txt"
uv run --extra soccerchat python scripts/soccerchat_smoke.py \
  --video "$VIDEO" --n "$N" --out "$LOG_DIR/smoke" 2>&1 | tee "$LOG_DIR/smoke.log"
RC=${PIPESTATUS[0]}

echo "done: $(date)  exit=$RC" | tee -a "$LOG_DIR/status.txt"
echo "Results: $LOG_DIR/smoke/results.json"
exit "$RC"
