#!/bin/bash
#SBATCH --job-name=sv_det_eval
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
#SBATCH --output=/home/b.weinstein/logs/sv_det_eval_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_det_eval_%j.err

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

# Both facebook/sam3 and julianzu9612/RFDETR-Soccernet are already cached here
# and past their gates, so the job loads fully offline (no token, no network).
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== FOOTPASS in-domain detector eval ==="
echo "start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY -u "$REPO/slurm/eval_detector_footpass.py" \
    --game "${1:-game_24_H1}" \
    --windows 10 \
    --window-frames 40 \
    --arms sam3,rfdetr
RC=$?
echo "done:  $(date) (rc=$RC)"
exit $RC
