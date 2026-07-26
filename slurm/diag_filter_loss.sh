#!/bin/bash
#SBATCH --job-name=sv_filt_diag
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:30:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_filt_diag_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_filt_diag_%j.err
set -uo pipefail
REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true
echo "start: $(date)"
$PY -u "$REPO/slurm/diag_filter_loss.py"
echo "done: $(date)"
