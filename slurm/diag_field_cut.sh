#!/bin/bash
#SBATCH --job-name=sv_diag_fieldcut
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:45:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_diag_fieldcut_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_diag_fieldcut_%j.err

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== field cut: old centred rectangle vs top-only ==="
echo "start: $(date)"
echo "node:  $(hostname)"

"$PY" slurm/diag_field_cut.py


  

echo "done: $(date)"
