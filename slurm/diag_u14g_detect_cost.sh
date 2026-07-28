#!/bin/bash
#SBATCH --job-name=sv_diag_cost
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:40:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/sv_diag_cost_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_diag_cost_%j.err

set -uo pipefail
cd /orange/ewhite/b.weinstein/soccer-video-analysis
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/blue/ewhite/b.weinstein/soccer-vision/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python slurm/diag_u14g_detect_cost.py
