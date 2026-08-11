#!/bin/bash
#SBATCH --job-name=sv_bench_sam
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
#SBATCH --output=/home/b.weinstein/logs/sv_bench_sam_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_bench_sam_%j.err

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/orange/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

export HF_HOME=/orange/ewhite/b.weinstein/soccer-vision/hf_cache
export TORCH_HOME=/orange/ewhite/b.weinstein/soccer-vision/torch_cache

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== SAM speed benchmark ==="
echo "start: $(date)"
$PY -u "$REPO/slurm/bench_sam_speed.py"
echo "done:  $(date)"
