#!/bin/bash
#SBATCH --job-name=sv_sam3_val
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
#SBATCH --output=/home/b.weinstein/logs/sv_sam3_val_%j.out
#SBATCH --error=/home/b.weinstein/logs/sv_sam3_val_%j.err

set -uo pipefail

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

# SAM3 weights live in the DEFAULT hf cache (not the soccer-vision hf_cache),
# already downloaded + past the gate. Load fully offline -> no token needed.
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load ffmpeg/4.3.1 2>/dev/null || module load ffmpeg 2>/dev/null || true

echo "=== SAM3 text-prompt validation ==="
echo "start: $(date)"
$PY -u "$REPO/slurm/validate_sam3.py"
RC=$?
echo "done:  $(date) (rc=$RC)"
exit $RC
