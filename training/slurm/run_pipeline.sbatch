#!/bin/bash
#SBATCH --job-name=sv_pipeline
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ben.weinstein@weecology.org
#SBATCH --account=ewhite

# Run full soccer-vision pipeline on a match video.
# Single-GPU: RF-DETR inference only (~140ms/frame on L4).
#
# Usage:
#   sbatch run_pipeline.sbatch /path/to/match.mp4
#   sbatch run_pipeline.sbatch /path/to/match.mp4 /path/to/output_dir

REPO=/orange/ewhite/b.weinstein/soccer-video-analysis
PY=/blue/ewhite/b.weinstein/envs/soccer-vision/bin/python
cd "$REPO"

VIDEO=${1:?Usage: sbatch run_pipeline.sbatch VIDEO [OUTPUT_DIR]}
OUTPUT_DIR=${2:-/blue/ewhite/b.weinstein/soccer-vision-data/runs/$(basename "${VIDEO%.*}")}

"$PY" -m soccer_vision.cli.process "$VIDEO" --output-dir "$OUTPUT_DIR"
