#!/bin/bash
# Build a dedicated env for FOOTPASS / TAAD training (see training/FOOTPASS.md).
# Kept separate from the soccer-vision env (torch 2.12) because FOOTPASS pins the
# older stack (py3.11 / torch 2.1) and depends on pytorchvideo (X3D backbone).
#
#   bash scripts/setup_footpass_env.sh
#
# Then: sbatch training/slurm/train_footpass_taad.sbatch  (uses $FOOTPASS_PY).
set -euo pipefail

ENV_DIR=${FOOTPASS_ENV:-/blue/ewhite/b.weinstein/envs/footpass}
UV=${UV:-uv}

echo "Creating $ENV_DIR (python 3.11)..."
"$UV" venv --python 3.11 "$ENV_DIR"
PY="$ENV_DIR/bin/python"

# Torch 2.1 (cu121 wheels; forward-compatible with the newer HPG driver).
"$UV" pip install --python "$PY" \
  torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# FOOTPASS runtime deps (README "Getting Started" + train-script imports).
"$UV" pip install --python "$PY" \
  "numpy==1.26.*" "albumentations==2.0.8" opencv-python-headless "h5py>=3.14" \
  decord tensorboard tqdm \
  pytorchvideo fvcore iopath

# GNN variant only (optional; comment out if unused):
# "$UV" pip install --python "$PY" torch_geometric==2.6.1

echo
echo "Verifying imports + X3D backbone load..."
"$PY" - <<'PYEOF'
import torch, torchvision, decord, h5py, albumentations, cv2
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| cuda avail", torch.cuda.is_available())
print("decord", decord.__version__, "| h5py", h5py.__version__,
      "| albumentations", albumentations.__version__)
# X3D-S is pulled from torch.hub on first use (needs pytorchvideo).
m = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
n = sum(p.numel() for p in m.parameters())
print(f"x3d_s loaded OK: {n/1e6:.1f}M params")
PYEOF

echo
echo "Done. Set FOOTPASS_PY=$PY for the SLURM job (default already points here)."
