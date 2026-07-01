"""Smoke-test a recokick ball-action-spotting checkpoint.

Loads a pytorch-argus `.pth` from the vendored recokick repo and runs a single
forward pass on a random grayscale frame stack — no video, no VPF, no NDA data
required. Confirms the checkpoint is a valid argus MultiDimStacker and that the
architecture instantiates and produces per-class logits.

Usage:
    python scripts/bas_smoke_test.py \
        vendor/ball-action-spotting/data/gdrive_weights/action/experiments/\
action_sampling_weights_002/model-019-0.797827.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "vendor" / "ball-action-spotting"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to an argus .pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=256,
                        help="H=W of the random input (EfficientNet is size-agnostic)")
    args = parser.parse_args()

    if not VENDOR.exists():
        sys.exit(f"Vendored repo not found: {VENDOR}\n"
                 "Clone it: git clone https://github.com/recokick/ball-action-spotting "
                 f"{VENDOR}")
    # Make `import src...` resolve to the vendored recokick package, and register
    # BallActionModel with argus so load_model can reconstruct it.
    sys.path.insert(0, str(VENDOR))
    import argus  # noqa: E402
    import src.argus_models  # noqa: F401,E402  (registers BallActionModel)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = argus.load_model(str(args.checkpoint), device=args.device,
                             optimizer=None, loss=None)
    model.eval()

    nn_name, nn_params = model.params["nn_module"]
    num_frames = nn_params.get("num_frames", 15)
    num_classes = nn_params["num_classes"]
    print(f"  nn_module     : {nn_name}")
    print(f"  backbone      : {nn_params.get('model_name')}")
    print(f"  num_classes   : {num_classes}")
    print(f"  num_frames    : {num_frames}")
    n_params = sum(p.numel() for p in model.nn_module.parameters())
    print(f"  parameters    : {n_params/1e6:.2f}M")

    # (batch, num_frames grayscale frames, H, W)
    x = torch.rand(1, num_frames, args.size, args.size, device=args.device)
    with torch.no_grad():
        logits = model.nn_module(x)
    print(f"  input shape   : {tuple(x.shape)}")
    print(f"  output shape  : {tuple(logits.shape)}")

    assert logits.shape == (1, num_classes), \
        f"expected (1, {num_classes}), got {tuple(logits.shape)}"
    probs = torch.sigmoid(logits)[0]
    assert torch.isfinite(probs).all(), "non-finite outputs"
    print(f"  sigmoid range : [{probs.min():.4f}, {probs.max():.4f}]")
    print("\nPASS: checkpoint loads and forward pass produces valid per-class outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
