"""Train a team ball action spotting model using sn-teamspotting (T-DEED baseline).

T-DEED localizes which team performed each ball action — a finer-grained
task than classic action spotting. Trained on SoccerNet Ball Action Spotting
2025 data.

Prerequisites:
  pip install SoccerNet
  git clone https://github.com/SoccerNet/sn-teamspotting.git

Usage:
  # Step 1: Download data
  python train_teamspotting.py --download-only --data-dir data/sn-bas

  # Step 2: Extract features (requires video files)
  python train_teamspotting.py --extract-features --data-dir data/sn-bas

  # Step 3: Train T-DEED
  python train_teamspotting.py --train --data-dir data/sn-bas --epochs 30

  # Step 4: Export for soccer-vision
  python train_teamspotting.py --export --checkpoint runs/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TEAM_ACTION_LABELS = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT",
    "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL",
]

TEAM_LABEL_TO_SV = {
    "PASS": "pass",
    "DRIVE": "drive",
    "HEADER": "header",
    "HIGH PASS": "pass",
    "OUT": "ball_out",
    "CROSS": "cross",
    "THROW IN": "throw_in",
    "SHOT": "shot",
    "BALL PLAYER BLOCK": "block",
    "PLAYER SUCCESSFUL TACKLE": "tackle",
    "FREE KICK": "free_kick",
    "GOAL": "goal",
}


def download_data(data_dir: Path):
    """Download SoccerNet Ball Action Spotting data."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        sys.exit("Install SoccerNet SDK: pip install SoccerNet")

    downloader = SoccerNetDownloader(LocalDirectory=str(data_dir))

    print("Downloading SoccerNet Ball Action Spotting labels...")
    downloader.downloadDataTask(
        task="spotting-ball-2024",
        split=["train", "valid", "test", "challenge"],
    )
    print(f"Data saved to {data_dir}")


def extract_features(data_dir: Path, output_dir: Path):
    """Extract video features for T-DEED training.

    This requires the actual SoccerNet match videos, which are large
    and require a NDA-signed download.
    """
    print("Feature extraction for T-DEED requires:")
    print("  1. SoccerNet match videos (download via SoccerNet SDK with password)")
    print("  2. A ResNet or InternVideo feature extractor")
    print()
    print("Steps:")
    print(f"  cd sn-teamspotting/")
    print(f"  python extract_features.py --data_dir {data_dir} --output {output_dir}")
    print()
    print("If you have pre-extracted features, place them in:")
    print(f"  {output_dir}/{{split}}/{{game}}/features.npy")


def train_tdeed(data_dir: Path, output_dir: Path, epochs: int = 30):
    """Train T-DEED model."""
    print("Training T-DEED for team ball action spotting...")
    print()
    print("If sn-teamspotting is installed:")
    print(f"  cd sn-teamspotting/")
    print(f"  python main.py --data_dir {data_dir} \\")
    print(f"    --output_dir {output_dir} \\")
    print(f"    --epochs {epochs} \\")
    print(f"    --model T-DEED \\")
    print(f"    --batch_size 8 \\")
    print(f"    --lr 1e-4")
    print()

    sn_repo = Path("sn-teamspotting")
    if not sn_repo.exists():
        print("sn-teamspotting repo not found. Clone it:")
        print("  git clone https://github.com/SoccerNet/sn-teamspotting.git")
        return

    import subprocess
    cmd = [
        sys.executable, str(sn_repo / "main.py"),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--epochs", str(epochs),
        "--model", "T-DEED",
        "--batch_size", "8",
        "--lr", "1e-4",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def export_checkpoint(checkpoint_path: Path, output_path: Path):
    """Convert trained T-DEED checkpoint to soccer-vision format."""
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    export = {
        "model_state_dict": state.get("state_dict", state.get("model_state_dict", state)),
        "label_map": TEAM_LABEL_TO_SV,
        "num_classes": len(TEAM_ACTION_LABELS),
        "source": "sn-teamspotting/T-DEED",
        "task": "team_ball_action_spotting",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, output_path)
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train T-DEED team ball action spotting")
    parser.add_argument("--data-dir", default="data/sn-bas")
    parser.add_argument("--output-dir", default="training/sn_spotting/runs_teamspotting")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint path for export")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.download_only:
        download_data(data_dir)
    elif args.extract_features:
        extract_features(data_dir, output_dir / "features")
    elif args.train:
        train_tdeed(data_dir, output_dir, args.epochs)
    elif args.export:
        if not args.checkpoint:
            sys.exit("--export requires --checkpoint")
        export_checkpoint(Path(args.checkpoint), output_dir / "teamspotting_sv.pt")
    else:
        print("Specify --download-only, --extract-features, --train, or --export")


if __name__ == "__main__":
    main()
