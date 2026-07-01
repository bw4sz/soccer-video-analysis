"""Train an action spotting model using SoccerNet data + OSL-ActionSpotting framework.

This script:
  1. Downloads SoccerNet action spotting data (labels + pre-extracted features)
  2. Configures an E2E-Spot or NetVLAD-based spotting model
  3. Trains on SoccerNet-v2 labels
  4. Exports a checkpoint usable by soccer-vision

Prerequisites:
  pip install SoccerNet lightning
  git clone https://github.com/OpenSportsLab/OSL-ActionSpotting.git

Usage:
  # Download data first
  python train_action_spotting.py --download-only

  # Train with pre-extracted Baidu features (fastest)
  python train_action_spotting.py --features baidu --epochs 50

  # Train end-to-end from video (requires GPU, slow)
  python train_action_spotting.py --mode e2e --epochs 100 --gpus 1

  # Export trained checkpoint for soccer-vision
  python train_action_spotting.py --export --checkpoint runs/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# SoccerNet action labels → soccer-vision taxonomy
SOCCERNET_TO_SV = {
    "Penalty": "penalty",
    "Kick-off": "kickoff",
    "Goal": "goal",
    "Substitution": "substitution",
    "Offside": "offside",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Clearance": "clearance",
    "Ball out of play": "ball_out",
    "Throw-in": "throw_in",
    "Foul": "foul",
    "Indirect free-kick": "free_kick",
    "Direct free-kick": "free_kick",
    "Corner": "corner_kick",
    "Yellow card": "yellow_card",
    "Red card": "red_card",
    "Yellow->red card": "red_card",
}


def load_soccernet_password(explicit: str | None = None) -> str | None:
    """Resolve the SoccerNet NDA password for restricted downloads.

    Order of precedence: explicit arg -> $SOCCERNET_PASSWORD ->
    .soccernet_token at the repo root. Returns None if none is found
    (public files still download; NDA files will 401).
    """
    import os

    if explicit:
        return explicit.strip()
    env = os.environ.get("SOCCERNET_PASSWORD")
    if env:
        return env.strip()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        token = parent / ".soccernet_token"
        if token.exists():
            return token.read_text().strip()
    return None


def download_soccernet_data(data_dir: Path, password: str | None = None):
    """Download SoccerNet-v2 labels and pre-extracted features."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        sys.exit("Install SoccerNet SDK: pip install SoccerNet")

    downloader = SoccerNetDownloader(LocalDirectory=str(data_dir))
    password = load_soccernet_password(password)
    if password:
        downloader.password = password
    else:
        print("WARNING: no SoccerNet NDA password found "
              "($SOCCERNET_PASSWORD or .soccernet_token); "
              "Baidu features are NDA-restricted and will fail to download.")

    print("Downloading SoccerNet-v2 labels...")
    downloader.downloadGames(
        files=["Labels-v2.json"],
        split=["train", "valid", "test"],
    )

    print("Downloading pre-extracted Baidu features...")
    downloader.downloadGames(
        files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"],
        split=["train", "valid", "test"],
    )

    print(f"Data saved to {data_dir}")


def setup_osl_config(
    data_dir: Path,
    output_dir: Path,
    features: str = "baidu",
    epochs: int = 50,
) -> dict:
    """Generate OSL-ActionSpotting config for training."""
    config = {
        "runner": {
            "type": "trainer",
            "max_epochs": epochs,
            "accelerator": "auto",
            "devices": 1,
        },
        "dataset": {
            "name": "SoccerNetv2",
            "data_path": str(data_dir),
            "features": features,
            "framerate": 2,
            "window_size_sec": 15,
            "num_classes": 17,
        },
        "model": {
            "type": "NetVLAD" if features != "e2e" else "E2ESpot",
            "num_classes": 17,
            "vocab_size": 64,
            "window_size": 15,
            "framerate": 2,
        },
        "optimizer": {
            "type": "Adam",
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },
        "output_dir": str(output_dir),
    }

    config_path = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    return config


def train_with_osl(config_path: Path):
    """Launch OSL-ActionSpotting training."""
    try:
        import lightning as L
    except ImportError:
        sys.exit("Install lightning: pip install lightning")

    print(f"Training with config: {config_path}")
    print("NOTE: If OSL-ActionSpotting is not installed, run:")
    print("  git clone https://github.com/OpenSportsLab/OSL-ActionSpotting.git")
    print("  cd OSL-ActionSpotting && pip install -e .")

    try:
        from oslactionspotting.core.trainer import OSLTrainer

        with open(config_path) as f:
            config = json.load(f)

        trainer = OSLTrainer(config)
        trainer.fit()
        print(f"Training complete. Checkpoint: {config['output_dir']}/best.ckpt")
    except ImportError:
        print("\nOSL-ActionSpotting not installed. Manual training steps:")
        print("  1. git clone https://github.com/OpenSportsLab/OSL-ActionSpotting.git")
        print("  2. cd OSL-ActionSpotting && pip install -e .")
        print(f"  3. python -m oslactionspotting.train --config {config_path}")


def export_checkpoint(checkpoint_path: Path, output_path: Path):
    """Convert a trained OSL checkpoint to soccer-vision format."""
    import torch

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    export = {
        "model_state_dict": state.get("state_dict", state.get("model_state_dict")),
        "label_map": SOCCERNET_TO_SV,
        "num_classes": 17,
        "source": "OSL-ActionSpotting",
        "task": "action_spotting",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, output_path)
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SoccerNet action spotting model")
    parser.add_argument("--data-dir", default="data/soccernet", help="Data directory")
    parser.add_argument("--output-dir", default="training/sn_spotting/runs", help="Output dir")
    parser.add_argument("--download-only", action="store_true", help="Only download data")
    parser.add_argument("--password", help="SoccerNet NDA password (default: "
                        "$SOCCERNET_PASSWORD or .soccernet_token)")
    parser.add_argument("--features", default="baidu",
                        choices=["baidu", "resnet", "e2e"],
                        help="Feature type (default: baidu)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--export", action="store_true", help="Export checkpoint")
    parser.add_argument("--checkpoint", help="Checkpoint path for export")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.download_only:
        download_soccernet_data(data_dir, args.password)
        return

    if args.export:
        if not args.checkpoint:
            sys.exit("--export requires --checkpoint")
        export_checkpoint(
            Path(args.checkpoint),
            output_dir / "action_spotting_sv.pt",
        )
        return

    # Full training flow
    download_soccernet_data(data_dir, args.password)
    config = setup_osl_config(data_dir, output_dir, args.features, args.epochs)
    train_with_osl(output_dir / "config.json")


if __name__ == "__main__":
    main()
