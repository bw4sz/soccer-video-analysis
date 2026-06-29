"""Train DeepLabv3 line segmentation for sn-calibration.

sn-calibration ships a dataloader and pretrained weights but no training
script. This wraps their dataloader with a standard PyTorch training loop
for fine-tuning on wide-angle single-camera footage.

Two modes:
  1. Fine-tune on SoccerNet calibration data (broadcast footage)
  2. Fine-tune on custom annotated frames (our wide-angle footage)

Usage:
  # Download SoccerNet calibration data
  python train.py --download --data-dir /orange/ewhite/soccer-vision/sn-calib

  # Fine-tune from pretrained on SoccerNet data
  python train.py \
    --data-dir /orange/ewhite/soccer-vision/sn-calib \
    --pretrained weights/deeplabv3_sn.pth \
    --epochs 30 --lr 1e-4 --batch-size 8

  # Fine-tune on custom frames (after annotation)
  python train.py \
    --data-dir /orange/ewhite/soccer-vision/custom-calib \
    --pretrained weights/deeplabv3_sn.pth \
    --epochs 20 --lr 5e-5 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50


NUM_LINE_CLASSES = 27  # 26 semantic line classes + background


def download_sn_calib_data(data_dir: Path):
    """Download SoccerNet calibration data."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        sys.exit("pip install SoccerNet")

    downloader = SoccerNetDownloader(LocalDirectory=str(data_dir))
    downloader.downloadDataTask(
        task="calibration-2023",
        split=["train", "valid", "test"],
    )
    print(f"Data saved to {data_dir}")


def build_model(num_classes: int = NUM_LINE_CLASSES, pretrained_path: str | None = None):
    """Build DeepLabv3-ResNet50 for line segmentation."""
    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)

    if pretrained_path:
        print(f"Loading pretrained weights: {pretrained_path}")
        state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)

    return model


def build_sn_dataloader(data_dir: Path, split: str, batch_size: int, num_workers: int):
    """Build dataloader from SoccerNet calibration data.

    SoccerNet calibration data has per-image JSON annotations with
    26 semantic line classes as point coordinates. We convert these to
    segmentation masks for DeepLabv3 training.
    """
    import cv2
    import numpy as np
    from torch.utils.data import Dataset

    class SNCalibDataset(Dataset):
        def __init__(self, root: Path, split: str, img_size: int = 960):
            self.img_size = img_size
            self.samples = []

            split_dir = root / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Split not found: {split_dir}")

            import json
            for ann_path in sorted(split_dir.rglob("*.json")):
                img_path = ann_path.with_suffix(".jpg")
                if not img_path.exists():
                    img_path = ann_path.with_suffix(".png")
                if img_path.exists():
                    self.samples.append((img_path, ann_path))

            print(f"  {split}: {len(self.samples)} samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, ann_path = self.samples[idx]

            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img[:, :, ::-1].copy()  # BGR -> RGB
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            import json
            with open(ann_path) as f:
                ann = json.load(f)

            # Build segmentation mask from line annotations
            mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
            h_orig = ann.get("height", 1080)
            w_orig = ann.get("width", 1920)

            for line_class_id, line_data in enumerate(ann.get("lines", []), start=1):
                if line_class_id >= NUM_LINE_CLASSES:
                    break
                points = line_data.get("points", [])
                if len(points) >= 2:
                    pts = np.array([
                        [int(p["x"] * self.img_size / w_orig),
                         int(p["y"] * self.img_size / h_orig)]
                        for p in points
                    ], dtype=np.int32)
                    cv2.polylines(mask, [pts], False, line_class_id, thickness=3)

            mask = torch.from_numpy(mask)
            return img, mask

    dataset = SNCalibDataset(data_dir, split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float,
    device: str,
    output_dir: Path,
    comet: bool = False,
):
    """Standard PyTorch training loop with validation."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    experiment = None
    if comet:
        try:
            import comet_ml
            experiment = comet_ml.Experiment(project_name="soccer-vision-calibration")
            experiment.log_parameters({"epochs": epochs, "lr": lr, "device": device})
        except ImportError:
            print("comet_ml not installed, skipping logging")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            out = model(images)["out"]
            loss = criterion(out, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} batch {batch_idx}: loss={loss.item():.4f}")

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # Validate
        val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    out = model(images)["out"]
                    val_loss += criterion(out, masks).item()
            val_loss /= max(1, len(val_loader))

        # Log
        lr_now = scheduler.get_last_lr()[0]
        msg = f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.4f}"
        msg += f" lr={lr_now:.2e}"
        print(msg)

        if experiment:
            experiment.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss or 0,
                "lr": lr_now,
            }, epoch=epoch)

        # Save best
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "num_classes": NUM_LINE_CLASSES,
            }, ckpt_path)
            print(f"  Saved best: {ckpt_path} (val_loss={val_loss:.4f})")

    # Save final
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "num_classes": NUM_LINE_CLASSES,
    }, output_dir / "final.pth")


def main():
    parser = argparse.ArgumentParser(description="Train sn-calibration DeepLabv3")
    parser.add_argument("--data-dir", required=True, help="SoccerNet calibration data dir")
    parser.add_argument("--pretrained", help="Pretrained DeepLabv3 weights")
    parser.add_argument("--output-dir", default="training/sn_calib/runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--comet", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_sn_calib_data(Path(args.data_dir))
        return

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(pretrained_path=args.pretrained)
    data_dir = Path(args.data_dir)

    train_loader = build_sn_dataloader(data_dir, "train", args.batch_size, args.num_workers)
    val_loader = build_sn_dataloader(data_dir, "valid", args.batch_size, args.num_workers)

    train(model, train_loader, val_loader, args.epochs, args.lr, device,
          Path(args.output_dir), comet=args.comet)


if __name__ == "__main__":
    main()
