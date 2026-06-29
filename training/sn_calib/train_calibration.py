"""Train or fine-tune sn-calibration for single-camera field registration.

The pretrained sn-calibration weights (Google Drive) are trained on broadcast
footage. For wide-angle single-camera (e.g. youth soccer), the line detector
may need fine-tuning on frames from our camera setup.

This script:
  1. Downloads sn-calibration pretrained weights (if not already present)
  2. Optionally generates training data from our own footage
  3. Fine-tunes the DeepLabv3 line segmentation on our camera frames
  4. Exports a checkpoint compatible with soccer-vision

Prerequisites:
  pip install SoccerNet
  git clone https://github.com/SoccerNet/sn-calibration.git

Usage:
  # Download pretrained weights
  python train_calibration.py --download-weights

  # Generate training data from our footage (manually annotate lines)
  python train_calibration.py --generate-data --video tests/fixtures/sample_match_a.mp4

  # Fine-tune on our data
  python train_calibration.py --finetune --data-dir data/calib_finetune --epochs 20

  # Test pretrained on our footage
  python train_calibration.py --test --video tests/fixtures/sample_match_a.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


GOOGLE_DRIVE_WEIGHT_ID = "1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM"
WEIGHT_FILENAME = "sn_calibration_weights.pth"


def download_weights(output_dir: Path):
    """Download sn-calibration weights from Google Drive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / WEIGHT_FILENAME

    if weight_path.exists():
        print(f"Weights already exist: {weight_path}")
        return weight_path

    print("Downloading sn-calibration weights from Google Drive...")
    print(f"  File ID: {GOOGLE_DRIVE_WEIGHT_ID}")

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_WEIGHT_ID}"
        gdown.download(url, str(weight_path), quiet=False)
        print(f"  Saved to {weight_path}")
    except ImportError:
        print("Install gdown: pip install gdown")
        print(f"Or download manually from:")
        print(f"  https://drive.google.com/file/d/{GOOGLE_DRIVE_WEIGHT_ID}")
        print(f"Save to: {weight_path}")

    return weight_path


def generate_training_data(video_path: Path, output_dir: Path, n_frames: int = 50):
    """Extract frames for manual annotation of field lines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "images"
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, total // n_frames)

    print(f"Extracting {n_frames} frames from {video_path.name}...")
    extracted = []
    for fn in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        path = frames_dir / f"frame_{fn:08d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted.append({"frame": fn, "timestamp_s": round(fn / fps, 2), "path": str(path)})

    cap.release()

    manifest = output_dir / "manifest.json"
    with open(manifest, "w") as f:
        json.dump({"video": str(video_path), "frames": extracted}, f, indent=2)

    print(f"Extracted {len(extracted)} frames to {frames_dir}")
    print(f"Manifest: {manifest}")
    print()
    print("Next steps for annotation:")
    print("  1. Annotate field lines in each frame (use LabelMe or CVAT)")
    print("  2. Save line segmentation masks as PNG in masks/ alongside images/")
    print("  3. Run: python train_calibration.py --finetune --data-dir", output_dir)


def test_on_video(video_path: Path, weights_dir: Path):
    """Test sn-calibration on our footage to evaluate quality."""
    from soccer_vision.registration.hough import compute_homography, pixel_to_field

    print(f"Testing registration on {video_path.name}...")
    print("(Using Hough fallback — sn-calibration integration is Phase 5)")
    print()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * 10))

    results = []
    for fn in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue

        H, ok = compute_homography(frame)
        ts = fn / fps
        result = {"frame": fn, "timestamp_s": round(ts, 2), "success": ok}

        if ok:
            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
            fx, fy = pixel_to_field(cx, cy, H)
            result["center_field"] = [round(fx, 2), round(fy, 2)]

        results.append(result)
        status = "OK" if ok else "FAIL"
        print(f"  F{fn} t={ts:.1f}s: {status}")

    cap.release()

    n_ok = sum(1 for r in results if r["success"])
    print(f"\nHough registration: {n_ok}/{len(results)} succeeded "
          f"({n_ok/max(1,len(results))*100:.0f}%)")

    if n_ok < len(results) * 0.5:
        print("\nLow success rate suggests fine-tuning sn-calibration would help.")
        print("Run: python train_calibration.py --generate-data --video", video_path)


def finetune(data_dir: Path, output_dir: Path, epochs: int = 20):
    """Fine-tune sn-calibration DeepLabv3 on annotated data."""
    print("Fine-tuning requires sn-calibration repo:")
    print("  git clone https://github.com/SoccerNet/sn-calibration.git")
    print()

    sn_repo = Path("sn-calibration")
    if not sn_repo.exists():
        print("sn-calibration repo not found.")
        print("Clone it and run the training script directly:")
        print(f"  cd sn-calibration && python train.py \\")
        print(f"    --data_dir {data_dir} \\")
        print(f"    --output_dir {output_dir} \\")
        print(f"    --epochs {epochs} \\")
        print(f"    --pretrained {output_dir.parent / WEIGHT_FILENAME}")
        return

    import subprocess
    cmd = [
        sys.executable, str(sn_repo / "train.py"),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--epochs", str(epochs),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train/fine-tune sn-calibration")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--video", help="Video path for generate-data or test")
    parser.add_argument("--data-dir", default="data/calib_finetune")
    parser.add_argument("--weights-dir", default="training/sn_calib/weights")
    parser.add_argument("--output-dir", default="training/sn_calib/runs")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.download_weights:
        download_weights(Path(args.weights_dir))
    elif args.generate_data:
        if not args.video:
            sys.exit("--generate-data requires --video")
        generate_training_data(Path(args.video), Path(args.data_dir))
    elif args.finetune:
        finetune(Path(args.data_dir), Path(args.output_dir), args.epochs)
    elif args.test:
        if not args.video:
            sys.exit("--test requires --video")
        test_on_video(Path(args.video), Path(args.weights_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
