"""
register.py

Optional pre-computation step for Pipeline B.
Runs KpSFR field registration on sampled frames to build a homography cache
that track.py can use instead of the built-in Hough-line fallback.

KpSFR (https://github.com/ericsujw/KpSFR) is a deep-learning field
registration model trained on World Cup footage. It outputs a homography
matrix (image → field template) per frame, and is more robust than Hough
lines when field markings are partially visible.

Setup (one-time):
    conda env create -f https://raw.githubusercontent.com/ericsujw/KpSFR/main/environment.yml
    conda activate kpsfr
    git clone https://github.com/ericsujw/KpSFR kpsfr/
    # Download pretrained weight:
    # https://cgv.cs.nthu.edu.tw/KpSFR_data/model/kpsfr_finetuned.pth
    # → kpsfr/checkpoint/kpsfr_finetuned.pth
    python register.py --video match.mp4 --kpsfr-repo kpsfr/
    # Outputs: registrations/homographies.json

Usage:
    python register.py --video match.mp4 --kpsfr-repo kpsfr/ [--fps 0.5]
    python track.py   --video match.mp4 --homography-cache registrations/homographies.json

Notes:
    - Registration runs at --fps (default 0.5 = once every 2 seconds).
      Dense enough for a 60-min match; interpolation fills the gaps.
    - If KpSFR fails on a frame (field not visible), that frame is skipped
      and the nearest neighbour in the cache is used by track.py.
    - KpSFR requires CUDA 11 + PyTorch 1.9; run this script in the kpsfr
      conda environment, then switch back to your main env for track.py.
    - If registration quality on Veo footage is poor, omit --homography-cache
      and track.py will fall back to Hough lines automatically.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path, fps_target, out_dir):
    """Extract frames at fps_target, return list of (frame_no, timestamp_s, path)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(round(native_fps / fps_target)))
    print(f"Video: {total} frames @ {native_fps:.2f} fps")
    print(f"Extracting every {interval} frames ({fps_target:.2f} fps registration)")

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for fn in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = fn / native_fps
        img_path = out_dir / f"frame_{fn:08d}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        entries.append((fn, ts, str(img_path)))
    cap.release()
    print(f"Extracted {len(entries)} frames to {out_dir}")
    return entries, native_fps


def run_kpsfr(kpsfr_repo: Path, image_paths: list, out_dir: Path):
    """
    Run KpSFR inference on a list of images.
    Returns dict: image_path → 3×3 homography as nested list, or None on failure.

    This calls KpSFR's inference.py via subprocess so it can run in its own
    conda environment without polluting the track.py environment.
    """
    import subprocess

    # Write a temporary file list for KpSFR's target_image argument
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        for p in image_paths:
            f.write(p + "\n")
        img_list_file = f.name

    # Build a minimal inference param text (mimicking inference.txt format)
    param_file = out_dir / "kpsfr_params.txt"
    ckpt = kpsfr_repo / "checkpoint" / "kpsfr_finetuned.pth"
    if not ckpt.exists():
        ckpt = kpsfr_repo / "checkpoint" / "kpsfr.pth"
    if not ckpt.exists():
        print(f"WARNING: KpSFR checkpoint not found at {ckpt.parent}/")
        print("  Download from: https://cgv.cs.nthu.edu.tw/KpSFR_data/model/")
        return {}

    # KpSFR inference.py is called with a param text file; replicate its format
    param_lines = [
        f"--ckpt_path {ckpt}",
        "--train_stage 1",
        "--sfp_finetuned True",
        f"--target_image {' '.join(image_paths[:50])}",  # KpSFR may limit batch
        f"--name register_out",
    ]
    param_file.write_text("\n".join(param_lines))

    inference_script = kpsfr_repo / "inference.py"
    cmd = [sys.executable, str(inference_script), str(param_file)]
    print(f"Running KpSFR inference on {len(image_paths)} frames…")
    result = subprocess.run(cmd, cwd=str(kpsfr_repo),
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("KpSFR stderr:", result.stderr[-2000:])
        print("KpSFR failed — homography cache will be empty.")
        return {}

    # KpSFR saves homographies to checkpoint/<name>/; parse them
    h_dir = kpsfr_repo / "checkpoint" / "register_out"
    homographies = {}
    for img_path in image_paths:
        stem = Path(img_path).stem
        h_file = h_dir / f"{stem}_homography.npy"
        if h_file.exists():
            H = np.load(str(h_file))
            homographies[img_path] = H.tolist()
        else:
            homographies[img_path] = None

    os.unlink(img_list_file)
    return homographies


def build_cache(entries, homographies, native_fps, video_path, out_path):
    """Build and save homographies.json keyed by frame number."""
    cache = {
        "video": str(video_path),
        "fps": native_fps,
        "source": "KpSFR",
        "homographies": [],
    }
    for fn, ts, img_path in entries:
        H = homographies.get(img_path)
        cache["homographies"].append({
            "frame": fn,
            "timestamp_s": round(ts, 3),
            "H": H,  # None if registration failed for this frame
        })
    with open(out_path, "w") as f:
        json.dump(cache, f, indent=2)
    valid = sum(1 for e in cache["homographies"] if e["H"] is not None)
    print(f"\nSaved {valid}/{len(entries)} valid homographies → {out_path}")
    if valid < len(entries) * 0.5:
        print("WARNING: <50% of frames registered successfully.")
        print("  This likely means KpSFR was not trained on this camera style.")
        print("  Use track.py without --homography-cache to fall back to Hough lines.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute KpSFR field registrations for track.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video", required=True)
    p.add_argument("--kpsfr-repo", required=True,
                   help="Path to cloned KpSFR repo")
    p.add_argument("--fps", type=float, default=0.5,
                   help="Registration sample rate in fps (default: 0.5 = every 2s)")
    p.add_argument("--out-dir", default="registrations",
                   help="Output directory (default: registrations/)")
    return p.parse_args()


def main():
    args = parse_args()
    kpsfr_repo = Path(args.kpsfr_repo)
    if not (kpsfr_repo / "inference.py").exists():
        sys.exit(f"KpSFR repo not found at {kpsfr_repo}. "
                 f"Clone: git clone https://github.com/ericsujw/KpSFR {kpsfr_repo}")

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"

    entries, native_fps = extract_frames(args.video, args.fps, frame_dir)
    image_paths = [e[2] for e in entries]

    homographies = run_kpsfr(kpsfr_repo, image_paths, out_dir)

    cache_path = out_dir / "homographies.json"
    build_cache(entries, homographies, native_fps, args.video, cache_path)

    print(f"\nNext: run track.py with:")
    print(f"  python track.py --video {args.video} "
          f"--homography-cache {cache_path}")


if __name__ == "__main__":
    main()
